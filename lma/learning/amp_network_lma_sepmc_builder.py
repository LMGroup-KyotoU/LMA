
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
from learning.amp_network_lma_builder import AMPLMABuilder
import torch
import torch.nn as nn
import numpy as np
import copy
from phc.learning.pnn import PNN
from rl_games.algos_torch import torch_ext

DISC_LOGIT_INIT_SCALE = 1.0


class AMPLMASEPMCBuilder(AMPLMABuilder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def build(self, name, **kwargs):
        net = AMPLMASEPMCBuilder.Network(self.params, **kwargs)
        return net

    class Network(AMPLMABuilder.Network):

        def __init__(self, params, **kwargs):
            self.input_shape = kwargs['input_shape']
            super().__init__(params, **kwargs)

            self._build_sepmc(self.input_shape)

        def load(self, params):
            super().load(params)

            self.target_action = params['target_action']
            self.task_size = len(self.target_action)
            self.target_action_size = self.target_action.count(True)
            self._sepmc_units = params['sepmc']['units']
            self._sepmc_activation = params['sepmc']['activation']
            self._sepmc_initializer = params['sepmc']['initializer']
            self._sepmc_action_scale = params['sepmc']['action_scale']
            self._sepmc_value_scale = params['sepmc']['value_scale']

            self._sepmc_freeze = params['ia'].get('freeze', False)

        def _build_sepmc(self, input_shape):
            self._sepmc_mlp = nn.Sequential()

            sepmc_input_shape = (input_shape[0] + self.local_op_obs_size, )

            mlp_args = {'input_size': sepmc_input_shape[0], 'units': self._sepmc_units, 'activation': self._sepmc_activation, 'dense_func': torch.nn.Linear}

            self._actor_sepmc_mlp = self._build_mlp(**mlp_args)
            self.mu_sepmc = torch.nn.Linear(self._sepmc_units[-1], self.target_action_size)
            self.mu_sepmc_act = self.activations_factory.create(self.space_config['mu_activation'])

            mlp_init = self.init_factory.create(**self._sepmc_initializer)
            for m in self._actor_sepmc_mlp.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

            if self.separate:
                mlp_critic_args = {'input_size': self.local_op_obs_size + self.value_size, 'units': self._sepmc_units, 'activation': self._sepmc_activation, 'dense_func': torch.nn.Linear}
                self._critic_sepmc_mlp = self._build_mlp(**mlp_critic_args)
                self.value_sepmc = torch.nn.Linear(self._sepmc_units[-1], self.value_size)
                self.value_sepmc_act = self.activations_factory.create(self.value_activation)

                for m in self._critic_sepmc_mlp.modules():
                    if isinstance(m, nn.Linear):
                        mlp_init(m.weight)
                        if getattr(m, "bias", None) is not None:
                            torch.nn.init.zeros_(m.bias)

            if self._sepmc_freeze:
                for param in self._actor_sepmc_mlp.parameters():
                    param.requires_grad = False
                for param in self.mu_sepmc.parameters():
                    param.requires_grad = False
                for param in self._critic_sepmc_mlp.parameters():
                    param.requires_grad = False
                for param in self.value_sepmc.parameters():
                    param.requires_grad = False

            return


        def eval_actor(self, obs_dict):
            # NOTE: only available in continuous mode
            self_obs = obs_dict['obs']
            local_op_obs = obs_dict['local_op_obs']
            task_obs = self_obs[..., -self.task_size:]
            obs = torch.cat([self_obs, local_op_obs], dim=-1)

            sepmc_a_out = self._actor_sepmc_mlp(obs)
            mu_sepmc = self.mu_sepmc_act(self.mu_sepmc(sepmc_a_out))

            # NOTE: modify target goal
            task_obs[..., self.target_action] += self._sepmc_action_scale * mu_sepmc
            obs_dict['obs'][..., -self.task_size:] = task_obs

            mu, sigma = super().eval_actor(obs_dict)
            lma_obs = torch.cat([mu, sigma, local_op_obs], dim = -1)
            a_out = self._actor_lma_mlp(lma_obs)
            mu_ia = self.mu_lma_act(self.mu_ia(a_out))
            mu += self._lma_action_scale * mu_ia

            return mu, sigma

        def eval_critic(self, obs_dict):
            local_op_obs = obs_dict['local_op_obs']

            value = super().eval_critic(obs_dict)
            lma_obs = torch.cat([value, local_op_obs], dim = -1)
            c_out = self._critic_lma_mlp(lma_obs)
            value_ia = self.value_lma_act(self.value_ia(c_out))
            value += self._lma_value_scale * value_ia

            return value
