
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
from learning.amp_network_builder import AMPBuilder
import torch
import torch.nn as nn
import numpy as np
import copy
from phc.learning.pnn import PNN
from rl_games.algos_torch import torch_ext

DISC_LOGIT_INIT_SCALE = 1.0


class AMPLMABuilder(AMPBuilder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def build(self, name, **kwargs):
        net = AMPLMABuilder.Network(self.params, **kwargs)
        return net

    class Network(AMPBuilder.Network):

        def __init__(self, params, **kwargs):
            self.local_op_obs_size = kwargs['local_op_obs_size']
            self.actions_num = kwargs['actions_num']
            # kwargs['task_obs_size'] -= self.local_op_obs_size

            super().__init__(params, **kwargs)

            self._build_ia((self.actions_num * 2 + self.local_op_obs_size, ))

        def load(self, params):
            super().load(params)

            self._lma_units = params['ia']['units']
            self._lma_activation = params['ia']['activation']
            self._lma_initializer = params['ia']['initializer']
            self._lma_action_scale = params['ia']['action_scale']
            self._lma_value_scale = params['ia']['value_scale']

            self._mlp_freeze = params['mlp']['freeze']

            self._lma_freeze = params['ia'].get('freeze', False)

        def _build_ia(self, input_shape):
            self._lma_mlp = nn.Sequential()

            mlp_args = {'input_size': input_shape[0], 'units': self._lma_units, 'activation': self._lma_activation, 'dense_func': torch.nn.Linear}

            self._actor_lma_mlp = self._build_mlp(**mlp_args)
            self.mu_ia = torch.nn.Linear(self._lma_units[-1], self.actions_num)
            self.mu_lma_act = self.activations_factory.create(self.space_config['mu_activation'])

            mlp_init = self.init_factory.create(**self._lma_initializer)
            for m in self._actor_lma_mlp.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

            if self.separate:
                mlp_critic_args = {'input_size': self.local_op_obs_size + self.value_size, 'units': self._lma_units, 'activation': self._lma_activation, 'dense_func': torch.nn.Linear}
                self._critic_lma_mlp = self._build_mlp(**mlp_critic_args)
                self.value_ia = torch.nn.Linear(self._lma_units[-1], self.value_size)
                self.value_lma_act = self.activations_factory.create(self.value_activation)

                for m in self._critic_lma_mlp.modules():
                    if isinstance(m, nn.Linear):
                        mlp_init(m.weight)
                        if getattr(m, "bias", None) is not None:
                            torch.nn.init.zeros_(m.bias)

            if self._lma_freeze:
                for param in self._actor_lma_mlp.parameters():
                    param.requires_grad = False
                for param in self.mu_ia.parameters():
                    param.requires_grad = False
                for param in self._critic_lma_mlp.parameters():
                    param.requires_grad = False
                for param in self.value_ia.parameters():
                    param.requires_grad = False



            return


        def eval_actor(self, obs_dict):
            lma_local_op_obs = obs_dict['local_op_obs']

            # NOTE: only available in continuous mode

            mu, sigma = super().eval_actor(obs_dict)
            lma_obs = torch.cat([mu, sigma, lma_local_op_obs], dim = -1)
            a_out = self._actor_lma_mlp(lma_obs)
            mu_ia = self.mu_lma_act(self.mu_ia(a_out))
            mu += self._lma_action_scale * mu_ia

            return mu, sigma

        def eval_critic(self, obs_dict):
            lma_local_op_obs = obs_dict['local_op_obs']

            value = super().eval_critic(obs_dict)
            lma_obs = torch.cat([value, lma_local_op_obs], dim = -1)
            c_out = self._critic_lma_mlp(lma_obs)
            value_ia = self.value_lma_act(self.value_ia(c_out))
            value += self._lma_value_scale * value_ia

            return value

        def load_base_net(self, checkpoint):
            # checkpoint = torch_ext.load_checkpoint(model_path)
            self.load_actor(checkpoint)
            self.load_critic(checkpoint)

        def load_actor(self, checkpoint):
            state_dict = self.actor_mlp.state_dict()
            state_dict['0.weight'].copy_(checkpoint['a2c_network.actor_mlp.0.weight'])
            state_dict['0.bias'].copy_(checkpoint['a2c_network.actor_mlp.0.bias'])
            state_dict['2.weight'].copy_(checkpoint['a2c_network.actor_mlp.2.weight'])
            state_dict['2.bias'].copy_(checkpoint['a2c_network.actor_mlp.2.bias'])
            state_dict['4.weight'].copy_(checkpoint['a2c_network.actor_mlp.4.weight'])
            state_dict['4.bias'].copy_(checkpoint['a2c_network.actor_mlp.4.bias'])
            mu_state_dict = self.mu.state_dict()
            mu_state_dict['weight'].copy_(checkpoint['a2c_network.mu.weight'])
            mu_state_dict['bias'].copy_(checkpoint['a2c_network.mu.bias'])

            if self._mlp_freeze:
                for param in self.actor_mlp.parameters():
                    param.requires_grad = False
                for param in self.mu.parameters():
                    param.requires_grad = False

        def load_critic(self, checkpoint):
            state_dict = self.critic_mlp.state_dict()
            state_dict['0.weight'].copy_(checkpoint['a2c_network.critic_mlp.0.weight'])
            state_dict['0.bias'].copy_(checkpoint['a2c_network.critic_mlp.0.bias'])
            state_dict['2.weight'].copy_(checkpoint['a2c_network.critic_mlp.2.weight'])
            state_dict['2.bias'].copy_(checkpoint['a2c_network.critic_mlp.2.bias'])
            state_dict['4.weight'].copy_(checkpoint['a2c_network.critic_mlp.4.weight'])
            state_dict['4.bias'].copy_(checkpoint['a2c_network.critic_mlp.4.bias'])
            value_state_dict = self.value.state_dict()
            value_state_dict['weight'].copy_(checkpoint['a2c_network.value.weight'])
            value_state_dict['bias'].copy_(checkpoint['a2c_network.value.bias'])

            if self._mlp_freeze:
                for param in self.critic_mlp.parameters():
                    param.requires_grad = False
                for param in self.value.parameters():
                    param.requires_grad = False
