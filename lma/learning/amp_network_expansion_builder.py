
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


class AMPExpansionBuilder(AMPBuilder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def build(self, name, **kwargs):
        net = AMPExpansionBuilder.Network(self.params, **kwargs)
        return net

    class Network(AMPBuilder.Network):

        def __init__(self, params, **kwargs):
            self.input_shape = kwargs['input_shape']
            self.local_op_obs_size = kwargs['local_op_obs_size']
            self.actions_num = kwargs['actions_num']
            # kwargs['task_obs_size'] -= self.local_op_obs_size

            # NOTE: for TEST (remove for Traning)
            # kwargs['input_shape'] = (self.input_shape[0] + self.local_op_obs_size, )

            super().__init__(params, **kwargs)

            self._build_expansion_layer((self.input_shape[0] + self.local_op_obs_size, ))

        def load(self, params):
            super().load(params)

            self._ia_units = params['ia']['units']
            self._ia_activation = params['ia']['activation']
            self._ia_initializer = params['ia']['initializer']

            self._mlp_freeze = params['mlp']['freeze']

            self._ia_freeze = params['ia'].get('freeze', False)

        def _build_expansion_layer(self, input_shape):
            self._ia_mlp = nn.Sequential()

            mlp_args = {'input_size': input_shape[0], 'units': self._ia_units, 'activation': self._ia_activation, 'dense_func': torch.nn.Linear}

            self._actor_ia_mlp = self._build_mlp(**mlp_args)

            mlp_init = self.init_factory.create(**self._ia_initializer)
            for m in self._actor_ia_mlp.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

            if self.separate:
                mlp_critic_args = {'input_size': input_shape[0], 'units': self._ia_units, 'activation': self._ia_activation, 'dense_func': torch.nn.Linear}
                self._critic_ia_mlp = self._build_mlp(**mlp_critic_args)

                for m in self._critic_ia_mlp.modules():
                    if isinstance(m, nn.Linear):
                        mlp_init(m.weight)
                        if getattr(m, "bias", None) is not None:
                            torch.nn.init.zeros_(m.bias)

            if self._ia_freeze:
                for param in self._actor_ia_mlp.parameters():
                    param.requires_grad = False
                    param.requires_grad = False
                for param in self._critic_ia_mlp.parameters():
                    param.requires_grad = False

            return


        def eval_actor(self, obs_dict):
            # NOTE: only available in continuous mode
            self_obs = obs_dict['obs']
            local_op_obs = obs_dict['local_op_obs']
            obs = torch.cat([self_obs, local_op_obs], dim=-1)

            a_out = self.actor_mlp(obs)
            mu = self.mu_act(self.mu(a_out))
            if self.space_config['fixed_sigma']:
                sigma = mu * 0.0 + self.sigma_act(self.sigma)
            else:
                sigma = self.sigma_act(self.sigma(a_out))

            return mu, sigma

        def eval_critic(self, obs_dict):
            self_obs = obs_dict['obs']
            local_op_obs = obs_dict['local_op_obs']
            obs = torch.cat([self_obs, local_op_obs], dim=-1)

            c_out = self.critic_mlp(obs)
            value = self.value_act(self.value(c_out))

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

            # NOTE: replace o.weight and 0.bias
            # import ipdb; ipdb.set_trace()
            old_in = state_dict['0.weight'].size(1)
            exp_state_dict = self._actor_ia_mlp.state_dict()
            exp_state_dict['0.weight'][:, :old_in] = state_dict['0.weight']
            self._actor_ia_mlp.load_state_dict(exp_state_dict)
            self.actor_mlp[0] = self._actor_ia_mlp[0]

            # state_dict['0.weight'] = exp_state_dict['0.weight']
            # state_dict['0.bias'] = exp_state_dict['0.bias']

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

            # NOTE: replace o.weight and 0.bias
            # import ipdb; ipdb.set_trace()
            old_in = state_dict['0.weight'].size(1)
            exp_state_dict = self._critic_ia_mlp.state_dict()
            exp_state_dict['0.weight'][:, :old_in] = state_dict['0.weight']
            self._critic_ia_mlp.load_state_dict(exp_state_dict)
            self.critic_mlp[0] = self._critic_ia_mlp[0]

            if self._mlp_freeze:
                for param in self.critic_mlp.parameters():
                    param.requires_grad = False
                for param in self.value.parameters():
                    param.requires_grad = False
