import torch
from copy import deepcopy


from rl_games.algos_torch import torch_ext
from phc.utils.running_mean_std import RunningMeanStd
from rl_games.common.player import BasePlayer
import learning.common_player as common_player

from rl_games.common.tr_helpers import unsqueeze_obs
from learning.amp_self_play_agent import construct_op_ck_name

def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action =  action * d + m
    return scaled_action

class AMPLMALocalOpObsSelfPlayPlayerContinuous(common_player.CommonPlayer):
    def __init__(self, config):
        # import pdb;pdb.set_trace()
        self._normalize_amp_input = config.get('normalize_amp_input', True)
        self._normalize_input = config['normalize_input']
        self._disc_reward_scale = config['disc_reward_scale']
        self.opp_agent = 1
        self.task_local_op_obs_size = config['task_local_op_obs_size']
        self._normalize_local_op_obs_input = config.get('normalize_local_op_obs_input', True)

        super().__init__(config)

        # self.local_op_obs_shape = (self.task_local_op_obs_size, )
        # self.obs_shape = (self.obs_shape[0] - self.task_local_op_obs_size, )

        self.local_op_obs_buffer = torch.zeros((0, self.task_local_op_obs_size), dtype=torch.float, device=self.device)
        self.action_buffer = torch.zeros((0, self.actions_num), dtype=torch.float, device=self.device)

        return

    def get_action(self, obs, is_determenistic=False):
        obs_orig = obs['obs'][..., :-self.task_local_op_obs_size]
        local_op_obs = obs['obs'][..., -self.task_local_op_obs_size:]
        env_num_agents = self.env.task.num_agents
        num_actors = obs_orig.shape[0] // env_num_agents

        if self.opp_agent == 1:
            self_obs = obs_orig[:num_actors]
            opponent_obs = obs_orig[num_actors:]
            self_obs = self._preproc_obs(self_obs)
            opponent_obs = self._preproc_obs(opponent_obs)

            # if self.has_batch_dimension == False:
            #     self_obs = unsqueeze_obs(self_obs)
            #     opponent_obs = unsqueeze_obs(opponent_obs)

            input_dict = {
                'is_train': False,
                'prev_actions': None,
                'obs': self_obs,
                'local_op_obs': self._preproc_local_op_obs(local_op_obs[:num_actors]),
                'rnn_states': self.states[:num_actors] if self.states is not None else self.states
            }

            opponent_input_dict = {
                'is_train': False,
                'prev_actions': None,
                'obs': opponent_obs,
                'local_op_obs': self._preproc_local_op_obs(local_op_obs[num_actors:]),
                'rnn_states': self.states[num_actors:] if self.states is not None else self.states
            }
        else:
            self_obs = obs_orig[num_actors:]
            opponent_obs = obs_orig[:num_actors]
            self_obs = self._preproc_obs(self_obs)
            opponent_obs = self._preproc_obs(opponent_obs)

            # if self.has_batch_dimension == False:
            #     self_obs = unsqueeze_obs(self_obs)
            #     opponent_obs = unsqueeze_obs(opponent_obs)

            input_dict = {
                'is_train': False,
                'prev_actions': None,
                'obs': self_obs,
                'local_op_obs': local_op_obs[num_actors:],
                'rnn_states': self.states[num_actors:] if self.states is not None else self.states
            }

            opponent_input_dict = {
                'is_train': False,
                'prev_actions': None,
                'obs': opponent_obs,
                'local_op_obs': local_op_obs[:num_actors],
                'rnn_states': self.states[:num_actors] if self.states is not None else self.states
            }


        with torch.no_grad():
            res_dict = self.model(input_dict)
            opponent_res_dict = self.opponent_model(opponent_input_dict)
        mu = res_dict['mus']
        action = res_dict['actions']

        # # temporal process
        # self.local_op_obs_buffer = torch.cat([self.local_op_obs_buffer, input_dict["local_op_obs"]], dim=0)
        # self.action_buffer = torch.cat([self.action_buffer, mu], dim=0)

        opponent_mu = opponent_res_dict['mus']
        opponent_action = opponent_res_dict['actions']

        self.states = res_dict['rnn_states'] if res_dict['rnn_states'] is None else torch.cat([res_dict['rnn_states'], opponent_res_dict['rnn_states']], dim=0)
        if is_determenistic:
            current_action = mu
            current_opponent_action = opponent_mu
        else:
            current_action = action
            current_opponent_action = opponent_action
        if self.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())
            current_opponent_action = torch.squeeze(current_opponent_action.detach())

        if self.opp_agent == 1:
            current_action = torch.cat([current_action, current_opponent_action], dim=0)
        else:
            current_action = torch.cat([current_opponent_action, current_action], dim=0)
        if self.clip_actions:
            return rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))
        else:
            return current_action


    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model.load_state_dict(checkpoint['model'])
        fn_op = construct_op_ck_name(fn)
        checkpoint_op = torch_ext.load_checkpoint(fn_op)
        self.opponent_model.load_state_dict(checkpoint_op['model'])
        if self.normalize_input:
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
            self.opponent_running_mean_std.load_state_dict(checkpoint_op['running_mean_std'])
        if self._normalize_amp_input:
            checkpoint = torch_ext.load_checkpoint(fn)
            self._amp_input_mean_std.load_state_dict(checkpoint['amp_input_mean_std'])
        if self._normalize_local_op_obs_input:
            checkpoint = torch_ext.load_checkpoint(fn)
            self.local_op_obs_running_mean_std.load_state_dict(checkpoint['local_op_obs_running_mean_std'])

        return

    def _build_net(self, config):

        if self.normalize_input:
            if "vec_env" in self.__dict__:
                obs_shape = torch_ext.shape_whc_to_cwh(self.env.task.get_running_mean_size())
                obs_shape = (obs_shape[0] - self.task_local_op_obs_size, )
            else:
                obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)
                obs_shape = (obs_shape[0] - self.task_local_op_obs_size, )
            self.running_mean_std = RunningMeanStd(obs_shape).to(self.device)
            self.running_mean_std.eval()

            self.opponent_running_mean_std = RunningMeanStd(obs_shape).to(self.device)
            self.opponent_running_mean_std.eval()

        config['mean_std'] = self.running_mean_std

        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.opponent_model = self.network.build(config)
        self.opponent_model.to(self.device)
        self.opponent_model.eval()
        self.is_rnn = self.model.is_rnn()

        if self._normalize_amp_input:
            self._amp_input_mean_std = RunningMeanStd(config['amp_input_shape']).to(self.device)
            self._amp_input_mean_std.eval()

        if self._normalize_local_op_obs_input:
            self.local_op_obs_running_mean_std = RunningMeanStd((self.task_local_op_obs_size, )).to(self.device)
            self.local_op_obs_running_mean_std.eval()
        return

    def _eval_critic(self, input):
        input = self._preproc_obs(input)
        return self.model.a2c_network.eval_critic(input)

    def _post_step(self, info):
        super()._post_step(info)
        if (self.env.task.viewer):
            self._amp_debug(info)

        return

    def _eval_task_value(self, input):
        input = self._preproc_obs(input)
        return self.model.a2c_network.eval_task_value(input)

    def _build_net_config(self, adjust_shape=True):
        config = super()._build_net_config()
        if (hasattr(self, 'env')):
            config['amp_input_shape'] = self.env.amp_observation_space.shape
            config['task_obs_size_detail'] = self.env.task.get_task_obs_size_detail()
            if self.env.task.has_task:
                config['self_obs_size'] = self.env.task.get_self_obs_size()
                config['task_obs_size'] = self.env.task.get_task_obs_size()
                config['local_op_obs_size'] = self.task_local_op_obs_size
                if adjust_shape:
                    config['input_shape'] = (config['input_shape'][0] - self.task_local_op_obs_size,)
                    config['task_obs_size'] = self.env.task.get_task_obs_size() - self.task_local_op_obs_size

        else:
            config['amp_input_shape'] = self.env_info['amp_observation_space']
            config['task_obs_size_detail'] = self.env_info['task_obs_size_detail']
            if 'self_obs_size' in self.env_info:
                config['self_obs_size'] = self.env_info['self_obs_size']
                config['task_obs_size'] = self.env_info['task_obs_size']
                config['local_op_obs_size'] = self.task_local_op_obs_size
                if adjust_shape:
                    config['input_shape'] = (config['input_shape'][0] - self.task_local_op_obs_size,)
                    config['task_obs_size'] = self.env_info['task_obs_size'] - self.task_local_op_obs_size


        return config

    def _amp_debug(self, info):
        return

    def _preproc_amp_obs(self, amp_obs):
        if self._normalize_amp_input:
            amp_obs = self._amp_input_mean_std(amp_obs)
        return amp_obs

    def _eval_disc(self, amp_obs):
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.model.a2c_network.eval_disc(proc_amp_obs)

    def _eval_actor(self, input):
        input = self._preproc_obs(input)
        return self.model.a2c_network.eval_actor(input)

    def _preproc_obs(self, obs_batch):

        if type(obs_batch) is dict:
            for k, v in obs_batch.items():
                obs_batch[k] = self._preproc_obs(v)
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0
        if self.normalize_input:
            obs_batch_proc = obs_batch[:, :self.running_mean_std.mean_size]
            obs_batch_out = self.running_mean_std(obs_batch_proc)
            obs_batch = torch.cat([obs_batch_out, obs_batch[:, self.running_mean_std.mean_size:]], dim=-1)

        return obs_batch

    def _preproc_local_op_obs(self, obs_batch):
        if type(obs_batch) is dict:
            for k, v in obs_batch.items():
                obs_batch[k] = self._preproc_local_op_obs(v)
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0

        if self.normalize_input:
            obs_batch_proc = obs_batch[:, :self.local_op_obs_running_mean_std.mean_size]
            obs_batch_out = self.local_op_obs_running_mean_std(obs_batch_proc)  # running through mean std, but do not use its value. use temp
            obs_batch_out = torch.cat([obs_batch_out, obs_batch[:, self.local_op_obs_running_mean_std.mean_size:]], dim=-1)

        return obs_batch_out

    # def _opponent_preproc_obs(self, obs_batch):

    #     if type(obs_batch) is dict:
    #         for k, v in obs_batch.items():
    #             obs_batch[k] = self._opponent_preproc_obs(v)
    #     else:
    #         if obs_batch.dtype == torch.uint8:
    #             obs_batch = obs_batch.float() / 255.0
    #     if self.normalize_input:
    #         obs_batch_proc = obs_batch[:, :self.opponent_running_mean_std.mean_size]
    #         obs_batch_out = self.opponent_running_mean_std(obs_batch_proc)
    #         obs_batch = torch.cat([obs_batch_out, obs_batch[:, self.opponent_running_mean_std.mean_size:]], dim=-1)

    #     return obs_batch

    def _calc_amp_rewards(self, amp_obs):
        disc_r = self._calc_disc_rewards(amp_obs)
        output = {
            'disc_rewards': disc_r
        }
        return output

    def _calc_disc_rewards(self, amp_obs):
        with torch.no_grad():
            disc_logits = self._eval_disc(amp_obs)
            prob = 1 / (1 + torch.exp(-disc_logits))
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.device)))
            disc_r *= self._disc_reward_scale
        return disc_r
