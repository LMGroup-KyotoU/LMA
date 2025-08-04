import torch
from copy import deepcopy


from rl_games.algos_torch import torch_ext
from phc.utils.running_mean_std import RunningMeanStd
from rl_games.common.player import BasePlayer
import learning.common_player as common_player

from rl_games.common.tr_helpers import unsqueeze_obs
from phc.learning.amp_self_play_agent import construct_op_ck_name

def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action =  action * d + m
    return scaled_action

class AMPSEPMCLocalOpObsSelfPlayPlayerContinuous(common_player.CommonPlayer):
    def __init__(self, config):
        # import pdb;pdb.set_trace()
        self._normalize_amp_input = config.get('normalize_amp_input', True)
        self._normalize_input = config['normalize_input']
        self._disc_reward_scale = config['disc_reward_scale']
        self.opp_agent = 1
        self.task_local_op_obs_size = config['task_local_op_obs_size']
        self.target_action = config['target_action']
        self.target_action_size = self.target_action.count(True)
        self._sepmc_action_scale = config['sepmc_action_scale']

        super().__init__(config)

        return

    def get_action(self, obs, is_determenistic=False):
        self_obs = obs['obs'][..., :-self.task_local_op_obs_size]

        env_num_agents = self.env.task.num_agents
        num_actors = self_obs.shape[0] // env_num_agents

        agent_0_obs = obs['obs'][:num_actors]
        agent_1_obs = obs['obs'][num_actors:]
        agent_0_obs = self._preproc_obs(agent_0_obs)
        agent_1_obs = self._preproc_obs(agent_1_obs)

        if self.has_batch_dimension == False:
            agent_0_obs = unsqueeze_obs(agent_0_obs)
            agent_1_obs = unsqueeze_obs(agent_1_obs)

        agent_0_input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': agent_0_obs,
            'rnn_states': self.states[:num_actors] if self.states is not None else self.states
        }

        agent_1_input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': agent_1_obs,
            'rnn_states': self.states[num_actors:] if self.states is not None else self.states
        }

        if self.opp_agent == 1:
            with torch.no_grad():
                agent_0_res_dict = self.model(agent_0_input_dict)
                agent_1_res_dict = self.opponent_model(agent_1_input_dict)
        else:
            with torch.no_grad():
                agent_0_res_dict = self.opponent_model(agent_0_input_dict)
                agent_1_res_dict = self.model(agent_1_input_dict)

        task_size = len(self.target_action)
        task_obs = self_obs[..., -task_size:]
        task_obs[:num_actors, self.target_action] += self._sepmc_action_scale * agent_0_res_dict['actions']
        task_obs[num_actors:, self.target_action] += self._sepmc_action_scale * agent_1_res_dict['actions']

        task_mu, task_sigma = self.env.task.get_task_policy_output(self_obs)

        agent_0_res_dict['adjusted_actions'] = task_mu[:num_actors]
        agent_1_res_dict['adjusted_actions'] = task_mu[:num_actors]


        if self.opp_agent == 1:
            action = agent_0_res_dict['adjusted_actions']
            opponent_action = agent_1_res_dict['adjusted_actions']
            self.states = agent_0_res_dict['rnn_states'] if agent_0_res_dict['rnn_states'] is None else torch.cat([agent_0_res_dict['rnn_states'], agent_1_res_dict['rnn_states']], dim=0)
        else:
            action = agent_1_res_dict['adjusted_actions']
            opponent_action = agent_0_res_dict['adjucsted_actions']
            self.states = agent_1_res_dict['rnn_states'] if agent_1_res_dict['rnn_states'] is None else torch.cat([agent_1_res_dict['rnn_states'], agent_0_res_dict['rnn_states']], dim=0)

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

        return

    def _build_net(self, config):

        if self.normalize_input:
            if "vec_env" in self.__dict__:
                obs_shape = torch_ext.shape_whc_to_cwh(self.env.task.get_running_mean_size())
            else:
                obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)
            self.running_mean_std = RunningMeanStd(obs_shape).to(self.device)
            self.running_mean_std.eval()
            # config['input_shape'] = obs_shape

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
            config['actions_num'] = self.target_action_size
            config['amp_input_shape'] = self.env.amp_observation_space.shape
            config['task_obs_size_detail'] = self.env.task.get_task_obs_size_detail()
            if self.env.task.has_task:
                config['self_obs_size'] = self.env.task.get_self_obs_size()
                config['local_op_obs_size'] = self.task_local_op_obs_size
                config['task_obs_size'] = self.env.task.get_task_obs_size() - self.task_local_op_obs_size

        else:
            config['actions_num'] = self.target_action_size
            config['amp_input_shape'] = self.env_info['amp_observation_space']
            config['task_obs_size_detail'] = self.env_info['task_obs_size_detail']
            if 'self_obs_size' in self.env_info:
                config['self_obs_size'] = self.env_info['self_obs_size']
                config['local_op_obs_size'] = self.task_local_op_obs_size
                config['task_obs_size'] = self.env_info['task_obs_size']- self.task_local_op_obs_size

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
