import torch
from copy import deepcopy
import pandas as pd
import numpy as np
import collections
import yaml
import os
from gym import spaces

from rl_games.common import tr_helpers
from rl_games.algos_torch import torch_ext
from phc.utils.running_mean_std import RunningMeanStd
from rl_games.common.player import BasePlayer
import learning.common_player as common_player

from rl_games.common.tr_helpers import unsqueeze_obs
from learning.amp_self_play_agent import construct_op_ck_name

from rl_games.common.object_factory import ObjectFactory
from rl_games.algos_torch.model_builder import ModelBuilder

from learning import amp_models
from learning import network_builder
from learning import amp_network_builder
from learning import amp_network_z_builder
from learning import amp_network_lma_builder
from learning import amp_self_play_players
from learning import amp_lma_w_freeze_local_op_obs_self_play_players
from learning import amp_lma_local_op_obs_self_play_players
from learning import amp_sepmc_local_op_obs_self_play_players


def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action =  action * d + m
    return scaled_action

def update_dict(dict_base, other):
    for k, v in other.items():
        if isinstance(v, collections.Mapping) and k in dict_base:
            update_dict(dict_base[k], v)
        else:
            dict_base[k] = v
    return dict_base

class HeteroLocalOpObsPlayerContinuousMatch(common_player.CommonPlayer):
    def __init__(self, config):
        super().__init__(config)

        # register model and network
        self.model_builder = ModelBuilder()
        # model
        self.model_builder.model_factory.register_builder('amp', lambda network, **kwargs: amp_models.ModelAMPContinuous(network))
        # network
        self.model_builder.network_factory.register_builder('a2c', lambda **kwargs: network_builder.A2CBuilder())
        self.model_builder.network_factory.register_builder('amp', lambda **kwargs: amp_network_builder.AMPBuilder())
        self.model_builder.network_factory.register_builder('amp_z', lambda **kwargs: amp_network_z_builder.AMPZBuilder())
        self.model_builder.network_factory.register_builder('amp_lma', lambda **kwargs: amp_network_lma_builder.AMPLMABuilder())

        # register player
        self.player_factory = ObjectFactory()
        self.player_factory.register_builder('amp_self_play', lambda **kwargs: amp_self_play_players.AMPSelfPlayPlayerContinuous(**kwargs))
        self.player_factory.register_builder('amp_lma_w_freeze_local_op_obs_self_play', lambda **kwargs: amp_lma_w_freeze_local_op_obs_self_play_players.AMPLMAFreezeLocalOpObsSelfPlayPlayerContinuous(**kwargs))
        self.player_factory.register_builder('amp_lma_local_op_obs_self_play', lambda **kwargs: amp_lma_local_op_obs_self_play_players.AMPLMALocalOpObsSelfPlayPlayerContinuous(**kwargs))
        self.player_factory.register_builder('amp_sepmc_local_op_obs_self_play', lambda **kwargs: amp_sepmc_local_op_obs_self_play_players.AMPSEPMCLocalOpObsSelfPlayPlayerContinuous(**kwargs))

        # self._normalize_input = config['normalize_input']
        # self.opp_agent = 1
        # self.task_local_op_obs_size = config['task_local_op_obs_size']
        # self._normalize_local_op_obs_input = config.get('normalize_local_op_obs_input', True)

        # load hetero player config
        with open(config["player_cfg"]["self"]["cfg"], 'r') as yml:
            player_cfg  = yaml.safe_load(yml)
            self.player_w_local_op_obs = True
            # TODO: need to discriminate finetune and scratch
            if player_cfg["params"]["algo"]["name"] in ["amp_self_play"]: # finutune doesn't use local op obs
                self.player_w_local_op_obs = False
            base_cfg = deepcopy(config)
            player_cfg["params"]["config"]['reward_shaper'] = tr_helpers.DefaultRewardsShaper(**player_cfg["params"]["config"]['reward_shaper'])
            player_params = update_dict(base_cfg, player_cfg["params"]["config"])
            player_params = self._add_env_info(player_params, self.player_w_local_op_obs)
            # player_params["load_path"] = os.path.join(os.path.dirname(player_params["load_path"]), "self", os.path.basename(player_params["load_path"]))
        with open(config["player_cfg"]["opponent"]["cfg"], 'r') as yml:
            player_cfg_op  = yaml.safe_load(yml)
            self.player_op_w_local_op_obs = True
            if player_cfg_op["params"]["algo"]["name"] in ["amp_self_play"]: # finutune doesn't use local op obs
                self.player_op_w_local_op_obs = False
            base_cfg = deepcopy(config)
            player_cfg_op["params"]["config"]['reward_shaper'] = tr_helpers.DefaultRewardsShaper(**player_cfg_op["params"]["config"]['reward_shaper'])
            player_params_op = update_dict(base_cfg, player_cfg_op["params"]["config"])
            player_params_op = self._add_env_info(player_params_op, self.player_op_w_local_op_obs)
            # player_params_op["load_path"] = os.path.join(os.path.dirname(player_params_op["load_path"]), "opponent", os.path.basename(player_params_op["load_path"]))

        # build model
        model = self.model_builder.load(player_cfg["params"])
        player_params["network"] = model
        model_op = self.model_builder.load(player_cfg_op["params"])
        player_params_op["network"] = model_op

        # create players
        self.player = self.player_factory.create(player_cfg["params"]["algo"]["name"], config=player_params)
        self.player_op = self.player_factory.create(player_cfg_op["params"]["algo"]["name"], config=player_params_op)

        # add env
        self.player.env = self.env
        self.player_op.env = self.env

        # restore each player
        self.player.restore(config["player_cfg"]["self"]["checkpoint"])
        self.player_op.restore(config["player_cfg"]["opponent"]["checkpoint"])

        self.csv_file_name = config.get("csv_file_name", "match_results.csv")

        return

    def _add_env_info(self, config, use_local_op_obs):
        config["env_info"] = self.env_info

        if not use_local_op_obs: # finutune doesn't use local op obs
            obs_size_orig = config["env_info"]["observation_space"].shape[0]
            config["env_info"]["observation_space"] = spaces.Box(np.ones(obs_size_orig - config["task_local_op_obs_size"]) * -np.Inf, np.ones(obs_size_orig - config["task_local_op_obs_size"]) * np.Inf)

        net_config = self._build_net_config()
        config["env_info"]['amp_observation_space'] = net_config['amp_input_shape']
        config["env_info"]['task_obs_size_detail'] = net_config['task_obs_size_detail']
        if self.env.task.has_task:
            config["env_info"]["num_agents"] = self.env.task.num_agents
            config["env_info"]["num_envs"] = self.env.task.num_agents
            config["env_info"]['self_obs_size'] = net_config['self_obs_size']
            config["env_info"]['task_obs_size'] = net_config['task_obs_size']

        return config

    def _build_net_config(self):
        config = super()._build_net_config()
        if (hasattr(self, 'env')):
            config['amp_input_shape'] = self.env.amp_observation_space.shape
            config['task_obs_size_detail'] = self.env.task.get_task_obs_size_detail()
            if self.env.task.has_task:
                config['self_obs_size'] = self.env.task.get_self_obs_size()
                config['task_obs_size'] = self.env.task.get_task_obs_size()

        else:
            config['amp_input_shape'] = self.env_info['amp_observation_space']
            config['task_obs_size_detail'] = self.env_info['task_obs_size_detail']
            if 'self_obs_size' in self.env_info:
                config['self_obs_size'] = self.env_info['self_obs_size']
                config['task_obs_size'] = self.env_info['task_obs_size']


        return config

    def restore(self, fn):
        # NOTE: already restored each player in __init__
        pass

    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_determenistic = self.is_determenistic
        sum_rewards = 0
        all_match_results = []
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        need_init_rnn = self.is_rnn
        for t in range(n_games):
            if games_played >= n_games:
                break

            obs_dict = self.env_reset()

            batch_size = 1
            batch_size = self.get_batch_size(obs_dict['obs'], batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            wins = torch.zeros(self.num_env, dtype=torch.int, device=self.device)
            cr = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

            print_game_res = False

            done_indices = []

            with torch.no_grad():
                for n in range(self.max_steps):

                    obs_dict = self.env_reset(done_indices)

                    if has_masks:
                        masks = self.env.get_action_mask()
                        action = self.get_masked_action(obs_dict, masks, is_determenistic)
                    else:
                        action = self.get_action(obs_dict, is_determenistic)

                    obs_dict, r, done, info = self.env_step(self.env, action)

                    wins += self.env.task.get_win_buf()

                    cr += r
                    steps += 1

                    self._post_step(info)

                    if render:
                        self.env.render(mode='human')
                        time.sleep(self.render_sleep)

                    all_done_indices = done.clone().nonzero(as_tuple=False)
                    done_indices = all_done_indices[::self.num_agents]
                    done_count = len(done_indices)
                    games_played += done_count

                    if done_count > 0:
                        if self.is_rnn:
                            for s in self.states:
                                s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0



                        done_indices_ = done.repeat(self.env_num_agents).clone().nonzero(as_tuple=False)
                        done_count_ = len(done_indices_)

                        cur_rewards = cr[done_indices_].sum().item()
                        self_done = done_indices_[done_indices_ < len(done)]
                        match_results = (wins[done > 0] > 0).detach().cpu().clone().flatten().numpy()
                        wins[done > 0] = 0
                        win_num = np.count_nonzero(match_results)
                        cur_steps = steps[done_indices_].sum().item()
                        # import pdb;pdb.set_trace()
                        cr = cr * (1.0 - done.float().repeat(self.env_num_agents))
                        steps = steps * (1.0 - done.float().repeat(self.env_num_agents))
                        sum_rewards += cur_rewards
                        all_match_results.extend(list(match_results))
                        sum_steps += cur_steps

                        game_res = 0.0
                        if isinstance(info, dict):
                            if 'battle_won' in info:
                                print_game_res = True
                                game_res = info.get('battle_won', 0.5)
                            if 'scores' in info:
                                print_game_res = True
                                game_res = info.get('scores', 0.5)
                        if self.print_stats:
                            if print_game_res:
                                print('reward:', cur_rewards / done_count_, 'steps:', cur_steps / done_count_, 'w:', win_num / len(match_results))
                            else:
                                print('reward:', cur_rewards / done_count_, 'steps:', cur_steps / done_count_, 'w:', win_num / len(match_results))
                        sum_game_res += game_res
                        # if batch_size//self.num_agents == 1 or games_played >= n_games:
                        if games_played >= n_games:
                            break

                    done_indices = done_indices[:, 0]

        print(sum_rewards)
        if print_game_res:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps / games_played * n_game_life, 'winrate:', np.count_nonzero(all_match_results) / len(all_match_results))
        else:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps / games_played * n_game_life, 'winrate:', np.count_nonzero(all_match_results) / len(all_match_results))

        # save match results
        df = pd.DataFrame(data={"win": all_match_results})
        df.to_csv(self.csv_file_name)

        return

    def get_action(self, obs, is_determenistic=False):
        if self.player_w_local_op_obs:
            player_obs = obs
        else:
            player_obs = {"obs": obs["obs"][:, :- self.config["task_local_op_obs_size"]]}
        if self.player_op_w_local_op_obs:
            player_op_obs = obs
        else:
            player_op_obs = {"obs": obs["obs"][:, :- self.config["task_local_op_obs_size"]]}

        action = self.player.get_action(player_obs, is_determenistic)
        action_op = self.player_op.get_action(player_op_obs, is_determenistic)

        num_actors = len(action) // 2

        if num_actors == 1:
            current_action = torch.stack([action[:num_actors], action_op[num_actors:]], dim=0)
        else:
            current_action = torch.cat([action[:num_actors], action_op[num_actors:]], dim=0)

        return current_action
