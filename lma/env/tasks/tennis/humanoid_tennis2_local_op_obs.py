import torch
from isaacgym import gymtorch, gymapi, gymutil
import lma.env.tasks.tennis.humanoid_tennis2 as humanoid_tennis2
from lma.env.tasks.tennis.humanoid_tennis2 import compute_tennis_ball_observations
from phc.utils import torch_utils
import torch.nn.functional as F
import math
import pdb

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import time

class HumanoidTennis2LocalOpObs(humanoid_tennis2.HumanoidTennis2):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        return

    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            obs_size = 21
        return obs_size

    def _compute_task_obs(self, env_ids=None):

        obs_list = []

        if (env_ids is None):
            ball_pos =  self._ball_pos.clone()
            ball_vel = self._ball_vel.clone()
            target_root_pos = self._target_root_pos.clone()
            target_ball_pos = self._ball_targets.clone()
            target_root_pos_op = self._target_root_pos_op.clone()
            target_ball_pos_op = self._ball_targets_op.clone()

        else:
            ball_pos = self._ball_pos[env_ids].clone()
            ball_vel = self._ball_vel[env_ids].clone()
            target_root_pos = self._target_root_pos[env_ids].clone()
            target_ball_pos = self._ball_targets[env_ids].clone()
            target_root_pos_op = self._target_root_pos_op[env_ids].clone()
            target_ball_pos_op = self._ball_targets_op[env_ids].clone()

        for i in range(self.num_agents):
            if (env_ids is None):
                root_states = self._humanoid_root_states_list[i]
                righthand_pos = self._rigid_body_pos_list[i][:, self._righthand_body_id, :].clone()
                righthand_rot = self._rigid_body_rot_list[i][:, self._righthand_body_id, :].clone()
                racket_pos = righthand_pos + quat_rotate(righthand_rot, self._racket_to_hand.clone())

                root_states_op = self._humanoid_root_states_list[(i + 1) % 2]
                righthand_pos_op = self._rigid_body_pos_list[(i + 1) % 2][:, self._righthand_body_id, :].clone()
                righthand_rot_op = self._rigid_body_rot_list[(i + 1) % 2][:, self._righthand_body_id, :].clone()
                racket_pos_op = righthand_pos_op + quat_rotate(righthand_rot_op, self._racket_to_hand.clone())

            else:
                root_states = self._humanoid_root_states_list[i][env_ids]
                righthand_pos = self._rigid_body_pos_list[i][env_ids, self._righthand_body_id, :].clone()
                righthand_rot = self._rigid_body_rot_list[i][env_ids, self._righthand_body_id, :].clone()
                racket_pos = righthand_pos + quat_rotate(righthand_rot, self._racket_to_hand[env_ids].clone())

                root_states_op = self._humanoid_root_states_list[(i + 1) % 2][env_ids]
                righthand_pos_op = self._rigid_body_pos_list[(i + 1) % 2][env_ids, self._righthand_body_id, :].clone()
                righthand_rot_op = self._rigid_body_rot_list[(i + 1) % 2][env_ids, self._righthand_body_id, :].clone()
                racket_pos_op = righthand_pos_op + quat_rotate(righthand_rot_op, self._racket_to_hand[env_ids].clone())

            if i==0:
                obs = compute_tennis_ball_observations(root_states, ball_pos, ball_vel, target_root_pos, racket_pos, target_ball_pos)
                local_op_obs = compute_opponent_observations(root_states, root_states_op, racket_pos_op)
                obs = torch.cat([obs, local_op_obs], dim=-1)
            else:
                obs = compute_tennis_ball_observations(root_states, ball_pos, ball_vel, target_root_pos_op, racket_pos, target_ball_pos_op)
                local_op_obs = compute_opponent_observations(root_states, root_states_op, racket_pos_op)
                obs = torch.cat([obs, local_op_obs], dim=-1)
            obs_list.append(obs)

        return obs_list

# @torch.jit.script
def compute_opponent_observations(root_states, root_states_op, racket_pos_op):
    # type: (Tensor, Tensor, Tensor) -> Tensor

    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    root_pos_op = root_states_op[:, 0:3]

    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    local_pos_op = quat_rotate(heading_rot, root_pos_op - root_pos)
    local_racket_pos_op = quat_rotate(heading_rot, racket_pos_op - root_pos)

    obs = torch.cat([local_pos_op, local_racket_pos_op],dim=-1)

    return obs

class HumanoidTennis2ZLocalOpObs(HumanoidTennis2LocalOpObs):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type, device_id=device_id, headless=headless)
        self.initialize_z_models()
        return

    def step(self, actions):
        super().step_z(actions)
        return

    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)
        super()._setup_character_props_z()
        return
