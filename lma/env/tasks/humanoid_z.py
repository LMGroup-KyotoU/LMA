import time
import torch
import phc.env.tasks.humanoid as humanoid
from phc.env.tasks.humanoid_amp import remove_base_rot
from phc.utils import torch_utils
from typing import OrderedDict

from isaacgym.torch_utils import *
from phc.utils.flags import flags
from rl_games.algos_torch import torch_ext
import torch.nn as nn
from phc.learning.pnn import PNN
from collections import deque
from phc.utils.torch_utils import project_to_norm

from learning.network_loader import load_z_encoder, load_z_decoder, load_mlp
from phc.utils.running_mean_std import RunningMeanStd

from easydict import EasyDict as edict

HACK_MOTION_SYNC = False

class HumanoidZ(humanoid.Humanoid):

    def initialize_z_models(self):
        check_points = [torch_ext.load_checkpoint(ck_path) for ck_path in self.models_path]
        ### Loading Distill Model ###
        self.distill_model_config = self.cfg['env']['distill_model_config']
        self.embedding_size_distill = self.distill_model_config['embedding_size']
        self.embedding_norm_distill = self.distill_model_config['embedding_norm']
        self.fut_tracks_distill = self.distill_model_config['fut_tracks']
        self.num_traj_samples_distill = self.distill_model_config['numTrajSamples']
        self.traj_sample_timestep_distill = self.distill_model_config['trajSampleTimestepInv']
        self.fut_tracks_dropout_distill = self.distill_model_config['fut_tracks_dropout']
        self.z_activation = self.distill_model_config['z_activation']
        self.distill_z_type = self.distill_model_config.get("z_type", "sphere")

        self.embedding_partition_distill = self.distill_model_config.get("embedding_partion", 1)
        self.dict_size_distill = self.distill_model_config.get("dict_size", 1)
        ### Loading Distill Model ###

        ### Loading task policy Model ###
        self.use_task_model = self.cfg['env'].get('use_task_model', False)
        if self.use_task_model:
            self.task_policies = edict({})
            for n, (k, v) in enumerate(self.cfg['env']['task_models'].items()):
                task_model_config = edict(v['task_model_config'])
                task_checkpoint = torch_ext.load_checkpoint(v['model'])
                self.task_policies[str(n)] = self.load_task_policy(k, task_checkpoint, task_model_config)
        ### Loading task policy Model ###


        self.z_all = self.cfg['env'].get("z_all", False)

        self.use_vae_prior_loss = self.cfg['env'].get("use_vae_prior_loss", False)
        self.use_vae_prior = self.cfg['env'].get("use_vae_prior", False)
        self.use_vae_fixed_prior = self.cfg['env'].get("use_vae_fixed_prior", False)
        self.use_vae_sphere_prior = self.cfg['env'].get("use_vae_sphere_prior", False)
        self.use_vae_sphere_posterior = self.cfg['env'].get("use_vae_sphere_posterior", False)

        self.decoder = load_z_decoder(check_points[0], activation = self.z_activation, z_type = self.distill_z_type, device = self.device)
        self.encoder = load_z_encoder(check_points[0], activation = self.z_activation, z_type = self.distill_z_type, device = self.device)
        self.power_acc = torch.zeros((self.num_envs, 2)).to(self.device)
        self.power_usage_coefficient = self.cfg["env"].get("power_usage_coefficient", 0.005)

        self.running_mean, self.running_var = check_points[-1]['running_mean_std']['running_mean'], check_points[-1]['running_mean_std']['running_var']

        if self.use_task_model:
            gt_num = len(self.task_policies)
        else:
            gt_num = 1

        if self.save_kin_info:
            self.kin_dict = OrderedDict()
            self.kin_dict.update({
                "gt_mus": torch.zeros([self.num_envs,self.cfg['env'].get("embedding_size", 256) * gt_num]),
                "gt_sigmas": torch.zeros([self.num_envs,self.cfg['env'].get("embedding_size", 256) * gt_num]),
                # "gt_z": torch.zeros([self.num_envs,self.cfg['env'].get("embedding_size", 256) * gt_num]),
                }) # current root pos + root for future aggergration

    def get_task_policy_num(self):
        return(len(self.task_policies))

    def _setup_character_props_z(self):
        self._num_actions = self.cfg['env'].get("embedding_size", 256)
        return

    def get_task_obs_size_detail_z(self):
        task_obs_detail = OrderedDict()

        ### For Z
        task_obs_detail['proj_norm'] = self.cfg['env'].get("proj_norm", True)
        task_obs_detail['embedding_norm'] = self.cfg['env'].get("embedding_norm", 3)
        task_obs_detail['embedding_size'] = self.cfg['env'].get("embedding_size", 256)
        task_obs_detail['z_readout'] = self.cfg['env'].get("z_readout", False)
        task_obs_detail['z_type'] = self.cfg['env'].get("z_type", "sphere")
        task_obs_detail['num_unique_motions'] = self._motion_lib._num_unique_motions
        return task_obs_detail

    def step_z(self, actions_z):

        # if self.dr_randomizations.get('actions', None):
        #     actions = self.dr_randomizations['actions']['noise_lambda'](actions)
        # if flags.server_mode:
            # t_s = time.time()
        # t_s = time.time()
        with torch.no_grad():
            # Apply trained Model.

            ################ GT-Z ################
            self_obs_size = self.get_self_obs_size()
            if self.obs_v == 2:
                self_obs_size = self_obs_size//self.past_track_steps
                obs_buf = self.obs_buf.view(self.num_envs * self.num_agents, self.past_track_steps, -1)
                curr_obs = obs_buf[:, -1]
                self_obs = ((curr_obs[:, :self_obs_size] - self.running_mean.float()[:self_obs_size]) / torch.sqrt(self.running_var.float()[:self_obs_size] + 1e-05))
            else:
                self_obs = (self.obs_buf[:, :self_obs_size] - self.running_mean.float()[:self_obs_size]) / torch.sqrt(self.running_var.float()[:self_obs_size] + 1e-05)

            if self.distill_z_type == "hyper":
                actions_z = self.decoder.hyper_layer(actions_z)
            if self.distill_z_type == "vq_vae":
                if self.is_discrete:
                    indexes = actions_z
                else:
                    B, F = actions_z.shape
                    indexes = actions_z.reshape(B, -1, self.embedding_size_distill).argmax(dim = -1)
                task_out_proj = self.decoder.quantizer.embedding.weight[indexes.view(-1)]
                print(f"\r {indexes.numpy()[0]}", end = '')
                actions_z = task_out_proj.view(-1, self.embedding_size_distill)
            elif self.distill_z_type == "vae":
                if self.use_vae_prior:
                    z_prior_out = self.decoder.z_prior(self_obs)
                    prior_mu = self.decoder.z_prior_mu(z_prior_out)

                    actions_z = prior_mu + actions_z

                if self.use_vae_sphere_posterior:
                    actions_z = project_to_norm(actions_z, 1, "sphere")
                else:
                    actions_z = project_to_norm(actions_z, self.cfg['env'].get("embedding_norm", 5), "none")

            else:
                actions_z = project_to_norm(actions_z, self.cfg['env'].get("embedding_norm", 5), self.distill_z_type)


            if self.z_all:
                x_all = self.decoder.decoder(actions_z)
            else:
                self_obs = torch.clamp(self_obs, min=-5.0, max=5.0)
                x_all = self.decoder.decoder(torch.cat([self_obs, actions_z], dim = -1))

                # z_prior_out = self.decoder.z_prior(self_obs); prior_mu, prior_log_var = self.decoder.z_prior_mu(z_prior_out), self.decoder.z_prior_logvar(z_prior_out); print(prior_mu.max(), prior_mu.min())
                # print('....')

            actions = x_all

        if self.save_kin_info:
            self.kin_dict['gt_mus'] = self.gt_mus
            self.kin_dict['gt_sigmas'] = self.gt_sigmas

        # actions = x_all[:, 3]  # Debugging

        # apply actions
        self.pre_physics_step(actions)

        # step physics and render each frame
        self._physics_step()

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()
        if flags.server_mode:
            dt = time.time() - t_s
            print(f'\r {1/dt:.2f} fps', end='')

        # dt = time.time() - t_s
        # self.fps.append(1/dt)
        # print(f'\r {np.mean(self.fps):.2f} fps', end='')


        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)

    def load_task_policy(self, task_name, checkpoint, cfg):
        key_name = "a2c_network.actor_mlp"
        activation_func = torch_utils.activation_facotry(cfg.mlp_activation)

        loading_keys = [k for k in checkpoint['model'].keys() if k.startswith(key_name)]
        actor = load_mlp(loading_keys, checkpoint, activation_func, activate_all=True)

        mu_loading_keys = ["a2c_network.mu.weight", 'a2c_network.mu.bias']
        mu = load_mlp(mu_loading_keys, checkpoint, activation_func)
        mu_act = torch_utils.activation_facotry(cfg.mu_activation)()

        if cfg.fixed_sigma:
            sigma = nn.Parameter(torch.ones(cfg.embedding_size, requires_grad=False, dtype=torch.float32) * cfg.fixed_sigma_val, requires_grad=False)
        else:
            sigma_loading_keys = ["a2c_network.sigma.weight", 'a2c_network.sigma.bias']
            sigma = load_mlp(sigma_loading_keys, checkpoint, activation_func)
        sigma_act = torch_utils.activation_facotry(cfg.sigma_activation)()

        actor.to(self.device)
        mu.to(self.device)
        mu_act.to(self.device)
        sigma.to(self.device)
        sigma_act.to(self.device)

        actor.eval()
        mu.eval()
        mu_act.eval()
        if not cfg.fixed_sigma:
            sigma.eval()
        sigma_act.eval()

        if cfg.normalize_input:
            running_mean_std = RunningMeanStd((self.get_self_obs_size() + cfg.task_obs_size, )).to(self.device)
            running_mean_std.load_state_dict(checkpoint['running_mean_std'])

            running_mean_std.eval()

        task_obs = cfg.task_obs if "task_obs" in cfg else None

        return edict({"task_name": task_name, "actor": actor, "mu": mu, "mu_act": mu_act, "sigma": sigma, "sigma_act": sigma_act, "running_mean_std": running_mean_std, "fixed_sigma": cfg.fixed_sigma, "task_obs": task_obs})

    def get_task_policy_output(self, obs):
        if obs.dtype == torch.uint8:
            obs = obs.float() / 255.0

        self.gt_mus = torch.zeros([obs.shape[0], 0]).to(self.device)
        self.gt_sigmas = torch.zeros([obs.shape[0], 0]).to(self.device)

        with torch.no_grad():
            for num, task_policy in self.task_policies.items():
                if task_policy.task_obs:
                    all_task_obs = obs[:, -1 * len(task_policy.task_obs):]
                    task_obs = all_task_obs[:, task_policy.task_obs]
                    obs_proc = torch.cat([obs[:, : -1 * len(task_policy.task_obs)], task_obs], dim=-1)
                    self_obs = task_policy.running_mean_std(obs_proc)
                else:
                    obs_proc = obs[:, :task_policy.running_mean_std.mean_size]
                    self_obs = task_policy.running_mean_std(obs_proc)
                    self_obs = torch.cat([self_obs, obs[:, task_policy.running_mean_std.mean_size:]], dim=-1)

                a_out = task_policy.actor(self_obs)
                mu = task_policy.mu_act(task_policy.mu(a_out))
                if task_policy.fixed_sigma:
                    sigma = mu * 0.0 + task_policy.sigma_act(task_policy.sigma).to(self.device)
                else:
                    sigma = self.task_policy.sigma_act(self.task_policy.sigma(a_out))

                self.gt_mus = torch.cat([self.gt_mus, mu], dim=-1)
                self.gt_sigmas = torch.cat([self.gt_sigmas, sigma], dim=-1)

        return self.gt_mus, self.gt_sigmas

    def get_task_action(self, obs):
        gt_mus, gt_sigmas = self.get_task_policy_output(obs)

        self.gt_actions = torch.zeros([obs.shape[0], 0]).to(self.device)

        # NOTE: temporally assume #task_policy = 1
        assert len(self.task_policies) == 1

        with torch.no_grad():
            for num, task_policy in self.task_policies.items():

                distr = torch.distributions.Normal(gt_mus, torch.exp(gt_sigmas))
                actions_z = distr.sample().squeeze()

                ################ GT-Z ################
                self_obs_size = self.get_self_obs_size()
                if self.obs_v == 2:
                    self_obs_size = self_obs_size//self.past_track_steps
                    obs_buf = self.obs_buf.view(self.num_envs * self.num_agents, self.past_track_steps, -1)
                    curr_obs = obs_buf[:, -1]
                    self_obs = ((curr_obs[:, :self_obs_size] - self.running_mean.float()[:self_obs_size]) / torch.sqrt(self.running_var.float()[:self_obs_size] + 1e-05))
                else:
                    self_obs = (self.obs_buf[:, :self_obs_size] - self.running_mean.float()[:self_obs_size]) / torch.sqrt(self.running_var.float()[:self_obs_size] + 1e-05)

                if self.distill_z_type == "hyper":
                    actions_z = self.decoder.hyper_layer(actions_z)
                if self.distill_z_type == "vq_vae":
                    if self.is_discrete:
                        indexes = actions_z
                    else:
                        B, F = actions_z.shape
                        indexes = actions_z.reshape(B, -1, self.embedding_size_distill).argmax(dim = -1)
                    task_out_proj = self.decoder.quantizer.embedding.weight[indexes.view(-1)]
                    print(f"\r {indexes.numpy()[0]}", end = '')
                    actions_z = task_out_proj.view(-1, self.embedding_size_distill)
                elif self.distill_z_type == "vae":
                    if self.use_vae_prior:
                        z_prior_out = self.decoder.z_prior(self_obs)
                        prior_mu = self.decoder.z_prior_mu(z_prior_out)

                        actions_z = prior_mu + actions_z

                    if self.use_vae_sphere_posterior:
                        actions_z = project_to_norm(actions_z, 1, "sphere")
                    else:
                        actions_z = project_to_norm(actions_z, self.cfg['env'].get("embedding_norm", 5), "none")

                else:
                    actions_z = project_to_norm(actions_z, self.cfg['env'].get("embedding_norm", 5), self.distill_z_type)


                if self.z_all:
                    x_all = self.decoder.decoder(actions_z)
                else:
                    self_obs = torch.clamp(self_obs, min=-5.0, max=5.0)
                    x_all = self.decoder.decoder(torch.cat([self_obs, actions_z], dim = -1))

                actions = x_all

        return actions
