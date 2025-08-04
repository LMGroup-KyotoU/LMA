import os
import subprocess
import numpy
import shutil

ITERATION_NUM=3
PLAYER_DICT = {
    'pulse_lma_w_freeze_local_op_obs': {
        'cfg': 'pulse_lma_w_freeze_local_op_obs_self_play.yaml',
        'dstep': 0,
    },
    'pulse_sepmc_local_op_obs': {
        'cfg': 'pulse_sepmc_local_op_obs_self_play.yaml',
        'dstep': 0,
    },
    'pulse_finetune': {
        'cfg': 'pulse_self_play.yaml',
        'dstep': 180000,
    },
    'pulse_lma_sepmc_local_op_obs': {
        'cfg': 'pulse_lma_sepmc_local_op_obs_self_play_tennis2.yaml',
        'dstep': 0,
    },
    'pulse_lma_local_op_obs': {
        'cfg': 'pulse_lma_local_op_obs_self_play_tennis2.yaml',
        'dstep': 0,
    },
}

CFG_DIR = "./lma/data/cfg/learning"
DST_DIR = "./output/HumanoidIm"
TASK_NAME = "tennis2_compete"
DUMMY_WEIGHT_PATH = "./output/HumanoidIm/tennis2_compete/pulse_lma_local_op_obs/0/Humanoid.pth"

START_NUM = 0
END_NUM = 10000
STEP_NUM = 2000
ENV_NUM = 50
# NOTE: all match num = player/games_num * 2 (in yaml file of learning)

def main():
    os.makedirs(os.path.join(DST_DIR, TASK_NAME, "match"), exist_ok=True)
    # dummy weight (Humanoid.pth) for running torch_runner approapriately
    shutil.copy(DUMMY_WEIGHT_PATH, os.path.join(DST_DIR, TASK_NAME, "match"))
    for n1 in range(ITERATION_NUM):
        for n2 in range(ITERATION_NUM):
            for i in range(START_NUM, END_NUM + 1, STEP_NUM):
                for j in range(START_NUM, END_NUM + 1, STEP_NUM):
                    for k1, v1 in PLAYER_DICT.items():
                        for k2, v2 in PLAYER_DICT.items():
                            if (n1 == n2) and (i == j) and (k1 == k2):
                                continue

                            save_csv_name = f"match_self_{k1}_{n1}_{i:08}_op_{k2}_{n2}_{j:08}.csv"
                            if os.path.isfile(os.path.join(DST_DIR, TASK_NAME, "match", k1, str(n1), save_csv_name)):
                                # aleady finished
                                continue

                            log_save_dir = os.path.join(DST_DIR, TASK_NAME, "match", TASK_NAME)
                            result_save_dir = os.path.join(DST_DIR, TASK_NAME, "match", k1, str(n1))
                            os.makedirs(log_save_dir, exist_ok=True)
                            os.makedirs(result_save_dir, exist_ok=True)

                            #weight path
                            k1_num = i + v1["dstep"]
                            k2_num = j + v2["dstep"]
                            self_weight = os.path.join(DST_DIR, TASK_NAME, k1, str(n1), f"Humanoid_{k1_num:08}.pth")
                            op_weight = os.path.join(DST_DIR, TASK_NAME, k2, str(n2), f"Humanoid_{k2_num:08}.pth")

                            command = ["python", "lma/run_hydra.py"]
                            args = [
                                "project_name=SMPLOlympics",
                                "num_agents=2",
                                "learning=hetero_local_op_obs_match",
                                "exp_name={}".format(os.path.join(TASK_NAME, "match")),
                                "env=env_amp_z_lma_tennis_match",
                                f"env.num_envs={ENV_NUM}",
                                "env.task=HumanoidTennis2ZLocalOpObs",
                                "env.enableTaskObs=True",
                                "env.plane.restitution=0.6",
                                "+env.contact_bodies=[\"R_Ankle\",\"L_Ankle\",\"R_Toe\",\"L_Toe\",\"R_Hand\"]",
                                "robot=smpl_humanoid_tennis_righthand",
                                "robot.has_upright_start=True",
                                "robot.real_weight_porpotion_boxes=False",
                                "env.numAMPObsSteps=10",
                                "env.shape_resampling_interval=500000",
                                "env.motion_file=./sample_data/amass_isaac_simple_run_upright_slim.pkl",
                                "env.episode_length=2000",
                                "headless=True",
                                "env.stateInit=Default",
                                "learning.params.config.player_cfg.self.cfg={}".format(os.path.join(CFG_DIR, v1["cfg"])),
                                "learning.params.config.player_cfg.opponent.cfg={}".format(os.path.join(CFG_DIR, v2["cfg"])),
                                "learning.params.config.player_cfg.self.checkpoint={}".format(self_weight),
                                "learning.params.config.player_cfg.opponent.checkpoint={}".format(op_weight),
                                "+learning.params.config.csv_file_name=match_results.csv",
                                "epoch=-1",
                                "no_log=True",
                                "no_virtual_display=True",
                                "test=True",
                            ]

                            command += args

                            print(f"\n\n---------- match {k1}[{n1}][{i:08}] v.s. {k2}[{n2}][{j:08}] ----------")
                            print("self cfg :{}, weight: {}".format(os.path.join(CFG_DIR, v1["cfg"]), self_weight))
                            print("opponent cfg :{}, weight: {}".format(os.path.join(CFG_DIR, v2["cfg"]), op_weight))
                            print("\n\n")

                            subprocess.run(command)

                            shutil.copy2("./match_results.csv", os.path.join(DST_DIR, TASK_NAME, "match", k1, str(n1), save_csv_name))
                            os.remove("./match_results.csv")


if __name__ == '__main__':
    main()
