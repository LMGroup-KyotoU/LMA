import os
import subprocess
import numpy
import shutil
import glob

ITERATION_NUM=3
METHOD_DICT = {
    'pulse_lma_w_freeze_local_op_obs': {
        'learning': 'pulse_lma_w_freeze_local_op_obs_self_play',
        'exp_name': 'pulse_lma_w_freeze_local_op_obs',
        'env': 'env_amp_z_lma_tennis',
        'task': 'HumanoidTennis2ZLocalOpObs',
    },
}
CFG_DIR = "./lma/data/cfg/learning"
DST_DIR = "./output/HumanoidIm"
TASK_NAME = "tennis2_compete"


def main():
    for i in range(ITERATION_NUM):
        for k, v in METHOD_DICT.items():

            weight_save_dir = os.path.join(DST_DIR, TASK_NAME, k, str(i))
            log_save_dir = os.path.join(DST_DIR, TASK_NAME, k, str(i), TASK_NAME, k)
            os.makedirs(weight_save_dir, exist_ok=True)
            os.makedirs(log_save_dir, exist_ok=True)

            shutil.copy(os.path.join(CFG_DIR, v["learning"] + ".yaml"), weight_save_dir)

            command = ["python", "lma/run_hydra.py"]
            args = [
                "project_name=SMPLOlympics",
                "num_agents=2",
                "learning={}".format(v["learning"]),
                "exp_name={}/{}/{}".format(TASK_NAME, v["exp_name"], str(i)),
                "env={}".format(v["env"]),
                "env.num_envs=512",
                "env.task={}".format(v["task"]),
                "env.enableTaskObs=True",
                "env.plane.restitution=0.6",
                "+env.contact_bodies=[\"R_Ankle\",\"L_Ankle\",\"R_Toe\",\"L_Toe\",\"R_Hand\"]",
                "robot=smpl_humanoid_tennis_righthand",
                "robot.has_upright_start=True",
                "robot.real_weight_porpotion_boxes=False",
                "env.numAMPObsSteps=10",
                "env.shape_resampling_interval=500000",
                "env.motion_file=./sample_data/video_tennis_afterproc_upright.pkl",
                "env.episode_length=2000",
                "headless=True",
                "env.stateInit=Default",
                "+env.compete_reward=True",
                "no_log=True",
                "+learning.params.config.save_initial_weight=True",
            ]

            command += args

            subprocess.run(command)

if __name__ == '__main__':
    main()
