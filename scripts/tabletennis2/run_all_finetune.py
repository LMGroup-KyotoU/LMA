import os
import subprocess
import numpy
import shutil
import glob

ITERATION_NUM=3
METHOD_DICT = {
    'pulse_finetune': {
        'learning': 'pulse_self_play',
        'exp_name': 'pulse_finetune',
        'env': 'env_amp_z',
        'task': 'HumanoidPP2Z',
    },
}
CFG_DIR = "./lma/data/cfg/learning"
DST_DIR = "./output/HumanoidIm"
TASK_NAME = "tabletennis2"

INITIAL_WEIGHT_PATH = "./output/HumanoidIm/tabletennis/pulse/Humanoid_00070000.pth"
RESTART_EPOCH_NUM = 70000
MAX_EPOCH_NUM = 80000


def main():
    for i in range(ITERATION_NUM):
        for k, v in METHOD_DICT.items():

            weight_save_dir = os.path.join(DST_DIR, TASK_NAME, k, str(i))
            log_save_dir = os.path.join(DST_DIR, TASK_NAME, k, str(i), TASK_NAME, k)
            os.makedirs(weight_save_dir, exist_ok=True)
            os.makedirs(log_save_dir, exist_ok=True)

            shutil.copy(os.path.join(CFG_DIR, v["learning"] + ".yaml"), weight_save_dir)
            shutil.copy(INITIAL_WEIGHT_PATH, weight_save_dir)

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
                "+env.contact_bodies=[\"R_Ankle\",\"L_Ankle\",\"R_Toe\",\"L_Toe\",\"R_Hand\"]",
                "robot=smpl_humanoid_pp",
                "robot.has_upright_start=True",
                "env.shape_resampling_interval=500000",
                "env.motion_file=./sample_data/pingpong1after_upright.pkl",
                "headless=True",
                "env.stateInit=Default",
                "no_log=True",
                "epoch={}".format(RESTART_EPOCH_NUM),
                "learning.params.config.max_epochs={}".format(MAX_EPOCH_NUM)
            ]

            command += args

            print(command)

            subprocess.run(command)

            # for p in glob.glob(DST_DIR + '/' + v["exp_name"] + '/*.pth'):
            #     if os.path.isfile(p):
            #         shutil.copy(p, weight_save_dir)

if __name__ == '__main__':
    main()
