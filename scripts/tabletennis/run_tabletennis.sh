# one person
## Hit task
### PULSE
LOG_DIR="output/HumanoidIm/tabletennis/pulse/tabletennis"
mkdir -p "$LOG_DIR"
export OMP_NUM_THREADS=1
python phc/run_hydra.py \
    project_name=SMPLOlympics num_agents=1 \
    learning=pulse exp_name=tabletennis/pulse  \
    env=env_amp_z env.num_envs=2048 env.task=HumanoidPPZ env.enableTaskObs=True +env.contact_bodies=["R_Ankle","L_Ankle","R_Toe","L_Toe","R_Hand"] \
    robot=smpl_humanoid_pp  robot.has_upright_start=True env.shape_resampling_interval=500000 \
    env.motion_file=./sample_data/amass_isaac_simple_run_upright_slim.pkl \
    headless=True env.stateInit=Default no_log=True \
    learning.params.config.max_epochs=100000
