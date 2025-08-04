## first person
#### PULSE + AMP
LOG_DIR="output/HumanoidIm/tennis/pulse_amp/tennis"
mkdir -p "$LOG_DIR"
export OMP_NUM_THREADS=1
python phc/run_hydra.py \
    project_name=SMPLOlympics num_agents=1 \
    learning=pulse_amp exp_name=tennis/pulse_amp \
    env=env_amp_z env.num_envs=2048 env.task=HumanoidTennisZ env.enableTaskObs=True env.plane.restitution=0.6 +env.contact_bodies=["R_Ankle","L_Ankle","R_Toe","L_Toe","R_Hand"] \
    robot=smpl_humanoid_tennis_righthand  robot.has_upright_start=True  robot.real_weight_porpotion_boxes=False env.shape_resampling_interval=500000 \
    env.motion_file=./sample_data/video_tennis_afterproc_upright.pkl headless=True env.episode_length=600 \
    headless=True env.stateInit=Default env.numAMPObsSteps=10 no_log=True \
    learning.params.config.max_epochs=180000
