[isaacgym]: https://docs.nvidia.com/isaac/isaacgym/doc/setup.htmlhttps://developer.nvidia.com/isaac-gym
[smpl_olympics]: https://github.com/SMPLOlympics/SMPLOlympics
[smpl]: https://smpl.is.tue.mpg.de/
[smplx]: https://smpl-x.is.tue.mpg.de/download.php
[pulse]: https://github.com/ZhengyiLuo/PULSE
[sepmc]: https://tencent-roboticsx.github.io/lifelike-agility-and-play/
[kl_penalty]: https://www.science.org/doi/10.1126/scirobotics.abo0235
[film]: https://dl.acm.org/doi/abs/10.5555/3504035.3504518


# Latent Motion Adjuster (LMA)

LMA: Latent Motion Adjuster for Physics-based Multi-agent Interaction
- Paper: [\[TMLR\]](https://openreview.net/forum?id=OR5ClOTaWU)

## Installation

We tested our code on Ubuntu 22.04.

1. Install [Isaac Gym][isaacgym]
2. Follow the instruction provides by Isaac Gym and create the conda environment:
    ```bash
    cd <isaacgym repository>
    create_conda_rlgpu.sh
    ```
3. Activate new conda environment:
    ```bash
    conda activate rlgpu
    ```
4. Clone [SMPLOlympics][smpl_olympics] repository in the `third_party/SMPLOlympics`:
    ```bash
    cd <LMA repository>
    git clone https://github.com/SMPLOlympics/SMPLOlympics.git third_party/SMPLOlympics
    ```
    or
    ```bash
    git submodule update --init --recursive
    ```

5. Create symbolic links to utilize the relevant code from [SMPLOlympics][smpl_olympics]:
    ```bash
    cd <LMA repository>
    ln -s third_party/SMPLOlympics/phc ./phc
    ln -s third_party/SMPLOlympics/poselib ./poselib

7. Install the required libraries for each repository:

    ```bash
    cd <LMA repository>/third_party/SMPLOlympics && pip install -r requirement.txt
    cd <isaacgym repository>/python && pip install -e .
    cd <LMA repository> && pip install -r requirement.txt
    ```

8. Download SMPL parameters from [SMPL][smpl] and [SMPLX][smplx]. Put them in the `data/smpl` folder, unzip them into 'data/smpl' folder. For SMPL, please download the v1.1.0 version, which contains the neutral humanoid. Rename the files `basicmodel_neutral_lbs_10_207_0_v1.1.0`, `basicmodel_m_lbs_10_207_0_v1.1.0.pkl`, `basicmodel_f_lbs_10_207_0_v1.1.0.pkl` to `SMPL_NEUTRAL.pkl`, `SMPL_MALE.pkl` and `SMPL_FEMALE.pkl`. For SMPLX, please download the v1.1 version. Rename The file structure should look like this:

    ```
    |-- data
        |-- smpl
            |-- SMPL_FEMALE.pkl
            |-- SMPL_NEUTRAL.pkl
            |-- SMPL_MALE.pkl
            |-- SMPLX_FEMALE.pkl
            |-- SMPLX_NEUTRAL.pkl
            |-- SMPLX_MALE.pkl

    ```

9. Download data and pretrained models from [SMPLOlympics][smpl_olympics]
    ```bash
    cd <LMA repository>
    bash third_party/SMPLOlympics/download_data.sh
    ```

# Training

## Training High-level policy for task stage

We leveraged the provided code from [SMPLOlympics][smpl_olympics] to acquire single-agent skills for table tennis and tennis.
   - table tennis:
      ```bash
      cd <LMA repository>
      bash ./lma/scripts/tabletennis/run_tabletennis.sh
      ```
   - tennis:
      ```bash
      cd <LMA repository>
      bash ./lma/scripts/tennis/run_tennis.sh
      ```

## Training LMA for multi-agent interaction stage

Update the high-level policy weight paths in the files under \<LMA repositry\>/lma/data/cfg/[learning/env].

- \<LMA repositry\>/lma/data/cfg/learning/*.yaml
    ```
    ...
        checkpoint: "<your weight path>"
    ...
    ```

- \<LMA repositry\>/lma/data/cfg/env/*.yaml
    ```
    ...
    task_models:
        ...
            model: "<your weight path>"
        ...
    ```

We provide python scripts to train our models and baselines:
  - method:
    - (i) scratch: [PULSE][pulse] (from scratch)
    - (ii) finetune: [PULSE][pulse] (fine tuning)
    - (iii) expansion: [PULSE][pulse] (fine tuning w/ expansion)
    - (iv) residual: [PULSE][pulse] + residual RL (action)
    - (v) film: [PULSE][pulse] + residual RL ([hidden-layer][film])
    - (vi) kl_penalty: [PULSE][pulse]+[KL][kl_penalty]
    - (vii) lma_w_freeze: [PULSE][pulse] + LMA (ours w/ freeze)
    - (viii) lma: [PULSE][pulse] + LMA (ours w/o freeze)
    - (ix) sepmc: [PULSE][pulse]+[SEPMC][sepmc]
    - (x) lma_w_freeze_sepmc: [PULSE][pulse] + LMA + [SEPMC][sepmc] (ours w freeze)
    - (xi) lma_sepmc: [PULSE][pulse] + LMA + [SEPMC][sepmc] (ours w/o freeze)

All scripts are in the `scripts` folder. Please check the contents of the script and pick one command for training.

   - Cooperation:
      ```bash
      cd <LMA repository>
      python3 ./scripts/<tablettennis2/tennis2>/run_all_<method_name>.py
      # e.g) python3 ./scripts/tennis2/run_all_lma.py
      ```
   - Competition:
      ```bash
      cd <LMA repository>
      python3 ./scripts/<tablettennis2/tennis2>/run_all_<method_name>_compete.py
      # e.g) python3 ./scripts/tennis2/run_all_lma_compete.py
      ```

To evaluate, append `no_virtual_display=True epoch=-1 test=True env.num_envs=1  headless=False ` to the end of the command.

## Run round-robin tournament

To evaluate the strength of the models trained by each method at each step, a round-robin tournament is conducted using the following python program:

    ```bash
    cd <LMA repository>
    python3 ./scripts/<tablettennis2/tennis2>/run_all_match.py
    ```

# Acknowledgement

This code builds upon the following repositories. Please visit the URLs to see the respective LICENSES:
- [SMPLOlympics](https://github.com/SMPLOlympics/SMPLOlympics)
- [SMPL model](https://smpl.is.tue.mpg.de/)
- [PULSE](https://github.com/ZhengyiLuo/PULSE)
