#!/usr/bin/env bash
# SANITY CHECK for the reverted DEPTH-camera distillation: build the student env with the raycast
# depth camera (60x106 -> crop/resize 58x87), load the latest overspeed A2Piper teacher, and run a
# couple of distill iterations to confirm: (1) depth_camera obs shape is (N,58,87), (2) the
# DepthOnlyFCBackbone58x87 forward runs, (3) FREEZE_ACTOR/LATENT_ONLY path is intact. 4 envs only.
# GPU1 (idle). parkour_il env (rsl-rl 2.3.3). PARKOUR_ROBOT=a2piper (20-DoF, num_prop=77).
set -e
REPO=/hdd0/sohyun/atec_2026_tesk_e/Isaaclab_Parkour
export CUDA_VISIBLE_DEVICES=1
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nonexistent_sanity_depth
export TMPDIR=/hdd0/sohyun/tmp
export PYTHONPATH="$REPO/parkour_tasks:$PYTHONPATH"
export PARKOUR_ROBOT=a2piper
export FREEZE_ACTOR=1
export NO_RNN=0
export LATENT_ONLY=1
export ENCODER_AUX_LAMBDA=1.0
export NUM_STEPS_PER_ENV=24
cd "$REPO"
exec /home/sohyun/.conda/envs/parkour_il/bin/python scripts/rsl_rl/train.py \
    --task Isaac-Extreme-Parkour-Student-Unitree-Go2-v0 \
    --seed 1 \
    --headless \
    --num_envs 4 \
    --max_iterations 2 \
    --load_run 2026-08-28_13-13-12_teacher_overspeed \
    --run_name sanity_depth
