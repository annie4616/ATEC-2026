#!/usr/bin/env bash
# DEPTH-camera encoder distillation (original extreme-parkour method, restored):
#   teacher actor FROZEN (FREEZE_ACTOR=1), train ONLY the depth_encoder as a pure regression onto
#   the teacher's TRUE scandots latent (LATENT_ONLY=1). Exteroception = raycast-simulated depth
#   camera (60x106 -> crop/resize 58x87) mounted at the A2 head-camera pose. GRU kept (NO_RNN=0).
# Teacher = latest overspeed A2Piper 20-DoF (model_47700, auto-loaded via --load_run). GPU1.
# parkour_il env (rsl-rl 2.3.3). num_envs = cfg default 192 (raycast depth is light enough).
set -e
REPO=/hdd0/sohyun/atec_2026_tesk_e/Isaaclab_Parkour
export CUDA_VISIBLE_DEVICES=1
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nonexistent_encoder_depth
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
    --load_run 2026-08-28_13-13-12_teacher_overspeed \
    --run_name encoder_depth
