#!/usr/bin/env bash
# Export the DEPTH matched pair from the depth-distill checkpoint (model_50800, latent_loss ~0.026):
#   - teacher actor (frozen model_47700 weights) -> policy.pt
#   - depth_encoder = RecurrentDepthBackbone wrapping DepthOnlyFCBackbone58x87 -> depth_latest.pt
#     (forward(depth_image_58x87, proprioception77)).
# Backbone is the cfg default DepthOnlyFCBackbone58x87 (no LIDAR_BACKBONE env). Match the training
# flags so the loaded modules line up. GPU1. parkour_il env.
set -e
REPO=/hdd0/sohyun/atec_2026_tesk_e/Isaaclab_Parkour
export CUDA_VISIBLE_DEVICES=1
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nonexistent_export_depth_50800
export TMPDIR=/hdd0/sohyun/tmp
export PYTHONPATH="$REPO/parkour_tasks:$PYTHONPATH"
export PARKOUR_ROBOT=a2piper
export LATENT_ONLY=1
export FREEZE_ACTOR=1
export NO_RNN=0
cd "$REPO"
exec taskset -c 0-191 /home/sohyun/.conda/envs/parkour_il/bin/python scripts/rsl_rl/play_teacheractor_export.py \
    --task Isaac-Extreme-Parkour-Student-Unitree-Go2-Play-v0 \
    --num_envs 4 --headless \
    --checkpoint "$REPO/logs/rsl_rl/unitree_go2_parkour/2026-09-01_08-37-52_encoder_depth/model_50800.pt"
