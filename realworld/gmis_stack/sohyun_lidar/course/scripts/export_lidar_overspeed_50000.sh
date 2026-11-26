#!/usr/bin/env bash
# Export the LIDAR overspeed-teacher encoder pair @ model_50000 (latent_loss min ~0.032, beats lo31k 0.038):
#   - teacher actor (frozen overspeed model_47700 weights) -> policy.pt
#   - depth_encoder = LidarScanBackbone (1D conv over 5760 rays) + GRU -> depth_latest.pt
# Uses the lidar working-copy repo so LidarScanBackbone is wired in. Match training flags. GPU1. parkour_il.
set -e
REPO=/hdd0/sohyun/atec_2026_tesk_e/Isaaclab_Parkour_lidar
export CUDA_VISIBLE_DEVICES=1
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nonexistent_export_lidar_ovsp
export TMPDIR=/hdd0/sohyun/tmp
export PYTHONPATH="$REPO/parkour_tasks:$PYTHONPATH"
export PARKOUR_ROBOT=a2piper
export LATENT_ONLY=1
export FREEZE_ACTOR=1
export NO_RNN=0
export LIDAR_BACKBONE=LidarScanBackbone
cd "$REPO"
exec taskset -c 0-191 /home/sohyun/.conda/envs/parkour_il/bin/python scripts/rsl_rl/play_teacheractor_export.py \
    --task Isaac-Extreme-Parkour-Student-Unitree-Go2-Play-v0 \
    --num_envs 4 --headless \
    --checkpoint "$REPO/logs/rsl_rl/unitree_go2_parkour/2026-09-03_07-16-35_encoder_lidar_overspeed/model_50000.pt"
