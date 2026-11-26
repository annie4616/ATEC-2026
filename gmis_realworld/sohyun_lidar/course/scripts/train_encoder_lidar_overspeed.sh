#!/usr/bin/env bash
# LIDAR encoder distill on the OVERSPEED teacher (speed-penalty retrained), same 1D-conv encoder as
# lo31k (LidarScanBackbone) + LATENT_ONLY, teacher actor FROZEN. This is the lidar counterpart of the
# depth version: identical teacher(47700, overspeed) so the speed-overshoot of the old lo31k (which
# used the pre-overspeed staircase teacher) is fixed. Working COPY repo (Isaaclab_Parkour_lidar) so
# the depth original stays intact; logs symlinked to share the teacher checkpoint. GPU1. parkour_il.
set -e
REPO=/hdd0/sohyun/atec_2026_tesk_e/Isaaclab_Parkour_lidar
export CUDA_VISIBLE_DEVICES=1
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nonexistent_lidar_overspeed
export TMPDIR=/hdd0/sohyun/tmp
export PYTHONPATH="$REPO/parkour_tasks:$PYTHONPATH"
export PARKOUR_ROBOT=a2piper
export FREEZE_ACTOR=1
export NO_RNN=0
export LATENT_ONLY=1
export ENCODER_AUX_LAMBDA=1.0
export NUM_STEPS_PER_ENV=24
export LIDAR_BACKBONE=LidarScanBackbone
cd "$REPO"
exec taskset -c 0-191 /home/sohyun/.conda/envs/parkour_il/bin/python scripts/rsl_rl/train.py \
    --task Isaac-Extreme-Parkour-Student-Unitree-Go2-v0 \
    --seed 1 --headless \
    ${SANITY:+--num_envs 4 --max_iterations 2} \
    --load_run 2026-08-28_13-13-12_teacher_overspeed \
    --run_name ${RUN_NAME:-encoder_lidar_overspeed}
