#!/usr/bin/env bash
# Overspeed lidar models with STRONGER rail steering to counter the latent-driven rightward yaw veer.
# KD 0.4->0.8, CLAMP 0.40->0.70 (depth-proven strong values), KEEP_D 0.9->1.2 (walk nearer course
# center). GUIDE_STEER=1. Tests whether stronger parallel-rail correction keeps the robot on the
# corridor (and whether better heading also helps it clear the slope that stalled it). GPU1.
set -e
RW=/hdd0/yewon/ATEC2026_Simulation_Challenge/real_world_preliminary/ATEC2026_Simulation_Challenge
export CUDA_VISIBLE_DEVICES=1
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nonexistent_strongrail_lidar
export TMPDIR=/hdd0/sohyun/tmp
export PYTHONPATH="$RW/eunha_rl:$RW/source/atec_rl_lab:$RW:/hdd0/sohyun:$PYTHONPATH"
export ACT_DEBUG=1
export ENV_MODE=auto
export INJECT_ATTITUDE=0
export SPEED_CLAMP=0
export CROSS_VX=0.5
export LIDAR_INCLUDE_RAILS=1
export GUIDE_STEER=1
export GUIDE_KEEP_D=1.2
export GUIDE_KD=0.8
export GUIDE_CLAMP=0.70
export GUIDE_LATERR_CLAMP=0.8
export USE_DEPTH_YAW=0
export LIDAR_MODEL_TAG=overspeed_
export WARP_ENCODER=0
cd "$RW"
exec /home/sohyun/.conda/envs/env_isaaclab/bin/python /hdd0/sohyun/tmp/course_student_a2.py \
    --headless --device cuda:0 --task ATEC-Course-A2Piper \
    --steps 2600 --out_dir /hdd0/sohyun/atec_real_world/lidar_overspeed_strongrail
