#!/usr/bin/env bash
# Validate the DEPTH-camera student on the Task A COURSE (record to bins):
#   - depth encoder (depth_taska_depth_student.pt) reads the raycast depth camera -> scandots latent
#   - frozen teacher actor (policy_taska_depth_student.pt) walks from that latent
#   - direction from the red rail via raw LiDAR (GUIDE_STEER=1, LIDAR_INCLUDE_RAILS=1) -- deployable
#   - ENV_MODE=auto: infer flat/rough from the depth-image scatter (fed to env_idx gait bit)
# GPU1 (user-specified). env_isaaclab conda env.
set -e
export CUDA_VISIBLE_DEVICES=1
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nonexistent_course_depth_student
export TMPDIR=/hdd0/sohyun/tmp
export PYTHONPATH=/hdd0/sohyun:$PYTHONPATH
export GUIDE_STEER=1
export LIDAR_INCLUDE_RAILS=1
export ENV_MODE=rough
# STRONGER rail steering: the depth student drifts left faster than the default gain/clamp can
# correct. Double the lateral gain, widen the heading-correction clamp and the lat-err range.
export GUIDE_KD=0.8
export GUIDE_CLAMP=0.70
export GUIDE_LATERR_CLAMP=0.8
export SPAWN_ALIGN=1
export CROSS_VX=0.5
export ACT_DEBUG=1
DEMO=/hdd0/sohyun/atec_2026_tesk_e/ATEC2026_Simulation_Challenge/demo
export PYTHONPATH="$DEMO:$PYTHONPATH"
cd /hdd0/sohyun/tmp
exec /home/sohyun/.conda/envs/env_isaaclab/bin/python course_depth_a2.py \
    --task ATEC-Course-A2Piper \
    --headless \
    --steps 2600 \
    --out_dir /hdd0/sohyun/atec_real_world/depth_student_course_railstrong \
    --device cuda:0
