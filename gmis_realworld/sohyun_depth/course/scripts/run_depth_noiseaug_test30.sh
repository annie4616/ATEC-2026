#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=1
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nonexistent_depth_noise03
export TMPDIR=/hdd0/sohyun/tmp
export GUIDE_STEER=1
export LIDAR_INCLUDE_RAILS=1
export ENV_MODE=rough
export GUIDE_KD=0.8
export GUIDE_CLAMP=0.70
export GUIDE_LATERR_CLAMP=0.8
export GUIDE_KEEP_D=1.2
export SPAWN_ALIGN=1
export CROSS_VX=0.5
export ACT_DEBUG=1
export DEPTH_NOISE=0.3
export DEPTH_MODEL_TAG=noise_
DEMO=/hdd0/sohyun/atec_2026_tesk_e/ATEC2026_Simulation_Challenge/demo
export PYTHONPATH="$DEMO:/hdd0/sohyun:$PYTHONPATH"
cd /hdd0/sohyun/tmp
exec /home/sohyun/.conda/envs/env_isaaclab/bin/python course_depth_a2.py \
    --task ATEC-Course-A2Piper --headless --steps 2600 \
    --out_dir /hdd0/sohyun/atec_real_world/depth_noiseaug_test30 --device cuda:0
