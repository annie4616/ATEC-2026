#!/usr/bin/env bash
# Noise robustness sweep: run the depth student (near-rail KEEP_D=1.2) at several DEPTH_NOISE levels
# sequentially on GPU1. Already have 0.0 (x=15.6, pk79%) and 0.2 (x=16.7, pk100%). Add 0.1/0.3/0.4.
set -e
export CUDA_VISIBLE_DEVICES=1
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nonexistent_depth_noise_sweep
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
DEMO=/hdd0/sohyun/atec_2026_tesk_e/ATEC2026_Simulation_Challenge/demo
export PYTHONPATH="$DEMO:/hdd0/sohyun:$PYTHONPATH"
cd /hdd0/sohyun/tmp
for NZ in 0.1 0.3 0.4; do
  tag=$(echo $NZ | tr -d '.')
  echo "===== DEPTH_NOISE=$NZ (noise$tag) ====="
  DEPTH_NOISE=$NZ /home/sohyun/.conda/envs/env_isaaclab/bin/python course_depth_a2.py \
      --task ATEC-Course-A2Piper --headless --steps 2600 \
      --out_dir /hdd0/sohyun/atec_real_world/depth_student_noise$tag --device cuda:0 \
      2>&1 | grep -E "\[PARKOUR\] (step 2[0-9]00|done|FINISH)|run_max"
  echo "===== done NZ=$NZ ====="
done
echo "ALL NOISE SWEEP DONE"
