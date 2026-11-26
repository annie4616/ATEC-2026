#!/bin/bash
# Waits for verify_gpu4 to finish, then runs the v2 keep-out finetunes for 0.24 and 0.3
# in parallel on GPU 4 (staggered). Fully detached-safe (no session dependencies).
set -u
RW=/hdd0/yewon/ATEC2026_Simulation_Challenge/real_world_preliminary/ATEC2026_Simulation_Challenge
PY=/home/yewon/.conda/envs/isaaclab/bin/python
TRAIN=$RW/grasp_rl/scripts/train_grasp.py
until [ -f $RW/grasp_rl/detached/verify_gpu4.done ]; do sleep 60; done
cd $RW
CUDA_VISIBLE_DEVICES=4 $PY $TRAIN --headless --device cuda:0 --task ATEC-Grasp-B2Piper-Stage2-v0 \
  --crouch_depth 0.24 --num_envs 1024 --max_iterations 2500 --run_name grasp1f_c024 \
  --width_range 0.030,0.066 --length_range 0.030,0.250 --height_range 0.020,0.130 \
  --resume $RW/grasp_rl/runs/grasp1e_c024_2026-07-26_07-43-09/model_2400.pt \
  > $RW/grasp_rl/detached/train_1f_c024.log 2>&1 &
P1=$!
sleep 90
D30A=$(ls -td $RW/grasp_rl/runs/grasp1e_c030_* | head -1)/model_2499.pt
CUDA_VISIBLE_DEVICES=4 $PY $TRAIN --headless --device cuda:0 --task ATEC-Grasp-B2Piper-Stage2-v0 \
  --crouch_depth 0.3 --num_envs 1024 --max_iterations 2500 --run_name grasp1f_c030 \
  --width_range 0.030,0.066 --length_range 0.030,0.250 --height_range 0.020,0.130 \
  --resume "$D30A" \
  > $RW/grasp_rl/detached/train_1f_c030.log 2>&1 &
P2=$!
wait $P1 $P2
echo "GPU4_V2_FINETUNES_DONE"
