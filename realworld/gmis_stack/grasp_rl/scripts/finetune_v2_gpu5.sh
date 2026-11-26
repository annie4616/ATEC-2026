#!/bin/bash
# v2 keep-out finetunes for 0.2 and 0.22 in parallel on GPU 5 (staggered), detached-safe.
set -u
RW=/hdd0/yewon/ATEC2026_Simulation_Challenge/real_world_preliminary/ATEC2026_Simulation_Challenge
PY=/home/yewon/.conda/envs/isaaclab/bin/python
TRAIN=$RW/grasp_rl/scripts/train_grasp.py
cd $RW
CUDA_VISIBLE_DEVICES=5 $PY $TRAIN --headless --device cuda:0 --task ATEC-Grasp-B2Piper-Stage2-v0 \
  --crouch_depth 0.2 --num_envs 1024 --max_iterations 2500 --run_name grasp1f_c020 \
  --width_range 0.030,0.066 --length_range 0.030,0.250 --height_range 0.020,0.130 \
  --resume $RW/grasp_rl/runs/grasp1e_c020_2026-07-26_07-43-06/model_2300.pt \
  > $RW/grasp_rl/detached/train_1f_c020.log 2>&1 &
P1=$!
sleep 90
CUDA_VISIBLE_DEVICES=5 $PY $TRAIN --headless --device cuda:0 --task ATEC-Grasp-B2Piper-Stage2-v0 \
  --crouch_depth 0.22 --num_envs 1024 --max_iterations 2500 --run_name grasp1f_c022 \
  --width_range 0.030,0.066 --length_range 0.030,0.250 --height_range 0.020,0.130 \
  --resume $RW/grasp_rl/runs/grasp1e_c022_2026-07-26_07-44-50/model_2300.pt \
  > $RW/grasp_rl/detached/train_1f_c022.log 2>&1 &
P2=$!
wait $P1 $P2
echo "GPU5_V2_FINETUNES_DONE"
