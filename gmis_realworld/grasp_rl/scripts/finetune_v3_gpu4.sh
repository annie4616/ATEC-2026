#!/bin/bash
# v3 keep-out finetunes (dense arm sampling + shaped depth penalty -8 + camera_hit 2-step)
# for 0.24 and 0.3 in parallel on GPU 4, resumed from the v2 (grasp1f) finals.
set -u
RW=/hdd0/yewon/ATEC2026_Simulation_Challenge/real_world_preliminary/ATEC2026_Simulation_Challenge
PY=/home/yewon/.conda/envs/isaaclab/bin/python
TRAIN=$RW/grasp_rl/scripts/train_grasp.py
cd $RW
CUDA_VISIBLE_DEVICES=4 $PY $TRAIN --headless --device cuda:0 --task ATEC-Grasp-B2Piper-Stage2-v0 \
  --crouch_depth 0.24 --num_envs 1024 --max_iterations 2000 --run_name grasp1g_c024 \
  --width_range 0.030,0.066 --length_range 0.030,0.250 --height_range 0.020,0.130 \
  --resume $RW/grasp_rl/runs/grasp1f_c024_2026-07-26_14-04-07/model_2499.pt \
  > $RW/grasp_rl/detached/train_1g_c024.log 2>&1 &
P1=$!
sleep 90
CUDA_VISIBLE_DEVICES=4 $PY $TRAIN --headless --device cuda:0 --task ATEC-Grasp-B2Piper-Stage2-v0 \
  --crouch_depth 0.3 --num_envs 1024 --max_iterations 2000 --run_name grasp1g_c030 \
  --width_range 0.030,0.066 --length_range 0.030,0.250 --height_range 0.020,0.130 \
  --resume $RW/grasp_rl/runs/grasp1f_c030_2026-07-26_14-05-38/model_2499.pt \
  > $RW/grasp_rl/detached/train_1g_c030.log 2>&1 &
P2=$!
wait $P1 $P2
echo "GPU4_V3_FINETUNES_DONE"
