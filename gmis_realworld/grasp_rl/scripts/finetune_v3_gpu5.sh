#!/bin/bash
# v3 keep-out finetunes (dense arm sampling + shaped depth penalty -8 + camera_hit 2-step)
# for 0.2 and 0.22 in parallel on GPU 5, resumed from the v2 (grasp1f) finals.
set -u
RW=/hdd0/yewon/ATEC2026_Simulation_Challenge/real_world_preliminary/ATEC2026_Simulation_Challenge
PY=/home/yewon/.conda/envs/isaaclab/bin/python
TRAIN=$RW/grasp_rl/scripts/train_grasp.py
cd $RW
CUDA_VISIBLE_DEVICES=5 $PY $TRAIN --headless --device cuda:0 --task ATEC-Grasp-B2Piper-Stage2-v0 \
  --crouch_depth 0.2 --num_envs 1024 --max_iterations 2000 --run_name grasp1g_c020 \
  --width_range 0.030,0.066 --length_range 0.030,0.250 --height_range 0.020,0.130 \
  --resume $RW/grasp_rl/runs/grasp1f_c020_2026-07-26_14-04-07/model_2499.pt \
  > $RW/grasp_rl/detached/train_1g_c020.log 2>&1 &
P1=$!
sleep 90
CUDA_VISIBLE_DEVICES=5 $PY $TRAIN --headless --device cuda:0 --task ATEC-Grasp-B2Piper-Stage2-v0 \
  --crouch_depth 0.22 --num_envs 1024 --max_iterations 2000 --run_name grasp1g_c022 \
  --width_range 0.030,0.066 --length_range 0.030,0.250 --height_range 0.020,0.130 \
  --resume $RW/grasp_rl/runs/grasp1f_c022_2026-07-26_14-05-37/model_2499.pt \
  > $RW/grasp_rl/detached/train_1g_c022.log 2>&1 &
P2=$!
wait $P1 $P2
echo "GPU5_V3_FINETUNES_DONE"
