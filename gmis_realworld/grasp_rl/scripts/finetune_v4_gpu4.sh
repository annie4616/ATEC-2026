#!/bin/bash
# v4 finetunes on GPU 4 (0.24, 0.3): reset-contamination fixes on top of v3 —
# RSI bank keep-out filter + 10-step camera_hit grace. Waits for the v3 queue to finish.
set -u
RW=/hdd0/yewon/ATEC2026_Simulation_Challenge/real_world_preliminary/ATEC2026_Simulation_Challenge
PY=/home/yewon/.conda/envs/isaaclab/bin/python
TRAIN=$RW/grasp_rl/scripts/train_grasp.py
until [ -f "$RW/grasp_rl/detached/final_v3_report.done" ]; do sleep 120; done
export GRASP_BANK_KOFILTER=1
export GRASP_CAMHIT_GRACE=10
cd $RW
CUDA_VISIBLE_DEVICES=4 $PY $TRAIN --headless --device cuda:0 --task ATEC-Grasp-B2Piper-Stage2-v0 \
  --crouch_depth 0.24 --num_envs 1024 --max_iterations 1500 --run_name grasp1h_c024 \
  --width_range 0.030,0.066 --length_range 0.030,0.250 --height_range 0.020,0.130 \
  --resume $RW/grasp_rl/runs/grasp1g_c024_2026-07-27_05-48-03/model_1999.pt \
  > $RW/grasp_rl/detached/train_1h_c024.log 2>&1 &
P1=$!
sleep 90
CUDA_VISIBLE_DEVICES=4 $PY $TRAIN --headless --device cuda:0 --task ATEC-Grasp-B2Piper-Stage2-v0 \
  --crouch_depth 0.3 --num_envs 1024 --max_iterations 1500 --run_name grasp1h_c030 \
  --width_range 0.030,0.066 --length_range 0.030,0.250 --height_range 0.020,0.130 \
  --resume $RW/grasp_rl/runs/grasp1g_c030_2026-07-27_05-49-34/model_1999.pt \
  > $RW/grasp_rl/detached/train_1h_c030.log 2>&1 &
P2=$!
wait $P1 $P2
echo "GPU4_V4_FINETUNES_DONE"
