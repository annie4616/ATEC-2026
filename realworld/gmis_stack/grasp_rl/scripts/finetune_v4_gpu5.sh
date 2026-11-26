#!/bin/bash
# v4 finetunes on GPU 5 (0.2, 0.22): reset-contamination fixes on top of v3 —
# RSI bank keep-out filter + 10-step camera_hit grace. Waits for the v3 queue to finish.
set -u
RW=/hdd0/yewon/ATEC2026_Simulation_Challenge/real_world_preliminary/ATEC2026_Simulation_Challenge
PY=/home/yewon/.conda/envs/isaaclab/bin/python
TRAIN=$RW/grasp_rl/scripts/train_grasp.py
until [ -f "$RW/grasp_rl/detached/final_v3_report.done" ]; do sleep 120; done
export GRASP_BANK_KOFILTER=1
export GRASP_CAMHIT_GRACE=10
cd $RW
CUDA_VISIBLE_DEVICES=5 $PY $TRAIN --headless --device cuda:0 --task ATEC-Grasp-B2Piper-Stage2-v0 \
  --crouch_depth 0.2 --num_envs 1024 --max_iterations 1500 --run_name grasp1h_c020 \
  --width_range 0.030,0.066 --length_range 0.030,0.250 --height_range 0.020,0.130 \
  --resume $RW/grasp_rl/runs/grasp1g_c020_2026-07-27_05-48-03/model_1999.pt \
  > $RW/grasp_rl/detached/train_1h_c020.log 2>&1 &
P1=$!
sleep 90
CUDA_VISIBLE_DEVICES=5 $PY $TRAIN --headless --device cuda:0 --task ATEC-Grasp-B2Piper-Stage2-v0 \
  --crouch_depth 0.22 --num_envs 1024 --max_iterations 1500 --run_name grasp1h_c022 \
  --width_range 0.030,0.066 --length_range 0.030,0.250 --height_range 0.020,0.130 \
  --resume $RW/grasp_rl/runs/grasp1g_c022_2026-07-27_05-49-34/model_1999.pt \
  > $RW/grasp_rl/detached/train_1h_c022.log 2>&1 &
P2=$!
wait $P1 $P2
echo "GPU5_V4_FINETUNES_DONE"
