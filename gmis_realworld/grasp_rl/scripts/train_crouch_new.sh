#!/bin/bash
# Parallel crouch trainings for NEW depths on one GPU (staggered starts).
# Usage: train_crouch_new.sh <gpu> <depth> [<depth> ...]
set -u
RW=/hdd0/yewon/ATEC2026_Simulation_Challenge/real_world_preliminary/ATEC2026_Simulation_Challenge
PY=/home/yewon/.conda/envs/isaaclab/bin/python
GPU=${1:?gpu}; shift
DONOR=$RW/logs/rsl_rl/final_minju/model_9998.pt
cd $RW
PIDS=()
for depth in "$@"; do
  CUDA_VISIBLE_DEVICES=$GPU $PY grasp_rl/scripts/train_crouch_depth.py --headless --device cuda:0 \
    --crouch_h "$depth" --donor "$DONOR" \
    > $RW/grasp_rl/detached/train_crouch_${depth}.log 2>&1 &
  PIDS+=($!)
  sleep 90
done
wait "${PIDS[@]}"
echo "CROUCH_TRAIN_DONE gpu=$GPU depths=$*"
