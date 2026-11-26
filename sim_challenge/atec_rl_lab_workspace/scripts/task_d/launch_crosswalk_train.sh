#!/bin/bash
# Climb-specific RL: train the WALK-ACROSS crossing policy (taskd_crosswalk_env_cfg,
# finalized 2026-06-28 15:51). This is the +20 path: robot steps down onto the
# settled box, walks across the pit, and CLIMBS the 0.4 m far rim back onto flat
# ground (the milestone that stayed 0 in every prior run). Warm-starts from the
# crossing-rough checkpoint cwk_rough_ws/model_1999.pt (same obs/action interface).
#
# GPU 2 (user preference). TMPDIR -> /hdd0/sohyun/tmp (default /tmp/isaaclab is
# owned by another user -> PermissionError). 4096 envs, half the cores (<=192).
set -e
cd /hdd0/sohyun/atec_2026_tesk_e/ATEC2026_Simulation_Challenge

source /home/sohyun/.conda/etc/profile.d/conda.sh
conda activate env_isaaclab

export PYTHONPATH=/hdd0/sohyun:$PYTHONPATH
export TMPDIR=/hdd0/sohyun/tmp
mkdir -p "$TMPDIR" logs
export CUDA_VISIBLE_DEVICES=2

PY=/home/sohyun/.conda/envs/env_isaaclab/bin/python
LOG=logs/train_crosswalk_$(date +%H%M%S).log

taskset -c 0-191 $PY scripts/rsl_rl/train.py \
  --task ATEC-TaskD-RL-CrossWalk-Unitree-B2-v0 \
  --num_envs 4096 \
  --max_iterations 5000 \
  --seed 1 \
  --headless \
  --resume --load_run cwk_rough_ws --checkpoint model_1999.pt \
  > "$LOG" 2>&1
