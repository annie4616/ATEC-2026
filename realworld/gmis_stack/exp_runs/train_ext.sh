#!/bin/bash
# Track B: champion-extension grasp training (boost-campaign X1 recipe, verbatim).
# Usage: train_ext.sh <run_name> <seed> <iters> [extra train_grasp args...]
# Env: GPU (default 0), STAGGER seconds to sleep before launch (default 0).
set -u
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-/home/yewon/.conda/envs/isaaclab/bin/python}
NAME=$1; SEED=$2; ITERS=$3; shift 3
sleep "${STAGGER:-0}"
source "$RW/eunha_rl/scripts/eunha_env.sh" s075
export GRASP_PERC_NOISE=rgbl
export GRASP_RUNS_DIR=$RW/exp_runs/runs
mkdir -p "$GRASP_RUNS_DIR" "$RW/exp_runs/logs"
cd "$RW"
GRASP_REAL_NOISE=1 CUDA_VISIBLE_DEVICES=${GPU:-0} $PY grasp_rl/scripts/train_grasp.py --headless --device cuda:0 \
  --task ATEC-Grasp-B2Piper-Stage2-v0 --crouch_depth 0.24 --num_envs 1024 \
  --max_iterations "$ITERS" --run_name "$NAME" --seed "$SEED" \
  --width_range 0.030,0.085 --length_range 0.030,0.250 --height_range 0.020,0.180 \
  --resume eunha_rl/s075/runs/egxs075_c024_2026-08-12_09-19-10/model_900.pt \
  "$@" > "$RW/exp_runs/logs/train_$NAME.log" 2>&1
echo "TRAIN_DONE_$NAME rc=$?"
grep -a 'Mean reward:' "$RW/exp_runs/logs/train_$NAME.log" | tail -3
