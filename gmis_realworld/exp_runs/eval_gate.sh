#!/bin/bash
# Standard grasp gate eval, identical to the boost-campaign evrate() protocol:
# rgbl perception noise, no real-noise, 32env x 400step, default object ranges.
# Usage: eval_gate.sh <ckpt(repo-rel or abs)> <depth> <log> [extra play_grasp args...]
set -u
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-/home/yewon/.conda/envs/isaaclab/bin/python}
CKPT=$1; DEPTH=$2; LOG=$3; shift 3
source "$RW/eunha_rl/scripts/eunha_env.sh" s075
export GRASP_PERC_NOISE=rgbl
cd "$RW"
GRASP_REAL_NOISE= CUDA_VISIBLE_DEVICES=${GPU:-0} $PY grasp_rl/scripts/play_grasp.py --headless --device cuda:0 \
  --task ATEC-Grasp-B2Piper-Stage2-v0 --crouch_depth "$DEPTH" --num_envs 32 --steps 400 \
  --checkpoint "$CKPT" "$@" > "$LOG" 2>&1
grep -m1 -oE 'grasp successes \([0-9.]+%' "$LOG" | grep -oE '[0-9.]+' | tail -1
