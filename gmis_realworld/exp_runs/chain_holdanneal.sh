#!/bin/bash
# v3: PN_HOLD annealing chain (2026-09-07). Direct frozen finetuning is flat at
# 0.3-0.5% after two designs (v1 bias ladder, v2 disturb+near-true) because the
# success bonus never fires — the gradient is shaping-only and it plateaus. The
# champ's dose-response (same harness, can): 10Hz 14.8-21% / 2.5Hz 5% / 0.5Hz 0% /
# frozen 0%. So climb THAT curve instead: start where success fires (hold=10),
# raise the hold each rung, resume-chaining checkpoints, and let the 2500-point
# terminal bonus keep firing the whole way to hold=infinity.
#   TAG=chainA GPU=0 SEED=1 bash exp_runs/chain_holdanneal.sh
#   TAG=chainB GPU=1 SEED=5 bash exp_runs/chain_holdanneal.sh
# One chain per GPU (full 1024-env speed). Kill by process GROUP.
set -u
RW=/hdd0/yewon/ATEC2026_Simulation_Challenge/ATEC_gmis
TAG=${TAG:?}; GPU=${GPU:?}; SEED=${SEED:-1}
RUNGS=${RUNGS:-10,20,40,80,160,320,100000}
ITERS_PER=${ITERS_PER:-350}
BIAS=${BIAS:-0.012,0.012,0.010}
# MISSION-FOCUSED object family (default): can 66x115, rind chunks, lying cans.
# The full 0.03-0.25 m family holds the champ at ~0% success even at hold=10 —
# judging rungs by that number killed two viable designs; the mission needs
# exactly can+rind (2+2 = 80 pts of task2). Lunchbox pinch is a separate skill.
export WRANGE=${WRANGE:-0.045,0.070}
export LRANGE=${LRANGE:-0.045,0.130}
export HRANGE=${HRANGE:-0.030,0.120}
RESUME=$RW/eunha_rl/s075/runs/egxs075_c024_2026-08-12_09-19-10/model_900.pt
CLOG=$RW/exp_runs/logs/chain_${TAG}.log
: > "$CLOG"
IFS=',' read -ra HOLDS <<< "$RUNGS"
_FIRST=1
for H in "${HOLDS[@]}"; do
  RTAG=${TAG}_h${H}
  # shrink the champ's oversized exploration std (up to 4.0 on the wrist dims —
  # deterministic eval 15-21% vs 0.5% under sampled rollouts) ONCE, entering the
  # chain; later rungs resume the previous rung's std untouched so PPO's entropy
  # pressure can re-grow it around the recovered behavior.
  if [ "$_FIRST" = "1" ]; then
    export GRASP_RESUME_STD_SCALE=${STDSCALE:-0.3}
  else
    export GRASP_RESUME_STD_SCALE=0
  fi
  _FIRST=0
  echo "=== rung hold=$H stdscale=$GRASP_RESUME_STD_SCALE resume=$RESUME $(date +%H:%M) ===" >> "$CLOG"
  TAG=$RTAG GPU=$GPU SEED=$SEED PN_HOLD=$H PN_BIAS=$BIAS ITERS=$ITERS_PER \
    RESUME=$RESUME bash "$RW/exp_runs/launch_frzft.sh"
  # launch_frzft execs train_grasp in the foreground of this loop iteration, so we
  # arrive here when the rung finishes (or dies). Chain from its newest checkpoint.
  NEXT=$(ls -t "$RW"/exp_runs/runs/${RTAG}_*/model_*.pt 2>/dev/null | head -1)
  if [ -z "$NEXT" ]; then
    echo "rung hold=$H produced no checkpoint — chain stops" >> "$CLOG"
    exit 1
  fi
  # log the rung's closing success rate for the chain overview
  tail -60 "$RW/exp_runs/logs/train_${RTAG}.log" 2>/dev/null \
    | grep -aE "grasp_success" | tail -1 >> "$CLOG"
  RESUME=$NEXT
done
echo "CHAIN_DONE final=$RESUME" >> "$CLOG"
