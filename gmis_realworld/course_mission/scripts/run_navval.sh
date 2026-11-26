#!/bin/bash
# Map-free navigation validation, one TAG per configuration.
#   TAG=fuse  FUSE=1 bash scripts/run_navval.sh     # steer phase B on navfuse
#   TAG=ema            bash scripts/run_navval.sh     # steer phase B on the raw EMA
# Phase A is map-steered in both, so the estimator error comparison is apples to apples.
set -u
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=/home/yewon/.conda/envs/isaaclab/bin/python
TAG=${TAG:?set TAG}
LOG=$RW/course_mission/logs/navval_$TAG.log
mkdir -p "$RW/course_mission/logs"
if [ -s "$LOG" ]; then echo "REFUSING: $LOG already holds a run"; exit 1; fi
source $RW/eunha_rl/scripts/eunha_env.sh ${EUNHA_TAG:-s0821}
cd $RW
echo "=== navval $TAG  FUSE=${FUSE:-0} seed=${SEED:-0} steps=${STEPS:-5200} ===" > "$LOG"
CUDA_VISIBLE_DEVICES=0 timeout --signal=KILL ${TMO:-5400} $PY \
  course_mission/scripts/validate_venue_nav.py --headless --device cuda:0 \
  --steps ${STEPS:-5200} --seed ${SEED:-0} ${FUSE:+--fuse} >> "$LOG" 2>&1
grep -a "\[NAV\]" "$LOG" | tail -25
grep -a -m1 -A14 "Traceback" "$LOG" | head -16
echo "NAVVAL_DONE_$TAG"
