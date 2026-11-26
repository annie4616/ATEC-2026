#!/bin/bash
# Leg joint loading vs hardware limits, walk vs crouch. GPU 0 only.
set -u
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=/home/yewon/.conda/envs/isaaclab/bin/python
LOG=$RW/course_mission/logs/probe_crouch_limits$( [ -n "${TAG:-}" ] && echo _$TAG ).log
if [ -s "$LOG" ]; then echo "REFUSING: $LOG already holds a run"; exit 1; fi
source $RW/eunha_rl/scripts/eunha_env.sh ${EUNHA_TAG:-s0821}
cd $RW
CUDA_VISIBLE_DEVICES=0 timeout --signal=KILL ${TMO:-3600} $PY \
  course_mission/scripts/probe_crouch_limits.py --headless --device cuda:0 \
  --depth ${DEPTH:-0.24} > "$LOG" 2>&1
grep -a "\[LIMITS\]" "$LOG"
grep -a -m1 -A12 "Traceback" "$LOG" | head -14
echo "CROUCH_LIMITS_DONE"
