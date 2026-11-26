#!/bin/bash
set -u
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=/home/yewon/.conda/envs/isaaclab/bin/python
source $RW/eunha_rl/scripts/eunha_env.sh ${EUNHA_TAG:-s0821}
cd $RW
CUDA_VISIBLE_DEVICES=${GPU:-0} timeout --signal=KILL 2400 $PY course_mission/scripts/smoke_mission.py \
  --headless --device cuda:0 --steps ${STEPS:-400} > $RW/course_mission/logs/smoke.log 2>&1
grep -E "\[SMOKE\]" $RW/course_mission/logs/smoke.log | head -40
grep -m1 -A12 "Traceback" $RW/course_mission/logs/smoke.log | head -14
echo "SMOKE_DONE"
