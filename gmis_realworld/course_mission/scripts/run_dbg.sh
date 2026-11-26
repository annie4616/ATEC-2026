#!/bin/bash
set -u
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=/home/yewon/.conda/envs/isaaclab/bin/python
source $RW/eunha_rl/scripts/eunha_env.sh ${EUNHA_TAG:-s0821}
cd $RW
CUDA_VISIBLE_DEVICES=${GPU:-0} timeout --signal=KILL 2400 $PY course_mission/scripts/debug_detect.py \
  --headless --device cuda:0 > $RW/course_mission/logs/dbg.log 2>&1
grep -E "\[DBG\]" $RW/course_mission/logs/dbg.log
grep -m1 -A14 "Traceback" $RW/course_mission/logs/dbg.log | head -16
echo "DBG_DONE"
