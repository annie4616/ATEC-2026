#!/bin/bash
set -u
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=/home/yewon/.conda/envs/isaaclab/bin/python
source $RW/eunha_rl/scripts/eunha_env.sh ${EUNHA_TAG:-s0821}
cd $RW
CUDA_VISIBLE_DEVICES=${GPU:-0} timeout --signal=KILL 5400 $PY course_mission/scripts/probe_manip.py \
  --headless --device cuda:0 > $RW/course_mission/logs/probe.log 2>&1
grep -E "\[PROBE\]|\[MISSION\]" $RW/course_mission/logs/probe.log | tail -40
grep -m1 -A14 "Traceback" $RW/course_mission/logs/probe.log | head -16
echo "PROBE_DONE"
