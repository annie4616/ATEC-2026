#!/bin/bash
# Usage: PROFILE=v2_balanced GPU=0 bash course_mission/scripts/run_mission.sh
set -u
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=/home/yewon/.conda/envs/isaaclab/bin/python
PROFILE=${PROFILE:-v2_balanced}
source $RW/eunha_rl/scripts/eunha_env.sh ${SPEED:-${EUNHA_TAG:-s0821}}
cd $RW
CUDA_VISIBLE_DEVICES=${GPU:-0} timeout --signal=KILL ${TMO:-21600} $PY \
  course_mission/solution_${PROFILE}.py --headless --device cuda:0 ${EXTRA:-} \
  > $RW/course_mission/logs/mission_${PROFILE}.log 2>&1
grep -E "\[MISSION\]|\[LAUNCH\]|=== score" $RW/course_mission/logs/mission_${PROFILE}.log | tail -40
grep -m1 -A14 "Traceback" $RW/course_mission/logs/mission_${PROFILE}.log | head -16
echo "MISSION_DONE_${PROFILE}"
