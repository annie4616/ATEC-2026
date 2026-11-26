#!/bin/bash
set -u
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=/home/yewon/.conda/envs/isaaclab/bin/python
source $RW/eunha_rl/scripts/eunha_env.sh ${EUNHA_TAG:-s0821}
cd $RW
CUDA_VISIBLE_DEVICES=0 timeout --signal=KILL 900 $PY course_mission/scripts/probe_camsync.py \
  --headless --device cuda:0 > $RW/course_mission/logs/camsync.log 2>&1
grep -aE "^\[SYNC\]" $RW/course_mission/logs/camsync.log
grep -a -m1 -A12 "Traceback" $RW/course_mission/logs/camsync.log | head -14
echo "CAMSYNC_DONE"
