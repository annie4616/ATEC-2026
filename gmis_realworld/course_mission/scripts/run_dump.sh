#!/bin/bash
set -u
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source $RW/eunha_rl/scripts/eunha_env.sh ${EUNHA_TAG:-s0821}
cd $RW
CUDA_VISIBLE_DEVICES=1 timeout --signal=KILL 1200 /home/yewon/.conda/envs/isaaclab/bin/python \
  course_mission/scripts/dump_frame.py --headless --device cuda:0 > $RW/course_mission/logs/dump.log 2>&1
grep -E "\[DUMP\]|Traceback" $RW/course_mission/logs/dump.log | head -5
echo "DUMP_DONE"
