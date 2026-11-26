#!/bin/bash
set -u
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=/home/yewon/.conda/envs/isaaclab/bin/python
source $RW/eunha_rl/scripts/eunha_env.sh ${EUNHA_TAG:-s0821}
cd $RW
CUDA_VISIBLE_DEVICES=0 timeout --signal=KILL 900 $PY course_mission/scripts/probe_idle_jitter.py \
  --headless --device cuda:0 > $RW/course_mission/logs/idle.log 2>&1
grep -aE "^\[IDLE\]" $RW/course_mission/logs/idle.log
grep -a -m1 -A6 "Traceback" $RW/course_mission/logs/idle.log | head -8
echo "IDLE_DONE"
