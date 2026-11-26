#!/bin/bash
set -u
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=/home/yewon/.conda/envs/isaaclab/bin/python
source $RW/eunha_rl/scripts/eunha_env.sh ${EUNHA_TAG:-s0821}
cd $RW
CUDA_VISIBLE_DEVICES=0 timeout --signal=KILL 5400 $PY course_mission/scripts/probe_manip.py \
  --headless --device cuda:0 --trials "redbag:grass,redbag:mud,lunchbox:pebble" \
  --out_dir $RW/course_mission/videos_bag > $RW/course_mission/logs/probe_bag.log 2>&1
grep -aE "\[PROBE\]|grasp start" $RW/course_mission/logs/probe_bag.log | tail -20
grep -a -m1 -A12 "Traceback" $RW/course_mission/logs/probe_bag.log | head -14
echo "PROBEBAG_DONE"
