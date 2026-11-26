#!/bin/bash
set -u
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=/home/yewon/.conda/envs/isaaclab/bin/python
cd $RW
CUDA_VISIBLE_DEVICES=0 timeout --signal=KILL 900 $PY eunha_rl/scripts/convert_basket.py \
  --headless > $RW/course_mission/logs/convert_basket.log 2>&1
tail -5 $RW/course_mission/logs/convert_basket.log
grep -a -m1 -A8 "Traceback" $RW/course_mission/logs/convert_basket.log | head -10
echo "CONVERT_DONE"
