#!/bin/bash
# Four lunchboxes, nothing else: the only class standing between 80 and 120.
set -u
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=/home/yewon/.conda/envs/isaaclab/bin/python
source $RW/eunha_rl/scripts/eunha_env.sh ${EUNHA_TAG:-s0821}
cd $RW
CUDA_VISIBLE_DEVICES=0 timeout --signal=KILL 3600 $PY course_mission/scripts/probe_manip.py \
  --headless --device cuda:0 --depth 0.24 \
  --trials lunchbox:mud,lunchbox:pebble,lunchbox:grass,lunchbox:mixed \
  --out_dir $RW/course_mission/videos_diag \
  > $RW/course_mission/logs/probe_lunchbox.log 2>&1
grep -aE "grasp FAILED|basket deposit|^\[PROBE\]" $RW/course_mission/logs/probe_lunchbox.log | tail -14
echo "PROBELB_DONE"
