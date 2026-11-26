#!/bin/bash
# CONTROL for the bag sweep. Every bag setting scored 0/3 with a post-crouch alignment
# error of 0.14-0.25 m against a 0.10 m tolerance, which points at the APPROACH rather
# than the pinch: a 3 mm wall is unreachable 20 cm off, but so is a 66 mm can. This runs
# the identical probe over a can, a rind and a bag in ONE session so their alignment
# errors are directly comparable.
set -u
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=/home/yewon/.conda/envs/isaaclab/bin/python
source $RW/eunha_rl/scripts/eunha_env.sh ${EUNHA_TAG:-s0821}
cd $RW
COURSE_VENUE=hk MISSION_VENUE_PHOTO=1 CUDA_VISIBLE_DEVICES=${GPU:-0} \
  timeout --signal=KILL 3600 $PY course_mission/scripts/probe_manip.py \
  --headless --device cuda:0 \
  --trials "can:grass,rind:grass,lunchbox:grass,redbag:grass" \
  --out_dir $RW/course_mission/videos_bag_control \
  > $RW/course_mission/logs/bagsweep/control.log 2>&1
echo CONTROL_DONE >> $RW/course_mission/logs/bagsweep/control.log
