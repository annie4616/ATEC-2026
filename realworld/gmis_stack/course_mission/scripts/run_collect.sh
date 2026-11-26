#!/bin/bash
set -u
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=/home/yewon/.conda/envs/isaaclab/bin/python
source $RW/eunha_rl/scripts/eunha_env.sh ${EUNHA_TAG:-s0821}
cd $RW
# noise ON: the offline tuning must see the pixels the deployed robot will see
CUDA_VISIBLE_DEVICES=0 timeout --signal=KILL 3600 $PY course_mission/scripts/collect_frames.py \
  --headless --device cuda:0 --steps 5000 --every 20 \
  > $RW/course_mission/logs/collect.log 2>&1
echo "COLLECT_DONE"
