#!/bin/bash
set -u
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=/home/yewon/.conda/envs/isaaclab/bin/python
source $RW/eunha_rl/scripts/eunha_env.sh ${EUNHA_TAG:-s0821}
cd $RW
CUDA_VISIBLE_DEVICES=0 timeout --signal=KILL 2700 $PY course_mission/scripts/smoke_mission.py \
  --headless --device cuda:0 --steps 1600 > $RW/course_mission/logs/smoke_rng.log 2>&1
echo "SMOKERNG_DONE"
