#!/bin/bash
set -u
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=/home/yewon/.conda/envs/isaaclab/bin/python
source $RW/eunha_rl/scripts/eunha_env.sh ${EUNHA_TAG:-s0821}
cd $RW
CUDA_VISIBLE_DEVICES=0 timeout --signal=KILL 5400 $PY course_mission/scripts/probe_latency.py \
  --headless --device cuda:0 --delays 0,1,2,3 --sections grass,mud,pebble \
  > $RW/course_mission/logs/latency.log 2>&1
echo "LATENCY_DONE"
