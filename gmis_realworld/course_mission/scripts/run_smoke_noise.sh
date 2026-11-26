#!/bin/bash
set -u
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=/home/yewon/.conda/envs/isaaclab/bin/python
source $RW/eunha_rl/scripts/eunha_env.sh ${EUNHA_TAG:-s0821}
cd $RW
CUDA_VISIBLE_DEVICES=0 timeout --signal=KILL 2400 $PY course_mission/scripts/smoke_mission.py \
  --headless --device cuda:0 --steps 1200 > $RW/course_mission/logs/smoke_noise.log 2>&1
grep -aE "\[SMOKE\] step|precision|recall|layout check" $RW/course_mission/logs/smoke_noise.log | tail -12
grep -a -m1 -A10 "Traceback" $RW/course_mission/logs/smoke_noise.log | head -12
echo "SMOKENOISE_DONE"
