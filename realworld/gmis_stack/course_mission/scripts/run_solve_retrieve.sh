#!/bin/bash
set -u
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=/home/yewon/.conda/envs/isaaclab/bin/python
cd $RW
export CUDA_MPS_PIPE_DIRECTORY=/tmp/yewon-no-mps
CUDA_VISIBLE_DEVICES=0 timeout --signal=KILL 2400 $PY course_mission/scripts/solve_retrieve.py \
  --headless --device cuda:0 > $RW/course_mission/logs/solve_retrieve.log 2>&1
grep -E "\[RET\]" $RW/course_mission/logs/solve_retrieve.log
grep -m1 -A6 "Traceback\|SystemExit" $RW/course_mission/logs/solve_retrieve.log | head -8
echo "SOLVE_DONE"
