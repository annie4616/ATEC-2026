#!/bin/bash
set -u
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=/home/yewon/.conda/envs/isaaclab/bin/python
cd $RW
export CUDA_MPS_PIPE_DIRECTORY=/tmp/yewon-no-mps
CUDA_VISIBLE_DEVICES=0 timeout --signal=KILL 1800 $PY course_mission/scripts/solve_retrieve.py \
  --headless --device cuda:0 --scan > $RW/course_mission/logs/scan_retrieve.log 2>&1
grep -E "\[SCAN\]" $RW/course_mission/logs/scan_retrieve.log | head -20
echo "SCAN_DONE"
