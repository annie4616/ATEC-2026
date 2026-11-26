#!/bin/bash
set -u
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=/home/yewon/.conda/envs/isaaclab/bin/python
export CUDA_MPS_PIPE_DIRECTORY=/tmp/yewon-no-mps
cd $RW
CUDA_VISIBLE_DEVICES=0 timeout --signal=KILL 1800 $PY eunha_rl/scripts/convert_basket.py --headless \
  > $RW/course_mission/logs/rebuild_basket.log 2>&1
tail -6 $RW/course_mission/logs/rebuild_basket.log
ls -la $RW/eunha_rl/assets/a2_piperx_basket.usd
echo "REBUILD_DONE"
