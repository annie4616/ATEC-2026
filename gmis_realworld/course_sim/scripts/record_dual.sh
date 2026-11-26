#!/bin/bash
# ONE run -> BOTH videos (overview + follow) with per-section friction terrain.
set -u
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=/home/yewon/.conda/envs/isaaclab/bin/python
export CUDA_MPS_PIPE_DIRECTORY=/tmp/yewon-no-mps
export EUNHA_ADAPT_CKPT=$RW/from_eunha/speed_1.0/adaptation.pt
export EUNHA_ADAPT_ACT=elu
cd $RW
V=$RW/course_sim/videos
CUDA_VISIBLE_DEVICES=5 timeout --signal=KILL 3000 $PY course_sim/scripts/walk_course.py --headless --device cuda:0 \
  --walk_ckpt from_eunha/speed_1.0/model_14999.pt > $V/walk_dual.log 2>&1
for m in overview follow; do
  [ -f "$V/walk_$m.mp4" ] && echo "WALKREC_OK walk_$m.mp4" || { echo "WALKREC_FAIL $m"; }
done
grep -E "final x=|per-section terrain" $V/walk_dual.log | tail -3
grep -A8 "Traceback" $V/walk_dual.log | tail -10
echo "WALKREC_DUAL_DONE"
