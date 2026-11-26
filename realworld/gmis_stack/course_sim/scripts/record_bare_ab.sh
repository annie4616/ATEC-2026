#!/bin/bash
set -u
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
E=$RW/eunha_rl
PY=/home/yewon/.conda/envs/isaaclab/bin/python
export CUDA_MPS_PIPE_DIRECTORY=/tmp/yewon-no-mps
export EUNHA_ADAPT_CKPT=$RW/from_eunha/speed_1.0/adaptation.pt
export EUNHA_ADAPT_ACT=elu
until grep -q "WALKREC_DUAL_DONE" $E/detached/wdual.log 2>/dev/null; do
  p=$(cat $E/detached/wdual.pid 2>/dev/null); kill -0 $p 2>/dev/null || break
  sleep 60
done
cd $RW
V=$RW/course_sim/videos
CUDA_VISIBLE_DEVICES=5 timeout --signal=KILL 3000 $PY course_sim/scripts/walk_course.py --headless --device cuda:0 \
  --task ATEC-Course-A2Bare --walk_ckpt from_eunha/speed_1.0/model_14999.pt --steps 2000 \
  > $V/walk_bare.log 2>&1
for m in overview follow; do
  [ -f "$V/walk_$m.mp4" ] && mv "$V/walk_$m.mp4" "$V/walk_bare_$m.mp4" && echo "BARE_OK walk_bare_$m.mp4"
done
grep -E "final x=" $V/walk_bare.log | tail -1
grep -A8 "Traceback" $V/walk_bare.log | tail -10
echo "BARE_AB_DONE"
