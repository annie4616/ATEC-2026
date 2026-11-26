#!/bin/bash
set -u
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
E=$RW/eunha_rl
PY=/home/yewon/.conda/envs/isaaclab/bin/python
export CUDA_MPS_PIPE_DIRECTORY=/tmp/yewon-no-mps
export EUNHA_ADAPT_CKPT=$RW/from_eunha/speed_0.75/adaptation.pt
export EUNHA_ADAPT_ACT=elu
until grep -q "BARE_AB_DONE" $E/detached/bareab.log 2>/dev/null; do
  p=$(cat $E/detached/bareab.pid 2>/dev/null); kill -0 $p 2>/dev/null || break
  sleep 60
done
cd $RW
V=$RW/course_sim/videos
CUDA_VISIBLE_DEVICES=5 timeout --signal=KILL 3000 $PY course_sim/scripts/walk_course.py --headless --device cuda:0 \
  --task ATEC-Course-A2Piper --walk_ckpt from_eunha/speed_0.75/model_14999.pt --vx 0.5 \
  > $V/walk_s075.log 2>&1
for m in overview follow; do
  [ -f "$V/walk_$m.mp4" ] && mv "$V/walk_$m.mp4" "$V/walk_s075_$m.mp4" && echo "S075_OK walk_s075_$m.mp4"
done
grep -E "final x=" $V/walk_s075.log | tail -1
grep -A8 "Traceback" $V/walk_s075.log | tail -10
echo "S075_REC_DONE"
