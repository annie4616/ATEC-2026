#!/bin/bash
# Record ONE camera mode of the course walk. Usage: record_one.sh <overview|follow>
set -u
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=/home/yewon/.conda/envs/isaaclab/bin/python
export CUDA_MPS_PIPE_DIRECTORY=/tmp/yewon-no-mps
export EUNHA_ADAPT_CKPT=$RW/from_eunha/speed_1.0/adaptation.pt
export EUNHA_ADAPT_ACT=elu
mode=${1:?}
cd $RW
V=$RW/course_sim/videos
FLAG=""; [ "$mode" = follow ] && FLAG="--follow"
CUDA_VISIBLE_DEVICES=5 timeout --signal=KILL 2400 $PY course_sim/scripts/walk_course.py --headless --device cuda:0 \
  --walk_ckpt from_eunha/speed_1.0/model_14999.pt $FLAG --name walk_$mode \
  > $V/walk_$mode.log 2>&1
if [ -f "$V/walk_${mode}-step-0.mp4" ]; then
  mv "$V/walk_${mode}-step-0.mp4" "$V/walk_${mode}.mp4"
  echo "WALKREC_OK walk_${mode}.mp4 $(grep -oE 'final x=.*' $V/walk_$mode.log | tail -1)"
else
  echo "WALKREC_FAIL $mode"; grep -A8 "Traceback" $V/walk_$mode.log | tail -10
fi
echo "WALKREC_ONE_DONE $mode"
