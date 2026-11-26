#!/bin/bash
# Sequential: (1) s100+arm dual, (2) bare A2 (arm A/B), (3) s075. One GPU5 job.
set -u
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=/home/yewon/.conda/envs/isaaclab/bin/python
export CUDA_MPS_PIPE_DIRECTORY=/tmp/yewon-no-mps
export EUNHA_ADAPT_ACT=elu
cd $RW
V=$RW/course_sim/videos
run() {  # tag task ckpt adapt vx steps
  local tag=$1 task=$2 ckpt=$3 adapt=$4 vx=$5 steps=$6
  EUNHA_ADAPT_CKPT=$RW/from_eunha/$adapt/adaptation.pt \
  CUDA_VISIBLE_DEVICES=5 timeout --signal=KILL 3000 $PY course_sim/scripts/walk_course.py --headless --device cuda:0 \
    --task "$task" --walk_ckpt "from_eunha/$adapt/model_14999.pt" --vx "$vx" --steps "$steps" \
    > $V/walk_${tag}.log 2>&1
  for m in overview follow; do
    [ -f "$V/walk_$m.mp4" ] && mv "$V/walk_$m.mp4" "$V/${tag}_$m.mp4" && echo "REC_OK ${tag}_$m.mp4"
  done
  grep -E "final x=" $V/walk_${tag}.log | tail -1
  grep -m1 -A6 "Traceback" $V/walk_${tag}.log | head -8
}
run s100arm ATEC-Course-A2Piper x speed_1.0 0.7 3500
run bare    ATEC-Course-A2Bare  x speed_1.0 0.7 2000
run s075arm ATEC-Course-A2Piper x speed_0.75 0.5 3500
echo "REC_ALL3_DONE"
