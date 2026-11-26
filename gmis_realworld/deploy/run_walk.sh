#!/bin/bash
# Walk the venue course with the 0824 checkpoint + its direction code, and record
# overview/follow videos. Swappable: WALK_CKPT / ADAPT_CKPT / ADAPT_ACT.
set -u
RW=$(cd "$(dirname "$0")/.." && pwd)
cd "$RW"
export EUNHA_ADAPT_CKPT="$RW/${ADAPT_CKPT:-from_eunha/0824/adaptation.pt}"
export EUNHA_ADAPT_ACT=${ADAPT_ACT:-relu}
export MISSION_VENUE_PHOTO=${VENUE_PHOTO:-1}
CUDA_VISIBLE_DEVICES=${GPU:-0} ${PY:-python} deploy/walk_course.py --headless --device cuda:0 \
  --walk_ckpt "$RW/${WALK_CKPT:-from_eunha/0824/model_22998.pt}" \
  --vx ${VX:-0.7} --steps ${STEPS:-4500} --out_dir "$RW/deploy/logs/walk" ${EXTRA:-}
