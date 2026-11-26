#!/bin/bash
# Record the latest RL checkpoint of the cross-training run to a video, and archive
# it under the run's videos/play/ folder with the checkpoint name + timestamp so
# successive recordings don't overwrite each other.
#
# Usage: record_latest_ckpt.sh <run_dir>
#   e.g. record_latest_ckpt.sh logs/rsl_rl/taskd_b2/2026-06-21_11-59-44
set -e
cd /hdd0/sohyun/atec_2026_tesk_e/ATEC2026_Simulation_Challenge

RUN_DIR="${1:-logs/rsl_rl/taskd_b2/2026-06-21_11-59-44}"
TASK="ATEC-TaskD-RL-Cross-Unitree-B2-v0"

# Pick the newest model_*.pt (highest iteration).
CKPT="$(ls -t "$RUN_DIR"/model_*.pt 2>/dev/null | head -1)"
if [ -z "$CKPT" ]; then echo "[record] no checkpoint in $RUN_DIR yet"; exit 0; fi
CKPT_NAME="$(basename "$CKPT" .pt)"           # e.g. model_1000
TS="$(date +%H%M%S)"

echo "[record] $CKPT  (run=$RUN_DIR)"
CUDA_VISIBLE_DEVICES=6 TMPDIR=/hdd0/sohyun/tmp PYTHONPATH=/hdd0/sohyun PYTHONUNBUFFERED=1 \
  /home/sohyun/.conda/envs/env_isaaclab/bin/python scripts/rsl_rl/play.py \
  --task "$TASK" --num_envs 16 --checkpoint "$CKPT" \
  --video --video_length 300 --headless

# play.py writes to <run>/videos/play/rl-video-step-0.mp4 (overwritten each run).
# Archive it under the checkpoint name + timestamp so the folder keeps a history.
SRC="$RUN_DIR/videos/play/rl-video-step-0.mp4"
if [ -f "$SRC" ]; then
  DST="$RUN_DIR/videos/play/${CKPT_NAME}_${TS}.mp4"
  cp "$SRC" "$DST"
  echo "[record] archived -> $DST"
else
  echo "[record] WARNING: expected video not found at $SRC"
fi
