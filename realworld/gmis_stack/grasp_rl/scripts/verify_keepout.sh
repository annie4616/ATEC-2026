#!/bin/bash
# Keep-out before/after verification + video suite for a list of depths on one GPU.
# Usage: verify_keepout.sh <gpu> <depth:before_ckpt:after_ckpt> [...]
# Waits for after_ckpt to exist. All logs under grasp_rl/detached/ (session-independent).
set -u
RW=/hdd0/yewon/ATEC2026_Simulation_Challenge/real_world_preliminary/ATEC2026_Simulation_Challenge
GPU=${1:?gpu}; shift
PY=/home/yewon/.conda/envs/isaaclab/bin/python
PLAY=$RW/grasp_rl/scripts/play_grasp.py
DIR=$RW/grasp_rl/detached
KO=${GRASP_KO_PREFIX:-ko}   # log prefix; override to keep a new round from clobbering old logs
mkdir -p "$DIR"
cd $RW
for spec in "$@"; do
  IFS=: read -r depth before after <<< "$spec"
  until [ -f "$after" ]; do sleep 30; done
  for tag in before after; do
    ck=$([ $tag = before ] && echo "$before" || echo "$after")
    echo "=== d$depth $tag ($ck) ==="
    CUDA_VISIBLE_DEVICES=$GPU $PY $PLAY --headless --device cuda:0 --task ATEC-Grasp-B2Piper-Stage2-v0 \
      --crouch_depth "$depth" --num_envs 8 --steps 400 --checkpoint "$ck" \
      > "$DIR/${KO}_${depth}_${tag}.log" 2>&1
    grep -oE "rollout: .*|keep-out env-steps: [0-9]+" "$DIR/${KO}_${depth}_${tag}.log"
  done
  bash $RW/grasp_rl/scripts/record_depth_videos.sh "$after" "$depth" "$GPU"
done
echo "VERIFY_DONE gpu=$GPU"
