#!/bin/bash
# Re-record the final v4 (grasp1h) video suites after the shared-intermediate mp4 race:
# one GPU records its two depths SEQUENTIALLY; the race itself is fixed in
# record_depth_videos.sh (private out_dir per recording).
# Usage: rerecord_v4_videos.sh <gpu> <depth:ckpt> [...]
set -u
RW=/hdd0/yewon/ATEC2026_Simulation_Challenge/real_world_preliminary/ATEC2026_Simulation_Challenge
GPU=${1:?gpu}; shift
export GRASP_BANK_KOFILTER=1
export GRASP_CAMHIT_GRACE=10
for spec in "$@"; do
  IFS=: read -r depth ckpt <<< "$spec"
  bash $RW/grasp_rl/scripts/record_depth_videos.sh "$ckpt" "$depth" "$GPU"
done
echo "REC_V4_DONE gpu=$GPU"
