#!/bin/bash
# v4 final queue for ONE gpu: wait for its v4 finetunes -> keep-out verification
# (before=v3/grasp1g, after=v4/grasp1h, BOTH under bank-filter + grace rules) -> videos.
# Usage: final_pipeline_v4.sh <gpu> <train_done_marker_name> <depth:before_ckpt:h_run_prefix> [...]
set -u
RW=/hdd0/yewon/ATEC2026_Simulation_Challenge/real_world_preliminary/ATEC2026_Simulation_Challenge
GPU=${1:?gpu}; MARKER=${2:?train done marker}; shift 2
D=$RW/grasp_rl/detached
until [ -f "$D/$MARKER.done" ]; do sleep 120; done
export GRASP_BANK_KOFILTER=1
export GRASP_CAMHIT_GRACE=10
# preserve v3 verification logs before this run overwrites them
for f in "$D"/ko_*.log; do [ -f "$f" ] && cp -n "$f" "${f%.log}.v3.log"; done
ARGS=()
for spec in "$@"; do
  IFS=: read -r depth before hprefix <<< "$spec"
  hrun=$(ls -td $RW/grasp_rl/runs/${hprefix}_2* | head -1)
  hckpt=$hrun/$(ls "$hrun" | grep -oE "model_[0-9]+.pt" | sort -t_ -k2 -n | tail -1)
  echo "final_v4: depth $depth after=$hckpt"
  ARGS+=("$depth:$before:$hckpt")
done
bash $RW/grasp_rl/scripts/verify_keepout.sh "$GPU" "${ARGS[@]}"
echo "FINAL_PIPELINE_V4_DONE gpu=$GPU"
