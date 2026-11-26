#!/bin/bash
# v3 final queue for ONE gpu: wait for its v3 finetunes -> keep-out verification
# (before=v2/grasp1f, after=v3/grasp1g, both under the NEW dense metric) -> final videos.
# Usage: final_pipeline_v3.sh <gpu> <train_done_marker_name> <depth:before_ckpt:g_run_prefix> [...]
set -u
RW=/hdd0/yewon/ATEC2026_Simulation_Challenge/real_world_preliminary/ATEC2026_Simulation_Challenge
GPU=${1:?gpu}; MARKER=${2:?train done marker}; shift 2
D=$RW/grasp_rl/detached
until [ -f "$D/$MARKER.done" ]; do sleep 120; done
# preserve v2 verification logs before this run overwrites them
for f in "$D"/ko_*.log; do [ -f "$f" ] && cp -n "$f" "${f%.log}.v2.log"; done
ARGS=()
for spec in "$@"; do
  IFS=: read -r depth before gprefix <<< "$spec"
  grun=$(ls -td $RW/grasp_rl/runs/${gprefix}_2* | head -1)
  gckpt=$grun/$(ls "$grun" | grep -oE "model_[0-9]+.pt" | sort -t_ -k2 -n | tail -1)
  echo "final_v3: depth $depth after=$gckpt"
  ARGS+=("$depth:$before:$gckpt")
done
bash $RW/grasp_rl/scripts/verify_keepout.sh "$GPU" "${ARGS[@]}"
echo "FINAL_PIPELINE_V3_DONE gpu=$GPU"
