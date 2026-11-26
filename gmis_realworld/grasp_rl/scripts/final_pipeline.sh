#!/bin/bash
# End-to-end final queue for ONE gpu: wait for its v2 finetunes -> keep-out verification
# (before=1d, after=v2/1f) -> final 3-video suite per depth. Fully detached-safe.
# Usage: final_pipeline.sh <gpu> <train_done_marker_name> <depth:before_ckpt:f_run_prefix> [...]
set -u
RW=/hdd0/yewon/ATEC2026_Simulation_Challenge/real_world_preliminary/ATEC2026_Simulation_Challenge
GPU=${1:?gpu}; MARKER=${2:?train done marker}; shift 2
D=$RW/grasp_rl/detached
until [ -f "$D/$MARKER.done" ]; do sleep 120; done
# preserve v1 verification logs before the final run overwrites them
for f in "$D"/ko_*.log; do [ -f "$f" ] && cp -n "$f" "${f%.log}.v1.log"; done
ARGS=()
for spec in "$@"; do
  IFS=: read -r depth before fprefix <<< "$spec"
  frun=$(ls -td $RW/grasp_rl/runs/${fprefix}_2* | head -1)
  fckpt=$frun/$(ls "$frun" | grep -oE "model_[0-9]+.pt" | sort -t_ -k2 -n | tail -1)
  echo "final: depth $depth after=$fckpt"
  ARGS+=("$depth:$before:$fckpt")
done
bash $RW/grasp_rl/scripts/verify_keepout.sh "$GPU" "${ARGS[@]}"
echo "FINAL_PIPELINE_DONE gpu=$GPU"
