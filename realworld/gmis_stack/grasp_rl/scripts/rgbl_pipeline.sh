#!/bin/bash
# RGB+LiDAR round verification for ONE gpu: wait for its trainings -> per depth
# keep-out eval (before = noise-free-trained donor UNDER noise, after = grasp1j) +
# 3-video suite. All outputs use new prefixes (korgbl_*, rgbl_d*) — nothing overwritten.
# Usage: rgbl_pipeline.sh <gpu> <train_done_marker> <depth:donor_prefix:j_prefix> [...]
set -u
RW=/hdd0/yewon/ATEC2026_Simulation_Challenge/real_world_preliminary/ATEC2026_Simulation_Challenge
GPU=${1:?gpu}; MARKER=${2:?marker}; shift 2
R=$RW/grasp_rl/runs
D=$RW/grasp_rl/detached
until [ -f "$D/$MARKER.done" ]; do sleep 120; done
export GRASP_PERC_NOISE=rgbl
export GRASP_BANK_KOFILTER=1
export GRASP_CAMHIT_GRACE=10
export GRASP_KO_PREFIX=korgbl
export GRASP_VIDEO_PREFIX=rgbl_d
resolve() { local d; d=$(ls -td $R/${1}_2* | head -1); echo "$d/$(ls "$d" | grep -oE 'model_[0-9]+\.pt' | sort -t_ -k2 -n | tail -1)"; }
ARGS=()
for spec in "$@"; do
  IFS=: read -r depth donor jprefix <<< "$spec"
  before=$(resolve "$donor"); after=$(resolve "$jprefix")
  echo "rgbl: depth $depth before=$before after=$after"
  ARGS+=("$depth:$before:$after")
done
bash $RW/grasp_rl/scripts/verify_keepout.sh "$GPU" "${ARGS[@]}"
echo "RGBL_PIPELINE_DONE gpu=$GPU"
