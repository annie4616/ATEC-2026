#!/bin/bash
# Checkpoint-ladder selection for the grasp1k noise round: training success peaks
# mid-run and decays, so evaluate a ladder of checkpoints UNDER the rgbl noise and
# keep the best. The winner's eval becomes korgbl_<depth>_after.log (k-final version
# backed up as *.kfinal.log) and the rgbl_d videos are re-recorded with the winner.
# Usage: rgbl_select_and_finalize.sh <gpu> <wait_marker|-> <depth:kprefix> [...]
set -u
RW=/hdd0/yewon/ATEC2026_Simulation_Challenge/real_world_preliminary/ATEC2026_Simulation_Challenge
PY=/home/yewon/.conda/envs/isaaclab/bin/python
PLAY=$RW/grasp_rl/scripts/play_grasp.py
GPU=${1:?gpu}; MARKER=${2:?marker or -}; shift 2
R=$RW/grasp_rl/runs
D=$RW/grasp_rl/detached
[ "$MARKER" != "-" ] && until [ -f "$D/$MARKER.done" ]; do sleep 120; done
export GRASP_PERC_NOISE=rgbl
export GRASP_BANK_KOFILTER=1
export GRASP_CAMHIT_GRACE=10
export GRASP_VIDEO_PREFIX=rgbl_d
cd $RW
for spec in "$@"; do
  IFS=: read -r depth kprefix <<< "$spec"
  run=$(ls -td $R/${kprefix}_2* | head -1)
  best_rate=-1; best_n=""; best_ck=""
  for n in 300 600 900 1200 1499; do
    [ -f "$run/model_$n.pt" ] || continue
    log=$D/selg_${depth}_$n.log
    # 32 envs: object sizes are drawn ONCE per env at startup, so few-env evals sample
    # only a handful of sizes and swing wildly run-to-run (38% vs 74% for one ckpt)
    CUDA_VISIBLE_DEVICES=$GPU $PY $PLAY --headless --device cuda:0 --task ATEC-Grasp-B2Piper-Stage2-v0 \
      --crouch_depth "$depth" --num_envs 32 --steps 400 --checkpoint "$run/model_$n.pt" \
      > "$log" 2>&1
    rate=$(grep -m1 -oE 'grasp successes \([0-9.]+%' "$log" | grep -oE '[0-9.]+' | tail -1)
    rate=${rate:-0}
    echo "SELG depth=$depth it=$n rate=${rate}%"
    if awk -v a="$rate" -v b="$best_rate" 'BEGIN{exit !(a>b)}'; then
      best_rate=$rate; best_n=$n; best_ck=$run/model_$n.pt
    fi
  done
  [ -n "$best_ck" ] || { echo "SELG_FAIL depth=$depth (no checkpoints found in $run)"; continue; }
  echo "SELG_BEST depth=$depth it=$best_n rate=${best_rate}%"
  cp -n "$D/korgbl_${depth}_after.log" "$D/korgbl_${depth}_after.kfinal.log" 2>/dev/null
  cp "$D/selg_${depth}_${best_n}.log" "$D/korgbl_${depth}_after.log"
  bash $RW/grasp_rl/scripts/record_depth_videos.sh "$best_ck" "$depth" "$GPU"
done
echo "SELG_DONE gpu=$GPU"
