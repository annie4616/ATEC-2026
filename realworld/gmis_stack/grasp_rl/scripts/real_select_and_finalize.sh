#!/bin/bash
# Real-noise round finalization for ONE gpu: per depth — (1) baseline eval of the rgbl
# winner UNDER the full real noise (koreal_<depth>_before.log), (2) checkpoint ladder of
# the grasp1m run (32 envs, 400 steps), (3) winner -> koreal_<depth>_after.log + real_d
# video suite. Nothing from earlier rounds is overwritten.
# Usage: real_select_and_finalize.sh <gpu> <wait_marker|-> <depth:donor_prefix:donor_iter:mprefix> [...]
set -u
RW=/hdd0/yewon/ATEC2026_Simulation_Challenge/real_world_preliminary/ATEC2026_Simulation_Challenge
PY=/home/yewon/.conda/envs/isaaclab/bin/python
PLAY=$RW/grasp_rl/scripts/play_grasp.py
GPU=${1:?gpu}; MARKER=${2:?marker or -}; shift 2
R=$RW/grasp_rl/runs
D=$RW/grasp_rl/detached
[ "$MARKER" != "-" ] && until [ -f "$D/$MARKER.done" ]; do sleep 120; done
export GRASP_PERC_NOISE=rgbl
export GRASP_REAL_NOISE=1
export GRASP_BANK_KOFILTER=1
export GRASP_CAMHIT_GRACE=10
export GRASP_VIDEO_PREFIX=real_d
resolve_iter() { local d; d=$(ls -td $R/${1}_2* | head -1); echo "$d/model_${2}.pt"; }
evalck() {  # ckpt out_log
  CUDA_VISIBLE_DEVICES=$GPU $PY $PLAY --headless --device cuda:0 --task ATEC-Grasp-B2Piper-Stage2-v0 \
    --crouch_depth "$depth" --num_envs 32 --steps 400 --checkpoint "$1" > "$2" 2>&1
  grep -m1 -oE 'grasp successes \([0-9.]+%' "$2" | grep -oE '[0-9.]+' | tail -1
}
cd $RW
for spec in "$@"; do
  IFS=: read -r depth dprefix diter mprefix <<< "$spec"
  brate=$(evalck "$(resolve_iter $dprefix $diter)" "$D/koreal_${depth}_before.log"); brate=${brate:-0}
  echo "SELR_BASE depth=$depth donor=${dprefix}/model_${diter} rate=${brate}%"
  run=$(ls -td $R/${mprefix}_2* | head -1)
  best_rate=-1; best_n=""; best_ck=""
  for n in 300 600 900 1200 1499; do
    [ -f "$run/model_$n.pt" ] || continue
    rate=$(evalck "$run/model_$n.pt" "$D/selr_${depth}_$n.log"); rate=${rate:-0}
    echo "SELR depth=$depth it=$n rate=${rate}%"
    if awk -v a="$rate" -v b="$best_rate" 'BEGIN{exit !(a>b)}'; then
      best_rate=$rate; best_n=$n; best_ck=$run/model_$n.pt
    fi
  done
  [ -n "$best_ck" ] || { echo "SELR_FAIL depth=$depth (no checkpoints in $run)"; continue; }
  echo "SELR_BEST depth=$depth it=$best_n rate=${best_rate}%"
  cp "$D/selr_${depth}_${best_n}.log" "$D/koreal_${depth}_after.log"
  bash $RW/grasp_rl/scripts/record_depth_videos.sh "$best_ck" "$depth" "$GPU"
done
echo "SELR_DONE gpu=$GPU"
