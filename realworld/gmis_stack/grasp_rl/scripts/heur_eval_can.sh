#!/bin/bash
# CAN-focused inference-time heuristic ablation, per depth x per method generation.
# Methods: v4 (noise-free training), rgbl (perception-noise winner), real (full-noise
# winner). Heuristics are evaluated INDIVIDUALLY first (a stacked combo was measured to
# destroy performance: 46.8% -> 15.4%); an adaptive combo of the singles that beat the
# baseline by >2%p is then tried; the best variant wins. 32 envs x 400 steps per eval.
# Usage: heur_eval_can.sh <gpu> <wait_marker|-> <depth> [...]
set -u
RW=/hdd0/yewon/ATEC2026_Simulation_Challenge/real_world_preliminary/ATEC2026_Simulation_Challenge
PY=/home/yewon/.conda/envs/isaaclab/bin/python
PLAY=$RW/grasp_rl/scripts/play_grasp.py
GPU=${1:?gpu}; MARKER=${2:?marker or -}; shift 2
R=$RW/grasp_rl/runs
D=$RW/grasp_rl/detached
V=$RW/grasp_rl/videos
[ "$MARKER" != "-" ] && until [ -f "$D/$MARKER.done" ]; do sleep 120; done
CAN="--shape cylinder --width_range 0.066,0.066 --length_range 0.066,0.066 --height_range 0.115,0.115"
COMMON="GRASP_BANK_KOFILTER=1 GRASP_CAMHIT_GRACE=10"
PRIOR="GRASP_DIMS_PRIOR=0.066,0.066,0.115"
EMA="GRASP_PN_EMA=0.4"
cd $RW

sel_iter() {  # tagfile-pattern depth -> winner iter parsed from selection logs
  grep -hE "$1 depth=$2 " $D/rgbl3_gpu5.log $D/rgbl3_gpu4.log $D/real2_gpu5.log $D/real2_gpu4.log 2>/dev/null \
    | head -1 | grep -oE 'it=[0-9]+' | grep -oE '[0-9]+'
}
rate_of() { grep -m1 -oE 'grasp successes \([0-9.]+%' "$1" 2>/dev/null | grep -oE '[0-9.]+' | tail -1; }

run_eval() {  # method depth ckpt tag heurflag extravars
  local log=$D/hcan_${1}_${2}_${4}.log
  env $COMMON $6 CUDA_VISIBLE_DEVICES=$GPU $PY $PLAY --headless --device cuda:0 \
    --task ATEC-Grasp-B2Piper-Stage2-v0 --crouch_depth "$2" --num_envs 32 --steps 400 \
    $CAN ${5:+--heur $5} --checkpoint "$3" > "$log" 2>&1
  local r; r=$(rate_of "$log"); r=${r:-0}
  echo "HCAN method=$1 depth=$2 var=$4 rate=${r}%" >&2   # stderr -> job log even when captured
  echo "$r"
}

for depth in "$@"; do
  t=${depth#0.}; [ ${#t} -eq 1 ] && t="${t}0"; tagd="c0$t"   # 0.2->c020, 0.21->c021, 0.3->c030
  # ---- resolve the three method checkpoints ------------------------------------------
  case $depth in
    0.21|0.23|0.25) CK_V4=$(ls -td $R/grasp1i_${tagd}_2* | head -1)/model_1999.pt ;;
    *)              CK_V4=$(ls -td $R/grasp1h_${tagd}_2* | head -1)/model_1499.pt ;;
  esac
  KIT=$(sel_iter SELG_BEST $depth); MIT=$(sel_iter SELR_BEST $depth)
  CK_K=$(ls -td $R/grasp1k_${tagd}_2* | head -1)/model_${KIT:-1499}.pt
  CK_M=$(ls -td $R/grasp1m_${tagd}_2* | head -1)/model_${MIT:-1499}.pt
  # deployment policy for the real regime = the better of {rgbl winner (SELR_BASE),
  # grasp1m ladder winner (SELR_BEST)} — at some depths the extra noise round lost ground
  srate() { grep -hE "$1 depth=$depth " $D/real2_gpu5.log $D/real2_gpu4.log 2>/dev/null | head -1 | grep -oE 'rate=[0-9.]+' | grep -oE '[0-9.]+'; }
  BR=$(srate SELR_BASE); MR=$(srate SELR_BEST)
  if [ -n "$BR" ] && [ -n "$MR" ] && awk -v a="$BR" -v b="$MR" 'BEGIN{exit !(a>b)}'; then
    CK_M=$CK_K
    echo "HCAN_NOTE depth=$depth real-method ckpt = rgbl winner (base ${BR}% > ladder ${MR}%)"
  fi
  echo "HCAN_CKPTS depth=$depth v4=$CK_V4 rgbl=$CK_K real=$CK_M"

  # ---- REAL method: baseline + singles + adaptive combo ------------------------------
  RVARS="GRASP_PERC_NOISE=rgbl GRASP_REAL_NOISE=1"
  base=$(run_eval real $depth "$CK_M" none   ""       "$RVARS" | tail -1)
  rp=$(run_eval  real $depth "$CK_M" prior  ""       "$RVARS $PRIOR" | tail -1)
  re=$(run_eval  real $depth "$CK_M" ema    ""       "$RVARS $EMA" | tail -1)
  ro=$(run_eval  real $depth "$CK_M" open   open     "$RVARS" | tail -1)
  rr=$(run_eval  real $depth "$CK_M" reopen reopen   "$RVARS" | tail -1)
  rs=$(run_eval  real $depth "$CK_M" shield shield   "$RVARS" | tail -1)
  cflag=""; cvars="$RVARS"
  awk -v a="$rp" -v b="$base" 'BEGIN{exit !(a>b+2)}' && cvars="$cvars $PRIOR"
  awk -v a="$re" -v b="$base" 'BEGIN{exit !(a>b+2)}' && cvars="$cvars $EMA"
  awk -v a="$ro" -v b="$base" 'BEGIN{exit !(a>b+2)}' && cflag="open,$cflag"
  awk -v a="$rr" -v b="$base" 'BEGIN{exit !(a>b+2)}' && cflag="reopen,$cflag"
  awk -v a="$rs" -v b="$base" 'BEGIN{exit !(a>b+2)}' && cflag="shield,$cflag"
  cflag=${cflag%,}
  if [ -n "$cflag" ] || [ "$cvars" != "$RVARS" ]; then
    rc=$(run_eval real $depth "$CK_M" combo "$cflag" "$cvars" | tail -1)
  else
    rc=-1
  fi
  # winner = argmax over evaluated variants
  best_tag=none; best_rate=$base
  for pair in "prior:$rp" "ema:$re" "open:$ro" "reopen:$rr" "shield:$rs" "combo:$rc"; do
    t=${pair%%:*}; v=${pair##*:}
    awk -v a="$v" -v b="$best_rate" 'BEGIN{exit !(a>b)}' && { best_tag=$t; best_rate=$v; }
  done
  echo "HCAN_BEST method=real depth=$depth var=$best_tag rate=${best_rate}%"

  # ---- rgbl + v4 methods: baseline + the real-method winner variant ------------------
  KVARS="GRASP_PERC_NOISE=rgbl"
  run_eval rgbl $depth "$CK_K" none "" "$KVARS" > /dev/null
  case $best_tag in
    none)   : ;;
    prior)  run_eval rgbl $depth "$CK_K" prior  ""     "$KVARS $PRIOR" > /dev/null ;;
    ema)    run_eval rgbl $depth "$CK_K" ema    ""     "$KVARS $EMA" > /dev/null ;;
    combo)  run_eval rgbl $depth "$CK_K" combo "$cflag" "$(echo "$cvars" | sed 's/GRASP_REAL_NOISE=1//')" > /dev/null ;;
    *)      run_eval rgbl $depth "$CK_K" $best_tag "$best_tag" "$KVARS" > /dev/null ;;
  esac
  run_eval v4 $depth "$CK_V4" none   ""     "" > /dev/null
  run_eval v4 $depth "$CK_V4" shield shield "" > /dev/null

  # ---- video: real-method winner on the can ------------------------------------------
  wflag=""; wvars="$RVARS"
  case $best_tag in
    prior) wvars="$RVARS $PRIOR" ;;
    ema)   wvars="$RVARS $EMA" ;;
    combo) wflag="$cflag"; wvars="$cvars" ;;
    none)  : ;;
    *)     wflag="$best_tag" ;;
  esac
  TMPV=$V/.rec_heur_${depth}_$$
  rm -rf "$TMPV"; mkdir -p "$TMPV"
  env $COMMON $wvars CUDA_VISIBLE_DEVICES=$GPU $PY $PLAY --headless --device cuda:0 \
    --task ATEC-Grasp-B2Piper-Stage2-v0 --crouch_depth "$depth" --num_envs 4 --steps 600 \
    $CAN ${wflag:+--heur $wflag} --checkpoint "$CK_M" --video --out_dir "$TMPV" \
    > "$V/heur_d${depth}_can.log" 2>&1
  [ -f "$TMPV/grasp_play-step-0.mp4" ] && mv "$TMPV/grasp_play-step-0.mp4" "$V/heur_d${depth}_can.mp4" \
    || echo "VIDEO_MISSING heur_d${depth}_can"
  rm -rf "$TMPV"
done
echo "HCAN_DONE gpu=$GPU"
