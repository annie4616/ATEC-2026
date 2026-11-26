#!/bin/bash
# Per-depth video suite: basic primitives / cylinder can / watermelon rind.
# args: <checkpoint> <depth> [gpu=5]
# outputs: grasp_rl/videos/d<depth>_{basic,can,rind}.mp4 (+ .log with success stats)
set -u
RW=/hdd0/yewon/ATEC2026_Simulation_Challenge/real_world_preliminary/ATEC2026_Simulation_Challenge
CKPT=${1:?checkpoint required}
DEPTH=${2:?depth tag required (e.g. 0.22)}
GPU=${3:-5}
PY=/home/yewon/.conda/envs/isaaclab/bin/python
PLAY=$RW/grasp_rl/scripts/play_grasp.py
V=${GRASP_VIDEO_DIR:-$RW/grasp_rl/videos}
mkdir -p "$V"
cd $RW

PFX=${GRASP_VIDEO_PREFIX:-d}   # output prefix; override to keep a new round from clobbering old videos

rec() {
  local name=$1; shift
  echo "=== ${PFX}${DEPTH} ${name} ==="
  # PRIVATE out_dir per recording: two GPUs record concurrently, and a shared
  # grasp_play-step-0.mp4 intermediate silently races (stale/cross-wired mp4s)
  local TMPV=$V/.rec_${PFX}${DEPTH}_${name}_$$
  rm -rf "$TMPV"; mkdir -p "$TMPV"
  CUDA_VISIBLE_DEVICES=$GPU $PY $PLAY --headless --device cuda:0 --task ATEC-Grasp-B2Piper-Stage2-v0 \
    --crouch_depth "$DEPTH" --num_envs 4 --steps 600 --checkpoint "$CKPT" --video --out_dir "$TMPV" \
    ${GRASP_REC_HEUR:+--heur $GRASP_REC_HEUR} "$@" \
    > "$V/${PFX}${DEPTH}_${name}.log" 2>&1
  if [ -f "$TMPV/grasp_play-step-0.mp4" ]; then
    mv "$TMPV/grasp_play-step-0.mp4" "$V/${PFX}${DEPTH}_${name}.mp4"
  else
    echo "VIDEO_MISSING ${PFX}${DEPTH}_${name} — check $V/${PFX}${DEPTH}_${name}.log"
  fi
  rm -rf "$TMPV"
  grep -E "rollout:" "$V/${PFX}${DEPTH}_${name}.log" || echo "(no rollout line — check log)"
}

rec basic --width_range 0.030,0.060 --length_range 0.030,0.060 --height_range 0.060,0.130
GRASP_OBJ_MASS=0.2 rec can   --width_range 0.066,0.066 --length_range 0.066,0.066 --height_range 0.115,0.115 --shape cylinder
rec rind  --width_range 0.050,0.050 --length_range 0.070,0.100 --height_range 0.025,0.040
if [ "${GRASP_REC_EXTRA:-}" = "1" ]; then
  # competition additions: 330 ml plastic bottle (~60x170, 0.2 kg) and the lunchbox
  # LOWER-BODY proxy (bottom 70 mm span x 160 long x 65 high; the 200x160 lid needs a
  # rim-pinch strategy that a solid-primitive proxy cannot represent)
  GRASP_OBJ_MASS=0.2 rec bottle --width_range 0.060,0.060 --length_range 0.060,0.060 --height_range 0.170,0.170 --shape cylinder
  GRASP_OBJ_MASS=0.1 rec lunchbox --width_range 0.070,0.070 --length_range 0.160,0.160 --height_range 0.065,0.065
fi
echo "VIDEOS_DONE depth=${DEPTH}"
ls -la "$V" | grep "${PFX}${DEPTH}_"
