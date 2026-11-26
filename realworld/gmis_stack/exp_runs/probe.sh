#!/bin/bash
# Track A: deployment-honest grasp probe for ATEC_gmis.
# Teleports to the bank anchor per object and runs approach -> crouch -> RL grasp,
# on the exact deploy plant of deploy/run_mission.sh (0824 walk pair + relu adapt,
# ARM_RATE 0.2, GRIP_RATE 0.02, ARM_DELAY 1, DET_SEED 1, GRASP_STEPS 500, photo world).
#
#   TAG=base bash exp_runs/probe.sh
#   TAG=rotcap MISSION_GRASP_ROT_CAP=0.35 bash exp_runs/probe.sh
#
# Knobs to vary are plain env vars (MISSION_*); TAG=base is the shipping control.
set -u
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-/home/yewon/.conda/envs/isaaclab/bin/python}
TAG=${TAG:?set TAG}
OUT=$RW/exp_runs/logs/probe_$TAG
LOG=$RW/exp_runs/logs/probe_$TAG.log
mkdir -p "$OUT"
# NEVER clobber a finished probe (venue-run lesson: a reused log overwrote a scoring run).
if [ -s "$LOG" ]; then
  echo "REFUSING: $LOG already holds a run; move it aside first"; exit 1
fi
# --- deploy plant (mirror of deploy/run_mission.sh) ---
export EUNHA_ADAPT_CKPT=$RW/from_eunha/0824/adaptation.pt
export EUNHA_ADAPT_ACT=relu
export MISSION_ARM_RATE=0.2 MISSION_GRIP_RATE=0.02 MISSION_ARM_DELAY=1
export MISSION_GRASP_DET_SEED=1 MISSION_DETECT_HZ=2
export MISSION_VENUE_PHOTO=${VENUE_PHOTO:-1} MISSION_GRASP_STEPS=500
export MISSION_AIM_X=${AIM_X:-0.12} MISSION_CLOSE_RETRY=${CLOSE_RETRY:-0.12}
export MISSION_BAG_HANDLE=1
export GRASP_TCP_OFFSET=0.135 GRASP_GRIP_OPEN=0.05
{
  echo "=== gmis grasp probe $TAG ==="
  env | grep -E '^MISSION_|^GRASP_|^EUNHA_' | sort
} > "$LOG"
cd "$RW"
CUDA_VISIBLE_DEVICES=${GPU:-0} timeout --signal=KILL ${TMO:-7200} $PY \
  course_mission/scripts/probe_manip.py --headless --device cuda:0 \
  --walk_ckpt "$RW/from_eunha/0824/model_22998.pt" \
  --seed ${SEED:-0} ${GRASP_CKPT:+--grasp_ckpt "$GRASP_CKPT"} \
  --trials "${TRIALS:-can:grass,can:mud,rind:grass,rind:mud,lunchbox:mud,lunchbox:grass}" \
  --trace_dir "$OUT/traces" \
  --out_dir "$OUT" >> "$LOG" 2>&1
echo "--- verdicts ---"
grep -a "\[VERDICT\]" "$LOG"
grep -acE "\[PROBE\]" "$LOG" >/dev/null && grep -a "\[PROBE\]" "$LOG" | tail -20
grep -a -m1 -A14 "Traceback" "$LOG" | head -16
# Render the annotated grasp-cam overlay (frames + reach/width/a_rot panel per trial)
# so failure points can be reviewed as video. No GPU needed.
if ls "$OUT"/traces/trace_*.npz >/dev/null 2>&1; then
  # After a successful render the ~470 MB/trial frame npz-es have served their purpose
  # (numbers live in the log, frames in the mp4) — keep them only if the render failed.
  $PY course_mission/scripts/make_grasp_overlay.py \
    --trace_dir "$OUT/traces" --out "$OUT/grasp_overlay_$TAG.mp4" >> "$LOG" 2>&1 \
    && { echo "OVERLAY $OUT/grasp_overlay_$TAG.mp4"; rm -f "$OUT"/traces/trace_*.npz; }
fi
echo "PROBE_DONE_$TAG"
