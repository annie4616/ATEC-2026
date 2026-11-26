#!/bin/bash
# Deployment-realism grasp probe. One TAG per configuration so several can run at once
# without fighting over a log file (run_probe.sh hardcodes logs/probe.log).
#
#   TAG=freeze MISSION_GRASP_FREEZE=1 bash scripts/run_deploy_probe.sh
#
# Everything it varies is an env var read by mission.py / runner.py, all default-off, so
# TAG=base is the current shipping behaviour and is the control for the others.
set -u
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=/home/yewon/.conda/envs/isaaclab/bin/python
TAG=${TAG:?set TAG}
OUT=$RW/course_mission/videos_deploy/$TAG
LOG=$RW/course_mission/logs/deploy_probe_$TAG.log
mkdir -p "$OUT" "$RW/course_mission/logs"
# NEVER clobber a finished probe (the venue-run lesson: an inherited OUT rewrote a
# scoring run's log and only a hand copy survived).
if [ -s "$LOG" ]; then
  echo "REFUSING: $LOG already holds a run; move it aside first"; exit 1
fi
source $RW/eunha_rl/scripts/eunha_env.sh ${EUNHA_TAG:-s0821}
cd $RW
{
  echo "=== deploy probe $TAG ==="
  echo "MISSION_ARM_RATE=${MISSION_ARM_RATE:-} MISSION_GRIP_RATE=${MISSION_GRIP_RATE:-}"
  echo "MISSION_ARM_DELAY=${MISSION_ARM_DELAY:-} MISSION_GRASP_FREEZE=${MISSION_GRASP_FREEZE:-}"
  echo "MISSION_GRASP_DET_SEED=${MISSION_GRASP_DET_SEED:-} MISSION_GRASP_HOLD=${MISSION_GRASP_HOLD:-} MISSION_IK_FRAME_FIX=${MISSION_IK_FRAME_FIX:-} SEED=${SEED:-0}"
  echo "MISSION_GRASP_POS_CAP=${MISSION_GRASP_POS_CAP:-} MISSION_GRASP_ROT_CAP=${MISSION_GRASP_ROT_CAP:-}"
  echo "walk=$EUNHA_WALK_CKPT adapt=$EUNHA_ADAPT_CKPT act=$EUNHA_ADAPT_ACT"
} > "$LOG"
CUDA_VISIBLE_DEVICES=${GPU:-0} timeout --signal=KILL ${TMO:-7200} $PY \
  course_mission/scripts/probe_manip.py --headless --device cuda:0 \
  --seed ${SEED:-0} ${GRASP_CKPT:+--grasp_ckpt "$GRASP_CKPT"} --trials "${TRIALS:-can:grass,can:mud,rind:grass,rind:mud,lunchbox:mud,lunchbox:grass}" \
  --out_dir "$OUT" >> "$LOG" 2>&1
echo "--- verdicts ---"
grep -a "\[VERDICT\]" "$LOG"
grep -acE "\[PROBE\]" "$LOG" >/dev/null && grep -a "\[PROBE\]" "$LOG" | tail -20
grep -a -m1 -A14 "Traceback" "$LOG" | head -16
echo "DEPLOY_PROBE_DONE_$TAG"
