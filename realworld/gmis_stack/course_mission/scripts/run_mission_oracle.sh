#!/bin/bash
# Perception ablation: MISSION_ORACLE_DETECT=1 replaces the camera detector with
# ground-truth positions + 1.5 cm noise. Everything else (search drive, crouch gate,
# RL grasp, basket deposit, bin sorting) runs for real.
set -u
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=/home/yewon/.conda/envs/isaaclab/bin/python
PROFILE=${PROFILE:-v2_balanced}
source $RW/eunha_rl/scripts/eunha_env.sh ${EUNHA_TAG:-s0821}
export MISSION_ORACLE_DETECT=1
OUT=${OUT:-$RW/course_mission/videos_oracle/$(date +%m%d_%H%M)}
mkdir -p "$OUT"
bash "$RW/course_mission/scripts/snapshot_src.sh" "$OUT"
# one log per run, stored beside its videos (see run_mission_persist.sh)
LOG=$OUT/mission.log
ln -sfn "$LOG" "$RW/course_mission/logs/mission_oracle_latest.log"
echo "output dir: $OUT"
cd $RW
CUDA_VISIBLE_DEVICES=${GPU:-0} timeout --signal=KILL ${TMO:-21600} $PY \
  course_mission/solution_${PROFILE}.py --headless --device cuda:0 \
  --out_dir "$OUT" ${EXTRA:-} > $LOG 2>&1
grep -E "\[MISSION\]|=== score" $LOG | tail -40
grep -m1 -A14 "Traceback" $LOG | head -16
echo "MISSION_DONE_oracle"
