#!/bin/bash
# The real thing: camera detection, no oracle. Everything the robot knows about object
# positions comes from the head camera.
set -u
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=/home/yewon/.conda/envs/isaaclab/bin/python
PROFILE=${PROFILE:-v2_balanced}
source $RW/eunha_rl/scripts/eunha_env.sh ${EUNHA_TAG:-s0821}
unset MISSION_ORACLE_DETECT
OUT=${OUT:-$RW/course_mission/videos_auto/$(date +%m%d_%H%M)}
mkdir -p "$OUT"
bash "$RW/course_mission/scripts/snapshot_src.sh" "$OUT"
# one log per run, stored beside its videos (see run_mission_persist.sh)
LOG=$OUT/mission.log
ln -sfn "$LOG" "$RW/course_mission/logs/mission_auto_latest.log"
echo "output dir: $OUT"
cd $RW
CUDA_VISIBLE_DEVICES=${GPU:-0} timeout --signal=KILL ${TMO:-21600} $PY \
  course_mission/solution_${PROFILE}.py --headless --device cuda:0 \
  --out_dir "$OUT" ${EXTRA:-} > $LOG 2>&1
grep -aE "\[MISSION\]|=== score" $LOG | tail -40
grep -a -m1 -A14 "Traceback" $LOG | head -16
echo "MISSION_DONE_auto"
