#!/bin/bash
# 120-point attempt: bags skipped, three cycles per object, big budget.
set -u
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=/home/yewon/.conda/envs/isaaclab/bin/python
OUT=${OUT:-$RW/course_mission/videos_persist/$(date +%m%d_%H%M)}
# NEVER clobber a finished run. OUT is inherited from the environment, so anything that
# re-enters this script with the same OUT lands on top of a previous result — and that
# nearly destroyed the 120/160 record: its mission.log was rewritten and only a hand-made
# copy survived. (The re-entry itself came from editing this file WHILE it was running:
# bash reads a script by byte offset, so an in-place edit makes a live shell resume at a
# shifted position and re-execute. Edit these scripts with write-to-temp + mv, which
# replaces the inode and leaves running shells on the old one.)
if [ -n "$(ls "$OUT"/report_*.json 2>/dev/null)" ]; then
  echo "REFUSING to overwrite $OUT — it already holds a finished run"
  OUT="${OUT}_$(date +%H%M%S)"
  echo "  writing to $OUT instead"
fi
mkdir -p "$OUT"
# The log belongs WITH the run it describes. Every run used to write the same
# logs/mission_persist<tag>.log, so the moment the next run started there was no way to
# say which video was the 100-point run and which was the 80. A stable *_latest symlink
# keeps the monitoring greps working.
LOG=$OUT/mission.log
ln -sfn "$LOG" "$RW/course_mission/logs/mission_persist${TAG:-}_latest.log"
source $RW/eunha_rl/scripts/eunha_env.sh ${EUNHA_TAG:-s0821}
# ORACLE=1 -> ground-truth detections (perception ablation); default = camera
if [ "${NO_VIDEO:-1}" = "1" ]; then export MISSION_NO_VIDEO=1; else unset MISSION_NO_VIDEO; fi
if [ "${ORACLE:-0}" = "1" ]; then export MISSION_ORACLE_DETECT=1; else unset MISSION_ORACLE_DETECT; fi
# snapshot AFTER the env vars are exported — PROVENANCE records the environment the
# run actually used, and recording it before the exports left that section blank.
bash "$RW/course_mission/scripts/snapshot_src.sh" "$OUT"
echo "mode: ${MISSION_ORACLE_DETECT:+ORACLE}${MISSION_ORACLE_DETECT:-CAMERA}"
cd $RW
CUDA_VISIBLE_DEVICES=${GPU:-0} timeout --signal=KILL ${TMO:-43200} $PY \
  course_mission/solution_v4_persist.py --headless --device cuda:0 \
  --out_dir "$OUT" ${EXTRA:-} > $LOG 2>&1
# leave a one-line summary in the run folder: "0817_1520  100/160  oracle"
SCORE=$(grep -ao "score (과제2) = [0-9]* / [0-9]*" "$LOG" | tail -1 | grep -o "[0-9]* / [0-9]*")
echo "$(basename "$OUT")  ${SCORE:-unfinished}  ${MISSION_ORACLE_DETECT:+oracle}${MISSION_ORACLE_DETECT:-camera}" \
     > "$OUT/RESULT.txt"
cat "$OUT/RESULT.txt"
bash "$RW/course_mission/scripts/verify_video.sh" "$OUT"
grep -aE "=== score|unload |search done" $LOG | tail -20
grep -a -m1 -A14 "Traceback" $LOG | head -16
echo "MISSION_DONE_persist"
