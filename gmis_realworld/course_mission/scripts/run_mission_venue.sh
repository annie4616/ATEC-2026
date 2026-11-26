#!/bin/bash
# 30-minute venue profile (90,000 steps). MISSION_HARVEST=near|far|all selects how many
# of the far-from-the-bins object copies to skip; see solution_v5_venue.py.
set -u
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=/home/yewon/.conda/envs/isaaclab/bin/python
OUT=${OUT:-$RW/course_mission/videos_venue/$(date +%m%d_%H%M)${TAG:-}}
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
# one log per run, stored beside its videos (see run_mission_persist.sh)
LOG=$OUT/mission.log
ln -sfn "$LOG" "$RW/course_mission/logs/mission_venue${TAG:-}_latest.log"
source $RW/eunha_rl/scripts/eunha_env.sh ${EUNHA_TAG:-s0821}
if [ "${NO_VIDEO:-1}" = "1" ]; then export MISSION_NO_VIDEO=1; else unset MISSION_NO_VIDEO; fi
if [ "${ORACLE:-0}" = "1" ]; then export MISSION_ORACLE_DETECT=1; else unset MISSION_ORACLE_DETECT; fi
export MISSION_HARVEST=${HARVEST:-all}   # "all" ships; near/far are A/B only (see solution_v5_venue.py)
# snapshot AFTER the env vars are exported — PROVENANCE records the environment the
# run actually used, and recording it before the exports left that section blank.
bash "$RW/course_mission/scripts/snapshot_src.sh" "$OUT"
echo "mode: ${MISSION_ORACLE_DETECT:+ORACLE}${MISSION_ORACLE_DETECT:-CAMERA}  harvest: $MISSION_HARVEST"
cd $RW
CUDA_VISIBLE_DEVICES=${GPU:-0} timeout --signal=KILL ${TMO:-43200} $PY \
  course_mission/solution_v5_venue.py --headless --device cuda:0 \
  --out_dir "$OUT" ${EXTRA:-} > $LOG 2>&1
SCORE=$(grep -ao "score (과제2) = [0-9]* / [0-9]*" "$LOG" | tail -1 | grep -o "[0-9]* / [0-9]*")
STEPS=$(grep -ao '"steps": [0-9]*' "$OUT"/report_venue.json 2>/dev/null | grep -o '[0-9]*')
echo "$(basename "$OUT")  ${SCORE:-unfinished}  ${STEPS:-?}steps  harvest=$MISSION_HARVEST  ${MISSION_ORACLE_DETECT:+oracle}${MISSION_ORACLE_DETECT:-camera}" \
     > "$OUT/RESULT.txt"
cat "$OUT/RESULT.txt"
bash "$RW/course_mission/scripts/verify_video.sh" "$OUT"
grep -aE "=== score|unload |search done|venue clock|harvest=" $LOG | tail -20
grep -a -m1 -A14 "Traceback" $LOG | head -16
echo "MISSION_DONE_venue"
