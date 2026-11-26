#!/bin/bash
# Does starting the grasp policy from the bank's own JOINT CONFIGURATION (not merely its
# position) recover the reach the Cartesian approach lost?
#
# Background: rewriting the hand-over as a Cartesian IK descent stopped the arm knocking
# the bag over, but moved the dominant failure to "never reached it" (3 of 4). IK has a
# null space, so the same fingertip position is reachable with the elbow anywhere on a
# circle, and the policy was trained from bank CONFIGURATIONS.
#
#   MISSION_BANK_SETTLE   steps of joint-space blend onto q0 AFTER the TCP has arrived.
#                         0 = position only (what measured 0/4 with 3 never-reached).
#
# Six objects per configuration rather than four: the previous sweep had the CAN fail in
# its baseline too, so at n=4 the noise swamps the effect being measured.
set -u
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=/home/yewon/.conda/envs/isaaclab/bin/python
L=$RW/course_mission/logs/banksettle; mkdir -p "$L"
TRIALS="redbag:grass,redbag:mud,redbag:pebble,can:grass,can:mud,rind:grass"
for c in "0:settle0" "40:settle40" "90:settle90"; do
  st=${c%%:*}; tag=${c#*:}
  out=$L/$tag.log
  [ -s "$out" ] && grep -q PROBE_END "$out" && { echo "SKIP $tag"; continue; }
  ( source $RW/eunha_rl/scripts/eunha_env.sh ${EUNHA_TAG:-s0821}
    cd $RW
    COURSE_VENUE=hk MISSION_VENUE_PHOTO=1 \
    MISSION_APPROACH_CLEAR=0.14 MISSION_GRASP_POS_CAP=0.030 MISSION_BANK_SETTLE=$st \
    CUDA_VISIBLE_DEVICES=${GPU:-0} timeout --signal=KILL 5400 $PY \
      course_mission/scripts/probe_manip.py --headless --device cuda:0 \
      --trials "$TRIALS" --out_dir $RW/course_mission/videos_bs_$tag > "$out" 2>&1
    echo PROBE_END >> "$out" )
  printf "\n  === settle=%s ===  %s\n" "$st" \
    "$(grep -aoE 'stable [0-9]+/[0-9]+ \| grasp [0-9]+/[0-9]+ \| basket [0-9]+/[0-9]+' $out | tail -1)"
  grep -aE "^\[PROBE\] trash_" "$out" | sed 's/\[PROBE\] /      /'
  grep -aoE "grasp FAILED [a-z_0-9]+: .*-> .*" "$out" | sed 's/.*-> /      · /' | sort | uniq -c
done
echo BANKSETTLE_DONE
