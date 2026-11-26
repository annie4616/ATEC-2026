#!/bin/bash
# MISSION_GRASP_ROT_CAP: the top-ranked fix from the thin-grasp diagnosis. The knob has
# existed since mission.py:52 and has NEVER been set in any script or run. Over 167
# instrumented trials the policy's own wrist-rotation output separates outcomes perfectly:
#   |a_rot| median >= 27  ->  0 successes in 75 trials, 67 of them "never reached it"
#   |a_rot| median <  27  ->  33 successes in 92
# and 45 % of all attempts land in that dead band. Capping the rotation component of the
# per-step IK command is one env var, and it should help EVERY class, not just the bag.
set -u
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=/home/yewon/.conda/envs/isaaclab/bin/python
L=$RW/course_mission/logs/rotcap; mkdir -p "$L"
for cap in 0 0.60 0.35; do
  tag=$(echo "cap$cap" | tr -d .)
  out=$L/$tag.log
  ( source $RW/eunha_rl/scripts/eunha_env.sh ${EUNHA_TAG:-s0821}
    cd $RW
    COURSE_VENUE=hk MISSION_VENUE_PHOTO=1 MISSION_GRASP_ROT_CAP=$cap \
    CUDA_VISIBLE_DEVICES=${GPU:-0} timeout --signal=KILL 3600 $PY \
      course_mission/scripts/probe_manip.py --headless --device cuda:0 \
      --trials "can:grass,rind:grass,lunchbox:grass,redbag:grass,redbag:mud,can:mud" \
      --out_dir $RW/course_mission/videos_rotcap_$tag > "$out" 2>&1 )
  printf "  ROT_CAP=%-5s %s\n" "$cap" "$(grep -aoE 'stable [0-9]+/[0-9]+ \| grasp [0-9]+/[0-9]+ \| basket [0-9]+/[0-9]+' $out | tail -1)"
  grep -aoE "finger width at closure [0-9.]+ mm|never closed" "$out" | sort | uniq -c | sed 's/^/      /'
done
echo ROTCAP_DONE
