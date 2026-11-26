#!/bin/bash
# "천천히 다가가고 IK 계산하면서" — the two knobs that implements, swept together.
#   MISSION_APPROACH_CLEAR  how far ABOVE the object the IK descent starts [m], 0 = old
#                           joint-space ramp that swept the hand through the bag
#   MISSION_GRASP_POS_CAP   metres/step cap on the grasp policy's own translation command.
#                           Uncapped it commands 70-102 mm per step, which is a third of a
#                           standing bag's height in one tick.
set -u
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=/home/yewon/.conda/envs/isaaclab/bin/python
L=$RW/course_mission/logs/slowapp; mkdir -p "$L"
for c in "0:0:baseline" "0.14:0:ikonly" "0.14:0.030:ik_slow30" "0.14:0.015:ik_slow15"; do
  cl=${c%%:*}; rest=${c#*:}; cap=${rest%%:*}; tag=${rest#*:}
  out=$L/$tag.log
  ( source $RW/eunha_rl/scripts/eunha_env.sh ${EUNHA_TAG:-s0821}
    cd $RW
    COURSE_VENUE=hk MISSION_VENUE_PHOTO=1 \
    MISSION_APPROACH_CLEAR=$cl MISSION_GRASP_POS_CAP=$cap \
    CUDA_VISIBLE_DEVICES=${GPU:-0} timeout --signal=KILL 3600 $PY \
      course_mission/scripts/probe_manip.py --headless --device cuda:0 \
      --trials "redbag:grass,redbag:mud,redbag:pebble,can:grass" \
      --out_dir $RW/course_mission/videos_slow_$tag > "$out" 2>&1 )
  printf "  %-11s clear=%-5s poscap=%-5s  %s\n" "$tag" "$cl" "$cap" \
    "$(grep -aoE 'stable [0-9]+/[0-9]+ \| grasp [0-9]+/[0-9]+ \| basket [0-9]+/[0-9]+' $out | tail -1)"
  grep -aoE "grasp FAILED [a-z_0-9]+: .*-> .*" "$out" | sed 's/.*-> /      /' | sort | uniq -c
done
echo SLOWAPP_DONE
