#!/bin/bash
# Sweep the paper-bag wall pinch. The bag is now an open-top thin-walled box
# (a2pro_rl/scripts/build_prop_usds.build_bag), so there is a single 3 mm layer to grip
# instead of the 55 mm void the old sealed model presented. These are the numbers that
# decide whether that layer is actually grippable.
#   MISSION_BAG_GRIP = "<finger width target>,<pinch depth below the rim>"
#   MISSION_BAG_OBB  = "<l>,<w>,<h>" told to the grasp policy
set -u
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=/home/yewon/.conda/envs/isaaclab/bin/python
GPU=${GPU:-0}
L=$RW/course_mission/logs/bagsweep
mkdir -p "$L"
# grip(w,d) : obb(l,w,h) : tag
CASES=${CASES:-"
0.003,0.030:0.130,0.010,0.060:base
0.003,0.020:0.130,0.010,0.060:shallow
0.003,0.045:0.130,0.010,0.060:deep
0.006,0.030:0.130,0.020,0.060:thickbite
0.003,0.030:0.130,0.055,0.060:oldobb
"}
for c in $CASES; do
  grip=${c%%:*}; rest=${c#*:}; obb=${rest%%:*}; tag=${rest#*:}
  out=$L/bag_$tag.log
  [ -s "$out" ] && grep -q PROBE_DONE "$out" && { echo "SKIP $tag (already done)"; continue; }
  ( source $RW/eunha_rl/scripts/eunha_env.sh ${EUNHA_TAG:-s0821}
    cd $RW
    COURSE_VENUE=hk MISSION_VENUE_PHOTO=1 \
    MISSION_BAG_GRIP=$grip MISSION_BAG_OBB=$obb \
    MISSION_SQUEEZE=${SQ:-0.002} MISSION_BAG_FRICTION=${FR:-1.4} \
    CUDA_VISIBLE_DEVICES=$GPU timeout --signal=KILL 3600 $PY \
      course_mission/scripts/probe_manip.py --headless --device cuda:0 \
      --trials "redbag:grass,redbag:mud,redbag:pebble" \
      --out_dir $RW/course_mission/videos_bag_$tag > "$out" 2>&1
    echo PROBE_DONE >> "$out" )
  echo "=== $tag  grip=$grip obb=$obb ==="
  grep -aE "\[PROBE\]" "$out" | tail -6
done
echo BAGSWEEP_DONE
