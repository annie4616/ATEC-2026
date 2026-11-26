#!/bin/bash
# Sweep how HARD the bag is gripped, which the first sweep never touched: every one of its
# 8 settings asked the fingers to stop 8 mm apart around a 3 mm wall, i.e. to close on air.
#   SQ = how far PAST the wall thickness the fingers are told to close [m]
#   FR = bag surface friction (the hold is entirely friction; there is no wrap)
set -u
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
for c in "0.002:1.4:sq2_fr14" "0.003:1.4:sq3_fr14" "0.002:2.0:sq2_fr20" \
         "0.003:2.0:sq3_fr20" "0.001:1.4:sq1_fr14"; do
  sq=${c%%:*}; rest=${c#*:}; fr=${rest%%:*}; tag=${rest#*:}
  CASES="0.003,0.030:0.130,0.055,0.060:$tag" SQ=$sq FR=$fr GPU=${GPU:-0} \
    bash $RW/course_mission/scripts/sweep_bag_grip.sh > /dev/null 2>&1
  f=$RW/course_mission/logs/bagsweep/bag_$tag.log
  printf "  %-12s squeeze=%s friction=%s  %s  최근접 %s\n" "$tag" "$sq" "$fr" \
    "$(grep -aoE 'grasp [0-9]+/[0-9]+' $f | tail -1)" \
    "$(grep -aoE 'closest TCP-object [0-9.]+' $f | awk '{print $3}' | sort -n | head -1)"
done
echo SQUEEZE_SWEEP_DONE
