#!/bin/bash
# Waits for both real-round finalizations, then writes grasp_rl/detached/FINAL_REPORT_REAL.txt
set -u
RW=/hdd0/yewon/ATEC2026_Simulation_Challenge/real_world_preliminary/ATEC2026_Simulation_Challenge
D=$RW/grasp_rl/detached
until [ -f "$D/real2_gpu5.done" ] && [ -f "$D/real2_gpu4.done" ]; do sleep 120; done
OUT=$D/FINAL_REPORT_REAL.txt
{
  echo "=== B2+Piper grasp: REAL-WORLD robustness round ($(date)) ==="
  echo "(RGB+LiDAR perception noise + proprioceptive sensor corruption (policy obs only,"
  echo " asymmetric critic) + dynamics randomization: object friction/mass, fingertip"
  echo " friction, arm PD gains. before = rgbl ladder winner evaluated UNDER the full"
  echo " real noise, after = grasp1m ladder winner. v4 keep-out rules, self-collision OFF.)"
  echo
  for depth in 0.2 0.21 0.22 0.23 0.24 0.25 0.3; do
    echo "--- depth $depth"
    for tag in before after; do
      f=$D/koreal_${depth}_${tag}.log
      echo "  [$tag] $(grep -oE 'keep-out env-steps: [0-9]+' $f 2>/dev/null | head -1) | $(grep -oE 'rollout: .*' $f 2>/dev/null | head -1)"
      grep -oE "terminations: \{[^}]*\}" $f 2>/dev/null | head -1 | sed 's/^/         /'
    done
    grep -hE "SELR_BEST depth=$depth " $D/real2_gpu5.log $D/real2_gpu4.log 2>/dev/null | head -1 | sed 's/^/  /'
    for kind in basic can rind; do
      echo "  [video $kind] $(grep -oE 'rollout: .*' $RW/grasp_rl/videos/real_d${depth}_${kind}.log 2>/dev/null | head -1) -> videos/real_d${depth}_${kind}.mp4"
    done
  done
} > "$OUT"
echo "REPORT_REAL_WRITTEN $OUT"
