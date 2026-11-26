#!/bin/bash
# Waits for both v3 final pipelines, then writes grasp_rl/detached/FINAL_REPORT_V3.txt
set -u
RW=/hdd0/yewon/ATEC2026_Simulation_Challenge/real_world_preliminary/ATEC2026_Simulation_Challenge
D=$RW/grasp_rl/detached
until [ -f "$D/final_v3_gpu5.done" ] && [ -f "$D/final_v3_gpu4.done" ]; do sleep 120; done
OUT=$D/FINAL_REPORT_V3.txt
{
  echo "=== B2+Piper grasp: v3 keep-out report ($(date)) ==="
  echo "(v3 = dense arm sampling incl. midpoints+TCP, shaped depth penalty margin 3.5cm,"
  echo " weight -8, camera_hit fail after 2 steps; before=v2/grasp1f, after=v3/grasp1g,"
  echo " BOTH measured under the new dense metric)"
  echo
  for depth in 0.2 0.22 0.24 0.3; do
    echo "--- depth $depth"
    for tag in before after; do
      f=$D/ko_${depth}_${tag}.log
      echo "  [$tag] $(grep -oE 'keep-out env-steps: [0-9]+' $f 2>/dev/null | head -1) | $(grep -oE 'rollout: .*' $f 2>/dev/null | head -1)"
      grep -oE "terminations: \{[^}]*\}" $f 2>/dev/null | head -1 | sed 's/^/         /'
    done
    for kind in basic can rind; do
      echo "  [video $kind] $(grep -oE 'rollout: .*' $RW/grasp_rl/videos/d${depth}_${kind}.log 2>/dev/null | head -1) -> videos/d${depth}_${kind}.mp4"
    done
  done
} > "$OUT"
echo "REPORT_V3_WRITTEN $OUT"
