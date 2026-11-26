#!/bin/bash
# Waits for both v4 final pipelines, then writes grasp_rl/detached/FINAL_REPORT_V4.txt
set -u
RW=/hdd0/yewon/ATEC2026_Simulation_Challenge/real_world_preliminary/ATEC2026_Simulation_Challenge
D=$RW/grasp_rl/detached
until [ -f "$D/final_v4_gpu5.done" ] && [ -f "$D/final_v4_gpu4.done" ]; do sleep 120; done
OUT=$D/FINAL_REPORT_V4.txt
{
  echo "=== B2+Piper grasp: v4 keep-out report ($(date)) ==="
  echo "(v4 = v3 dense metric + RSI bank keep-out filter + 10-step camera_hit grace;"
  echo " before=v3/grasp1g, after=v4/grasp1h, BOTH measured under the v4 rules)"
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
echo "REPORT_V4_WRITTEN $OUT"
