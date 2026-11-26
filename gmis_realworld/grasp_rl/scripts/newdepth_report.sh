#!/bin/bash
# Waits for both new-depth pipelines, then writes grasp_rl/detached/FINAL_REPORT_NEWDEPTHS.txt
set -u
RW=/hdd0/yewon/ATEC2026_Simulation_Challenge/real_world_preliminary/ATEC2026_Simulation_Challenge
D=$RW/grasp_rl/detached
until [ -f "$D/newdepth_gpu5.done" ] && [ -f "$D/newdepth_gpu5b.done" ] && [ -f "$D/newdepth_gpu4.done" ]; do sleep 120; done
OUT=$D/FINAL_REPORT_NEWDEPTHS.txt
{
  echo "=== B2+Piper grasp: NEW crouch depths 0.21/0.23/0.25 ($(date)) ==="
  echo "(crouch: 3000 it warm-started from final_minju; grasp: 2000 it grasp1i resumed from"
  echo " the nearest grasp1h neighbor, v4 rules: dense keep-out, penalty -8, bank filter,"
  echo " 10-step grace, camera_hit fail 2 steps, self-collision OFF."
  echo " before = neighbor donor zero-shot at the new depth, after = grasp1i finetune)"
  echo
  for depth in 0.21 0.23 0.25; do
    echo "--- depth $depth"
    grep -hE 'GATE_OK|GATE_FAIL' $D/newdepth_gpu5.log $D/newdepth_gpu5b.log $D/newdepth_gpu4.log 2>/dev/null | grep " $depth" | head -1 | sed 's/^/  /'
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
echo "REPORT_NEWDEPTHS_WRITTEN $OUT"
