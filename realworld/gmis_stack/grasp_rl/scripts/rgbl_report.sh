#!/bin/bash
# Waits for both rgbl pipelines, then writes grasp_rl/detached/FINAL_REPORT_RGBL.txt
set -u
RW=/hdd0/yewon/ATEC2026_Simulation_Challenge/real_world_preliminary/ATEC2026_Simulation_Challenge
D=$RW/grasp_rl/detached
until [ -f "$D/rgbl2_gpu5.done" ] && [ -f "$D/rgbl2_gpu4.done" ]; do sleep 120; done
OUT=$D/FINAL_REPORT_RGBL.txt
{
  echo "=== B2+Piper grasp: RGB+LiDAR perception round ($(date)) ==="
  echo "(assumption: real robot has RGB + LiDAR only, no depth camera. Object observations"
  echo " = 10 Hz world-frame estimate, per-episode bias xy<=1.2cm z<=1.0cm, jitter 4mm,"
  echo " OBB dims +/-12% — relative gripper-object error, same-camera bias cancels."
  echo " before = noise-free-trained ckpt evaluated UNDER noise, after = grasp1k"
  echo " noise-trained. v4 keep-out rules, self-collision OFF.)"
  echo
  for depth in 0.2 0.21 0.22 0.23 0.24 0.25 0.3; do
    echo "--- depth $depth"
    for tag in before after; do
      f=$D/korgbl_${depth}_${tag}.log
      echo "  [$tag] $(grep -oE 'keep-out env-steps: [0-9]+' $f 2>/dev/null | head -1) | $(grep -oE 'rollout: .*' $f 2>/dev/null | head -1)"
      grep -oE "terminations: \{[^}]*\}" $f 2>/dev/null | head -1 | sed 's/^/         /'
    done
    for kind in basic can rind; do
      echo "  [video $kind] $(grep -oE 'rollout: .*' $RW/grasp_rl/videos/rgbl_d${depth}_${kind}.log 2>/dev/null | head -1) -> videos/rgbl_d${depth}_${kind}.mp4"
    done
  done
} > "$OUT"
echo "REPORT_RGBL_WRITTEN $OUT"
