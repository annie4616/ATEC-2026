#!/bin/bash
# Diagnose the 0.25 crouch settle failure: settled height per training checkpoint
# (model_0 = donor weights) x hold duration (3s vs 10s). Writes probe jsons under
# grasp_rl/banks/probe_runs/ (main crouch_poses.json untouched).
set -u
RW=/hdd0/yewon/ATEC2026_Simulation_Challenge/real_world_preliminary/ATEC2026_Simulation_Challenge
PY=/home/yewon/.conda/envs/isaaclab/bin/python
RUN=$RW/logs/rsl_rl/crouch_sweep/runs/crouch0.25_2026-07-28_03-28-37
PR=$RW/grasp_rl/banks/probe_runs
rm -rf "$PR"; mkdir -p "$PR"
SPECS=""
for n in 0 500 1000 1500 2000 2500 2999; do
  [ -f "$RUN/model_$n.pt" ] || continue
  mkdir -p "$PR/ck$n"; cp "$RUN/model_$n.pt" "$PR/ck$n/"
  SPECS="${SPECS:+$SPECS,}$n:ck$n"
done
cd $RW
for T in 3 10; do
  CUDA_VISIBLE_DEVICES=4 $PY grasp_rl/scripts/extract_crouch_pose.py --headless --device cuda:0 \
    --runs_root "$PR" --skip_scan --extra_runs "$SPECS" --crouch_time "$T" \
    --out "$PR/hold_$T.json" > "$PR/hold_$T.log" 2>&1
done
$PY - <<'EOF'
import json
PR = "/hdd0/yewon/ATEC2026_Simulation_Challenge/real_world_preliminary/ATEC2026_Simulation_Challenge/grasp_rl/banks/probe_runs"
for T in (3, 10):
    try:
        p = json.load(open(f"{PR}/hold_{T}.json"))["poses"]
    except Exception as e:
        print(f"PROBE hold_{T}: FAILED to read ({e})")
        continue
    row = " ".join(f"it{k}={v['base_height_mean']:.3f}" for k, v in sorted(p.items(), key=lambda kv: float(kv[0])))
    print(f"PROBE hold {T}s: {row}")
EOF
echo "PROBE_025_DONE"
