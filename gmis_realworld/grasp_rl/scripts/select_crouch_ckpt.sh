#!/bin/bash
# Gate-based checkpoint selection for a crouch run: training at targets at/above the
# donor's settle drifts the settle DOWN over iterations, so the final checkpoint often
# fails the gate while an intermediate one holds closest to target. Measures the settled
# height (strict 10 s hold) of a checkpoint ladder, picks argmin |settle - depth|, and
# packages the winner as a fresh crouch<depth>_<stamp>/model_2999.pt run dir.
# Usage: select_crouch_ckpt.sh <gpu> <depth> <run_dir>
set -u
RW=/hdd0/yewon/ATEC2026_Simulation_Challenge/real_world_preliminary/ATEC2026_Simulation_Challenge
PY=/home/yewon/.conda/envs/isaaclab/bin/python
GPU=${1:?gpu}; DEPTH=${2:?depth}; RUN=${3:?run dir}
RUNS=$RW/logs/rsl_rl/crouch_sweep/runs
PR=$RW/grasp_rl/banks/select_${DEPTH}
rm -rf "$PR"; mkdir -p "$PR"
SPECS=""
for n in 0 300 600 900 1200 1500 2000 2500 2999; do
  [ -f "$RUN/model_$n.pt" ] || continue
  mkdir -p "$PR/ck$n"; cp "$RUN/model_$n.pt" "$PR/ck$n/"
  SPECS="${SPECS:+$SPECS,}$n:ck$n"
done
cd $RW
CUDA_VISIBLE_DEVICES=$GPU $PY grasp_rl/scripts/extract_crouch_pose.py --headless --device cuda:0 \
  --runs_root "$PR" --skip_scan --extra_runs "$SPECS" --crouch_time 10 \
  --out "$PR/sel.json" > "$PR/sel.log" 2>&1
BEST=$($PY - "$DEPTH" "$PR/sel.json" <<'EOF'
import json, sys
depth = float(sys.argv[1])
poses = json.load(open(sys.argv[2]))["poses"]
rows = sorted(((abs(v["base_height_mean"] - depth), -int(k), k, v["base_height_mean"]) for k, v in poses.items()))
d, _, k, z = rows[0]
print(f"{k} {z:.4f} {'PASS' if d <= 0.03 else 'FAIL'}")
for _, _, kk, zz in sorted(rows, key=lambda r: int(r[2])):
    print(f"  it{kk}: {zz:.3f}", file=sys.stderr)
EOF
)
read -r IT Z VERDICT <<< "$BEST"
echo "SELECT depth=$DEPTH best=it$IT z=$Z verdict=$VERDICT"
if [ "$VERDICT" != "PASS" ]; then echo "SELECT_FAIL depth=$DEPTH (no checkpoint within +/-0.03)"; exit 1; fi
FB=$RUNS/crouch${DEPTH}_$(date +%Y-%m-%d_%H-%M-%S)
mkdir -p "$FB"; cp "$RUN/model_$IT.pt" "$FB/model_2999.pt"
echo "SELECT_OK depth=$DEPTH iter=$IT z=$Z packaged=$FB"
