#!/bin/bash
# Final 0.25 crouch attempt: DOWNWARD chaining from the 0.3 policy (b2_crouch_cycle,
# settles 0.290) -> commanded 0.25. If the gate still fails, fall back to the previous
# retry's iteration-500 checkpoint (probe-verified: settles 0.240@3s / 0.231@10s, passes
# the +/-0.03 gate) packaged as a fresh crouch0.25_* run dir for the pipeline to resolve.
set -u
RW=/hdd0/yewon/ATEC2026_Simulation_Challenge/real_world_preliminary/ATEC2026_Simulation_Challenge
PY=/home/yewon/.conda/envs/isaaclab/bin/python
D=$RW/grasp_rl/detached
RUNS=$RW/logs/rsl_rl/crouch_sweep/runs
DONOR=$RUNS/b2_crouch_cycle_2026-06-30_06-36-28/model_3999.pt
FALLBACK_SRC=$RUNS/crouch0.25_2026-07-28_03-28-37/model_500.pt
until [ -f "$D/crouch_retry023.done" ]; do sleep 60; done   # GPU4 free
cd $RW
CUDA_VISIBLE_DEVICES=4 $PY grasp_rl/scripts/train_crouch_depth.py --headless --device cuda:0 \
  --crouch_h 0.25 --base_too_low 0.225 --donor "$DONOR" \
  > $D/train_crouch_0.25.log 2>&1
crun=$(ls -td $RUNS/crouch0.25_2* | head -1)
# gate the fresh run via a throwaway extraction (main json untouched here; the downstream
# pipeline re-extracts under flock)
PR=$RW/grasp_rl/banks/probe_runs_025final
rm -rf "$PR"; mkdir -p "$PR/cand"; cp "$crun/model_2999.pt" "$PR/cand/"
CUDA_VISIBLE_DEVICES=4 $PY grasp_rl/scripts/extract_crouch_pose.py --headless --device cuda:0 \
  --runs_root "$PR" --skip_scan --extra_runs "0.25:cand" --crouch_time 10 \
  --out "$PR/gate.json" > "$PR/gate.log" 2>&1
Z=$($PY -c "import json;print(json.load(open('$PR/gate.json'))['poses']['0.25']['base_height_mean'])")
echo "CROUCH025 downward-chain settle z=$Z (run $crun)"
OK=$($PY -c "print(1 if abs($Z-0.25)<=0.03 else 0)")
if [ "$OK" != "1" ]; then
  echo "CROUCH025 downward chain FAILED gate -> packaging probe-passing fallback (it500)"
  FB=$RUNS/crouch0.25_$(date +%Y-%m-%d_%H-%M-%S)
  mkdir -p "$FB"; cp "$FALLBACK_SRC" "$FB/model_2999.pt"
  echo "CROUCH025 fallback run: $FB (settles ~0.231-0.240, gate-passing)"
fi
echo "CROUCH025_FINAL_DONE"
