#!/bin/bash
# Full new-depth chain for ONE gpu: wait for its crouch trainings -> per depth
# [pose extraction (flock: shared json) -> gate -> pregrasp bank] -> parallel grasp
# finetunes (v4 rules) -> keep-out verification + 3-video suite per depth.
# Usage: newdepth_pipeline.sh <gpu> <crouch_done_marker> <depth:resume_ckpt:run_name> [...]
set -u
RW=/hdd0/yewon/ATEC2026_Simulation_Challenge/real_world_preliminary/ATEC2026_Simulation_Challenge
PY=/home/yewon/.conda/envs/isaaclab/bin/python
GPU=${1:?gpu}; MARKER=${2:?crouch done marker}; shift 2
D=$RW/grasp_rl/detached
until [ -f "$D/$MARKER.done" ]; do sleep 120; done
export GRASP_BANK_KOFILTER=1
export GRASP_CAMHIT_GRACE=10
cd $RW

# ---- per depth: settled pose -> gate -> pregrasp bank (sequential on this GPU) ----
for spec in "$@"; do
  IFS=: read -r depth resume rname <<< "$spec"
  crun=$(ls -td $RW/logs/rsl_rl/crouch_sweep/runs/crouch${depth}_2* | head -1)
  if [ ! -f "$crun/model_2999.pt" ]; then echo "GATE_FAIL depth=$depth: no model_2999 in $crun"; exit 1; fi
  ( flock 9
    CUDA_VISIBLE_DEVICES=$GPU $PY grasp_rl/scripts/extract_crouch_pose.py --headless --device cuda:0 \
      --skip_scan --extra_runs "${depth}:$(basename "$crun")" > $D/pose_${depth}.log 2>&1
  ) 9> $D/.pose.lock
  $PY grasp_rl/scripts/check_crouch_pose.py "$depth" || exit 1
  CUDA_VISIBLE_DEVICES=$GPU $PY grasp_rl/scripts/build_reach_map.py --headless --device cuda:0 \
    --depth "$depth" > $D/bank_${depth}.log 2>&1
  grep -E 'pregrasp bank:' $D/bank_${depth}.log || { echo "GATE_FAIL bank ${depth}"; exit 1; }
done

# ---- parallel grasp finetunes on this GPU ----
PIDS=()
for spec in "$@"; do
  IFS=: read -r depth resume rname <<< "$spec"
  CUDA_VISIBLE_DEVICES=$GPU $PY grasp_rl/scripts/train_grasp.py --headless --device cuda:0 \
    --task ATEC-Grasp-B2Piper-Stage2-v0 --crouch_depth "$depth" --num_envs 1024 \
    --max_iterations 2000 --run_name "$rname" \
    --width_range 0.030,0.066 --length_range 0.030,0.250 --height_range 0.020,0.130 \
    --resume "$resume" > $D/train_${rname}.log 2>&1 &
  PIDS+=($!)
  sleep 90
done
wait "${PIDS[@]}"

# ---- verification (before = the neighbor-depth donor, zero-shot) + video suites ----
ARGS=()
for spec in "$@"; do
  IFS=: read -r depth resume rname <<< "$spec"
  grun=$(ls -td $RW/grasp_rl/runs/${rname}_2* | head -1)
  gckpt=$grun/$(ls "$grun" | grep -oE "model_[0-9]+.pt" | sort -t_ -k2 -n | tail -1)
  echo "newdepth: depth $depth after=$gckpt"
  ARGS+=("$depth:$resume:$gckpt")
done
bash $RW/grasp_rl/scripts/verify_keepout.sh "$GPU" "${ARGS[@]}"
echo "NEWDEPTH_PIPELINE_DONE gpu=$GPU"
