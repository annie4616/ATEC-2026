#!/bin/bash
# Full graspability matrix for a given checkpoint (arg 1), logs to arg 2 (dir).
# Covers: compact/elongated primitives regression, competition-prop stand-ins (can, rind),
# and the task_b USD objects in their graspable orientations.
set -u
RW=/hdd0/yewon/ATEC2026_Simulation_Challenge/real_world_preliminary/ATEC2026_Simulation_Challenge
CKPT=${1:?checkpoint path required}
OUT=${2:?output log dir required}
DEPTH=${3:-0.2}
GPU=${4:-5}
PY=/home/yewon/.conda/envs/isaaclab/bin/python
PLAY=$RW/grasp_rl/scripts/play_grasp.py
mkdir -p "$OUT"
cd $RW

run() {  # name, extra args...
  local name=$1; shift
  echo "=== $name ==="
  CUDA_VISIBLE_DEVICES=$GPU $PY $PLAY --headless --device cuda:0 --task ATEC-Grasp-B2Piper-Stage2-v0 \
    --crouch_depth $DEPTH --num_envs 8 --steps 400 --checkpoint "$CKPT" "$@" > "$OUT/$name.log" 2>&1
  echo "exit=$?"
}

run compact_30_60      --width_range 0.030,0.060 --length_range 0.030,0.060 --height_range 0.060,0.130
run elongated          --width_range 0.030,0.060 --length_range 0.100,0.250 --height_range 0.060,0.130
run can_66x115         --width_range 0.066,0.066 --length_range 0.066,0.066 --height_range 0.115,0.115 --shape cylinder
run rind_flat          --width_range 0.050,0.050 --length_range 0.070,0.100 --height_range 0.025,0.040
run usd_banana         --object_usd $RW/atec_robot_model/objects/task_b/011_banana.usd
run usd_sugar_rot      --object_usd $RW/atec_robot_model/objects/task_b/004_sugar_box.usd --object_quat 0.7071,0,0.7071,0
run usd_mustard_rot    --object_usd $RW/atec_robot_model/objects/task_b/006_mustard_bottle.usd --object_quat 0.7071,0,0.7071,0
run usd_cracker_rot    --object_usd $RW/atec_robot_model/objects/task_b/003_cracker_box.usd --object_quat 0.7071,0,0.7071,0
echo MATRIX_DONE
