#!/bin/bash
# Frozen-estimate eval ladder (2026-09-07, v2 — preset-based): success rate of a
# checkpoint under the sensing the venue actually allows — estimate FROZEN at reset
# (GRASP_PN_HOLD huge), rgbl noise, bias per variant. Standing AND lying cans (the
# mission drops objects from 0.20 m; they settle both ways; training only saw standing).
# Environment = the champ's canonical eunha s075 preset (A2-Piper, its keep-out box,
# KOFILTER+GRACE) — evaluating in the wrong plant scored the champ 0-2% at any refresh.
#   CKPT=exp_runs/runs/frz12_*/model_1500.pt TAG=frz12m1500 BIAS=0.012,0.012,0.010 GPU=0 \
#     bash exp_runs/eval_frz.sh
set -u
RW=/hdd0/yewon/ATEC2026_Simulation_Challenge/ATEC_gmis
PY=/home/yewon/.conda/envs/isaaclab/bin/python
CKPT=${CKPT:?}; TAG=${TAG:?}; GPU=${GPU:?}
if [ "$GPU" != "0" ] && [ "$GPU" != "1" ]; then echo "GPU must be 0 or 1"; exit 1; fi
source "$RW/eunha_rl/scripts/eunha_env.sh" s075
export GRASP_RUNS_DIR=$RW/exp_runs/runs
export GRASP_PERC_NOISE=rgbl
export GRASP_PN_HOLD=${PN_HOLD:-100000}
export GRASP_PN_BIAS=${BIAS:-0.012,0.012,0.010}
LOG=$RW/exp_runs/logs/evalfrz_${TAG}.log
: > "$LOG"
cd "$RW"
# standing can (66 dia x 115 tall)
CUDA_VISIBLE_DEVICES=$GPU $PY grasp_rl/scripts/play_grasp.py --headless --device cuda:0 \
  --task ATEC-Grasp-B2Piper-Stage2-v0 --crouch_depth 0.24 --checkpoint "$CKPT" \
  --shape cylinder --num_envs 32 --steps 1200 \
  --width_range 0.066,0.066 --height_range 0.115,0.115 >> "$LOG" 2>&1
echo "== standing done ==" >> "$LOG"
# lying can: cylinder rolled 90 deg (axis horizontal)
CUDA_VISIBLE_DEVICES=$GPU $PY grasp_rl/scripts/play_grasp.py --headless --device cuda:0 \
  --task ATEC-Grasp-B2Piper-Stage2-v0 --crouch_depth 0.24 --checkpoint "$CKPT" \
  --shape cylinder --num_envs 32 --steps 1200 --object_quat 0.7071,0.7071,0,0 \
  --width_range 0.066,0.066 --height_range 0.115,0.115 >> "$LOG" 2>&1
echo "== lying done ==" >> "$LOG"
grep -aE "rollout:|terminations:" "$LOG" | tail -4
