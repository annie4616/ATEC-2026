#!/bin/bash
# Frozen-estimate grasp finetune launcher (2026-09-07, v2 — preset-based).
#
# WHY. The honest mission (OBJ_HONEST=1) freezes the grasp estimate at the pre-crouch
# detection (NEAR_CUT + arm occlusion: true refresh rate is zero) and the shipping champ
# needs 10 Hz refresh: 6/6 refreshed vs 1/8 frozen, 0/592 in honest missions. This
# finetunes the champ under frozen sensing (GRASP_PN_HOLD huge) with a bias curriculum
# (GRASP_PN_BIAS env knob added to graspenv observations.py the same day).
#
# ENVIRONMENT CONTRACT — LEARNED THE HARD WAY. The champ egxs075_c024 is an
# A2-PIPER policy trained under eunha_rl/scripts/eunha_env.sh: robot a2piper, the
# 53-obs RMA legs with the s075 artifact set (elu adaptation per s075/adapt_act.txt),
# keep-out box 0.218..0.438 (NOT graspenv's 0.30..0.52 default), grip open 0.05,
# KOFILTER+GRACE. Recreating that by hand missed the robot itself (B2 spawned, 74%
# camera_hit, champ 0-2% at ANY refresh rate), so: SOURCE THE PRESET, then override
# only what this experiment is about.
#
#   TAG=frz12 GPU=0 bash exp_runs/launch_frzft.sh                 # stock 12mm bias
#   TAG=frz25far GPU=0 PN_BIAS=0.025,0.025,0.020 GRASP_RSI_FAR=0.25 ...
#   TAG=frz50far GPU=1 PN_BIAS=0.050,0.050,0.030 GRASP_RSI_FAR=0.25 ...
#   TAG=hold40 GPU=1 PN_HOLD=40 PN_BIAS=0.025,0.025,0.020 ...
#
# GPUs 0 and 1 ONLY (2/3 are teammates'). Logs: exp_runs/logs/train_<TAG>.log
# Kill by process GROUP (kill -9 -<pgid>) — Isaac ignores TERM.
set -u
RW=/hdd0/yewon/ATEC2026_Simulation_Challenge/ATEC_gmis
PY=/home/yewon/.conda/envs/isaaclab/bin/python
TAG=${TAG:?set TAG}
GPU=${GPU:?set GPU (0 or 1 only)}
if [ "$GPU" != "0" ] && [ "$GPU" != "1" ]; then echo "GPU must be 0 or 1"; exit 1; fi

# the champ's canonical environment (robot, legs, keep-out, banks, grace)
source "$RW/eunha_rl/scripts/eunha_env.sh" s075
# outputs go HERE — eunha_rl/s075/runs is chmod a-w on purpose (champion artifacts)
export GRASP_RUNS_DIR=$RW/exp_runs/runs

# the experiment: frozen, biased estimate
export GRASP_PERC_NOISE=rgbl
export GRASP_PN_HOLD=${PN_HOLD:-100000}          # frozen at reset unless overridden
export GRASP_PN_BIAS=${PN_BIAS:-0.012,0.012,0.010}
export GRASP_PN_JITTER=${PN_JITTER:-0.004}

RESUME=${RESUME:-$RW/eunha_rl/s075/runs/egxs075_c024_2026-08-12_09-19-10/model_900.pt}
ITERS=${ITERS:-2000}
ENVS=${ENVS:-1024}
SEED=${SEED:-1}

mkdir -p "$RW/exp_runs/logs"
LOG=$RW/exp_runs/logs/train_${TAG}.log
{
  echo "=== frozen finetune $TAG (gpu $GPU) $(date) ==="
  echo "PN_HOLD=$GRASP_PN_HOLD PN_BIAS=$GRASP_PN_BIAS PN_JITTER=$GRASP_PN_JITTER"
  echo "ROBOT=$GRASP_ROBOT ADAPT_ACT=$EUNHA_ADAPT_ACT KEEPOUT=$GRASP_KEEPOUT_LO..$GRASP_KEEPOUT_HI"
  echo "resume=$RESUME iters=$ITERS envs=$ENVS seed=$SEED"
} > "$LOG"
cd "$RW"
CUDA_VISIBLE_DEVICES=$GPU exec $PY grasp_rl/scripts/train_grasp.py --headless --device cuda:0 \
  --task ATEC-Grasp-B2Piper-Stage2-v0 --crouch_depth 0.24 \
  --num_envs "$ENVS" --max_iterations "$ITERS" --seed "$SEED" --run_name "$TAG" \
  --width_range "${WRANGE:-0.030,0.066}" --length_range "${LRANGE:-0.030,0.250}" \
  --height_range "${HRANGE:-0.020,0.130}" \
  --resume "$RESUME" >> "$LOG" 2>&1
