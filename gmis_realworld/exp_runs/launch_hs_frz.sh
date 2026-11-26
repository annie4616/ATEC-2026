#!/bin/bash
# Honest-mission validation of a frozen-retrained grasp checkpoint (2026-09-07).
# Full venue honesty: NAV_HONEST=1 OBJ_HONEST=1 (detection + grasp + verdict all
# sensor-only), strong-evidence hold verdict ON (default), post-crouch re-see ON
# (default, scoped NEAR_CUT 0.22).
#   TAG=fv0 GPU=0 CKPT=exp_runs/runs/frz12_XXX/model_1500.pt bash exp_runs/launch_hs_frz.sh
# CKPT is RW-relative (run_mission.sh prefixes $RW). Kill by process GROUP. GPUs 0/1 only.
set -u
RW=/hdd0/yewon/ATEC2026_Simulation_Challenge/ATEC_gmis
export PY=${PY:-/home/yewon/.conda/envs/isaaclab/bin/python}
TAG=${TAG:?}; GPU=${GPU:?}; CKPT=${CKPT:?}
if [ "$GPU" != "0" ] && [ "$GPU" != "1" ]; then echo "GPU must be 0 or 1"; exit 1; fi
export NAV_HONEST=1 OBJ_HONEST=1
export MISSION_NAV_HONEST=1 MISSION_OBJ_HONEST=1
export GRASP_CKPT=$CKPT
export GRASP_STEPS=${GRASP_STEPS:-760}
export SEED=${SEED:-0}
# v5 basket (sloped floor — recovery failures zeroed in hs16a); env_cfg's default
# is still the v1 USD, and the BASKET var itself is the on/off switch, not a path
export BASKET_USD=${BASKET_USD:-eunha_rl/assets/a2_piperx_basket_v5.usd}
# HYBRID CLOSER (0907): the frozen-retrained policy approaches without knocking but
# hovers instead of committing; the scripted pinch commits but closed once on air.
# So: one RL try (harmless hover, ~6% lucky convert), then the scripted pinch WITH
# the tactile sweep (reopen-shift-reclose, runner MISSION_IK_PROBES) from attempt 2.
export MISSION_RL_TRIES=${RL_TRIES:-1}
export MISSION_SKIP_LUNCHBOX=${SKIP_LUNCHBOX:-1}
cd "$RW"
# the mission's event stream IS stdout — tee it to the vm-round naming convention
# (deploy/logs/<TAG>_launch.log) or every mid-run diagnostic is lost to /dev/null
GPU=$GPU TAG=$TAG bash deploy/run_mission.sh > "$RW/deploy/logs/${TAG}_launch.log" 2>&1
