#!/bin/bash
# EUNHA-lineage env preset. Usage: source eunha_env.sh [s0821|s075|s100]
# Default s0821. EUNHA_TAG overrides the argument; EUNHA_ART overrides which
# grasp-artifact set (crouch pose banks) is paired with it.
# Points every GRASP_* path at eunha_rl/<speed>/ and switches stage-2 legs to the
# 53-obs RMA crouch controller (GRASP_LEG_EUNHA=1). Nothing in older lineages moves.
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
E=$RW/eunha_rl
# EUNHA_TAG overrides the positional argument. Twenty-six run_*.sh scripts pass "s075"
# literally, so this is how a new lineage is switched on without editing all of them.
# s0821 is the CURRENT lineage and the default. s075 produced every 120/160 result
# and stays reachable for A/B; s100 is the faster, older pair.
SP=${EUNHA_TAG:-${1:-s0821}}
# SRC = which checkpoint drop to walk with.  ART = which set of GRASP-side artifacts
# (crouch pose banks, run dirs) to use. They are separate because a new walk/crouch
# checkpoint does NOT invalidate the object-side tuning, but it DOES change the body poses
# the crouch controller can reach -- so s0821 starts on s075's banks and they have to be
# re-measured before that pairing is trusted. Override with EUNHA_ART.
case $SP in
  s100)  SRC=speed_1.0;  ART=s100;;
  s075)  SRC=speed_0.75; ART=s075;;
  # ART defaults to s0821, NOT s075. It pointed at s075 so the new lineage could start
  # on the champion banks, but GRASP_BANK_DIR is a WRITE path too: any bank rebuild
  # would have overwritten the artifacts behind every 120/160 result in place, with
  # no error and no backup. eunha_rl/s0821 is seeded with a COPY of them instead, and
  # eunha_rl/s075 is chmod a-w so the mistake cannot be made by any route.
  s0821) SRC=0821;       ART=${EUNHA_ART:-s0821};;
  *) echo "unknown speed tag $SP"; return 1 2>/dev/null || exit 1;;
esac
export GRASP_ROBOT=a2piper
export GRASP_LEG_EUNHA=1
unset GRASP_LEG_CLOCK
export EUNHA_ADAPT_CKPT=$RW/from_eunha/$SRC/adaptation.pt
if [ -f "$E/$ART/adapt_act.txt" ]; then
  export EUNHA_ADAPT_ACT=$(cat "$E/$ART/adapt_act.txt")
else
  export EUNHA_ADAPT_ACT=elu
fi
export GRASP_KEEPOUT_LO=0.218,-0.101,-0.037
export GRASP_KEEPOUT_HI=0.438,0.169,0.163
export CROUCH_LIMITS_PRESET=a2
export GRASP_CROUCH_RUNS=$E/$ART/crouch_runs
export GRASP_POSES_JSON=$E/$ART/banks/crouch_poses.json
export GRASP_BANK_DIR=$E/$ART/banks
export GRASP_RUNS_DIR=$E/$ART/runs
export GRASP_VIDEO_DIR=$E/$ART/videos
export GRASP_BANK_KOFILTER=1
export GRASP_CAMHIT_GRACE=10
export GRASP_GRIP_OPEN=0.05
export GRASP_TCP_OFFSET=0.135
export CUDA_MPS_PIPE_DIRECTORY=/tmp/yewon-no-mps
export EUNHA_WALK_CKPT=$RW/from_eunha/$SRC/model_14999.pt
mkdir -p "$E/$ART/crouch_runs" "$E/$ART/crouch_runs_raw" "$E/$ART/banks" "$E/$ART/runs" "$E/$ART/videos"
echo "[EUNHA] tag=$SP  walk=from_eunha/$SRC  artifacts=eunha_rl/$ART" >&2
