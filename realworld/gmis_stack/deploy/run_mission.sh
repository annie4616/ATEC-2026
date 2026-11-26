#!/bin/bash
# Full venue mission with SWAPPABLE policies. All paths are resolved relative to
# this repository — no absolute paths anywhere.
#
#   WALK_CKPT / ADAPT_CKPT / ADAPT_ACT   walking policy pair (+ adaptation activation)
#   CROUCH_CKPT                          crouch policy
#   GRASP_CKPT                           grasp policy (default: champion file)
#   GPU (default 0), SEED, TAG, NO_VIDEO (0 = record overview+follow mp4s)
#
# Examples:
#   bash deploy/run_mission.sh                                # 0824 walk + defaults
#   WALK_CKPT=from_eunha/speed_0.75/model_14999.pt \
#   ADAPT_CKPT=from_eunha/0821/adaptation.pt ADAPT_ACT=relu \
#     bash deploy/run_mission.sh                              # incumbent walk pair
#
# The body is one { } group so bash parses ALL of it before running any of it:
# bash otherwise reads a script lazily by byte offset, and editing this file while
# a launched mission is still running made the wrapper resume at a shifted offset
# when the mission ended — it re-executed the launch line and started a second,
# unplanned mission (0903_2222_svO).
{
set -u
RW=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-python}
cd "$RW"
export EUNHA_ADAPT_CKPT="$RW/${ADAPT_CKPT:-from_eunha/0824/adaptation.pt}"
export EUNHA_ADAPT_ACT=${ADAPT_ACT:-relu}
[ -n "${CROUCH_CKPT:-}" ] && export MISSION_CROUCH_CKPT="$RW/$CROUCH_CKPT"
[ -n "${GRASP_CKPT:-}" ] && export MISSION_GRASP_CKPT="$RW/$GRASP_CKPT"
export GRASP_ROBOT=a2piper GRASP_LEG_EUNHA=1
export GRASP_BANK_DIR="$RW/eunha_rl/s0821/banks_ik"
export GRASP_ARM_RATE=0.2 GRASP_ARM_DELAY=1
# deployment realism (the plant every scoring number was measured on)
# GRIP_RATE 0.02 m/step is 1.0 m/s at 50 Hz — EXACTLY arm_joint7/8's rating, so any
# PD overshoot lands over it (audit: arm_joint8 6.5/1.0 on 0.34 % of steps). 0.016 is
# 0.8 m/s, a 20 % margin, and costs under a step per close.
export MISSION_ARM_RATE=0.2 MISSION_GRIP_RATE=${GRIP_RATE:-0.016} MISSION_ARM_DELAY=1
export MISSION_GRIP_VEL_GUARD=${GRIP_VEL_GUARD:-1.0}
export MISSION_ARM_RATE_SLEW=${ARM_RATE_SLEW:-0}
export MISSION_GRASP_DET_SEED=1 MISSION_DETECT_HZ=2 MISSION_HARVEST=all
# venue configuration (photo world + deploy realism + route economy)
export MISSION_VENUE_PHOTO=${VENUE_PHOTO:-1}
export MISSION_FOOT_GUARD=1 MISSION_APRON_FALLBACK=1 MISSION_PERSIST_BUDGET=1
export MISSION_GRASP_STEPS=${GRASP_STEPS:-500} MISSION_USE_BASKET=${BASKET:-1}
export MISSION_MAX_CARRY=${MAX_CARRY:-3} MISSION_BASKET_RETRIES=${BASKET_RETRIES:-5}
export MISSION_VX=${VX:-0.55} MISSION_MAX_TRIPS=${MAX_TRIPS:-6} MISSION_MAX_PASSES=${MAX_PASSES:-2}
export MISSION_AIM_X=${AIM_X:-0.12} MISSION_CLOSE_RETRY=${CLOSE_RETRY:-0.12}
# 120-point bundle (exp/grasp-boost): every knob defaults to the pre-bundle behaviour
# unless set, so the incumbent runs stay reproducible. See runner.py for each one.
#   SURVEY=1        scan-first traverse: centreline + pan-and-look stations, memory picks
#   ZIGZAG_AMP      reactive sweep amplitude (profile hard-codes 1.1; audit wants 1.2)
#   FAR_RETRY/LAT_RETRY  post-crouch x/y error that triggers the one re-approach
#   OBJ_CAP         steps one object may burn across its cycles before it is written off
#   RL_TRIES        RL attempts per crouch before the scripted IK; IK_CLASSES limits IK
#   STALL_AT/STALL_REACH  abort an RL attempt that never got within REACH m by step AT
#   BAG_AFTER_CORE  bags wait until can/rind/lunchbox pairs are binned
[ -n "${SURVEY:-}" ] && export MISSION_SURVEY=$SURVEY
[ -n "${ZIGZAG_AMP:-}" ] && export MISSION_ZIGZAG_AMP=$ZIGZAG_AMP
[ -n "${FAR_RETRY:-}" ] && export MISSION_FAR_RETRY=$FAR_RETRY
[ -n "${LAT_RETRY:-}" ] && export MISSION_LAT_RETRY=$LAT_RETRY
[ -n "${OBJ_CAP:-}" ] && export MISSION_OBJ_CAP=$OBJ_CAP
[ -n "${RL_TRIES:-}" ] && export MISSION_RL_TRIES=$RL_TRIES
[ -n "${IK_CLASSES:-}" ] && export MISSION_RIGID_IK_CLASSES=$IK_CLASSES
[ -n "${STALL_AT:-}" ] && export MISSION_GRASP_STALL_AT=$STALL_AT
[ -n "${STALL_REACH:-}" ] && export MISSION_GRASP_STALL_REACH=$STALL_REACH
[ -n "${BAG_AFTER_CORE:-}" ] && export MISSION_BAG_AFTER_CORE=$BAG_AFTER_CORE
[ -n "${BAG_ROLL:-}" ] && export MISSION_BAG_ROLL=$BAG_ROLL
[ -n "${BAG_JAW:-}" ] && export MISSION_BAG_JAW=$BAG_JAW
[ -n "${BAG_THREAD_FIRST:-}" ] && export MISSION_BAG_THREAD_FIRST=$BAG_THREAD_FIRST
[ -n "${HAND_TO_BASKET:-}" ] && export MISSION_HAND_TO_BASKET=$HAND_TO_BASKET
[ -n "${TRIP_STALL:-}" ] && export MISSION_TRIP_STALL=$TRIP_STALL
[ -n "${TRIP_STALL_MIN:-}" ] && export MISSION_TRIP_STALL_MIN=$TRIP_STALL_MIN
[ -n "${GRASP_CKPT2:-}" ] && export MISSION_GRASP_CKPT2="$RW/$GRASP_CKPT2"
[ -n "${BASKET_USD:-}" ] && export MISSION_BASKET_USD="$RW/$BASKET_USD"
[ -n "${BASKET_REACH:-}" ] && export MISSION_BASKET_REACH=$BASKET_REACH
#   AIM_Y           post-crouch lateral sweet spot (+ = object left of the anchor)
#   REAPP_GAIN      fraction of the measured post-crouch error folded into the retry aim
#   LOOK_DIST       back off / turn until the target is this far ahead before a look (0 = off)
#   LBOX_DEFER      survey planner keeps the lunchbox for later while basket items lie ahead
#   ALIGN_CLASSES   square the heading up to these classes' long axis ("" = off)
#   DEPOSIT_SCALE   basket-deposit swing speed (1.0 = stock timings)
[ -n "${AIM_Y:-}" ] && export MISSION_AIM_Y=$AIM_Y
[ -n "${REAPP_GAIN:-}" ] && export MISSION_REAPP_GAIN=$REAPP_GAIN
[ -n "${LOOK_DIST:-}" ] && export MISSION_LOOK_DIST=$LOOK_DIST
[ -n "${LBOX_DEFER:-}" ] && export MISSION_LBOX_DEFER=$LBOX_DEFER
[ -n "${ALIGN_CLASSES+x}" ] && export MISSION_ALIGN_CLASSES=$ALIGN_CLASSES
# square the heading PERPENDICULAR to these classes' long axis (the bag's handle
# window faces along its short axis, so the thread needs the 90-deg version)
[ -n "${ALIGN_PERP_CLASSES+x}" ] && export MISSION_ALIGN_PERP_CLASSES=$ALIGN_PERP_CLASSES
[ -n "${ALIGN_TOL:-}" ] && export MISSION_ALIGN_TOL=$ALIGN_TOL
[ -n "${DEPOSIT_SCALE:-}" ] && export MISSION_DEPOSIT_SCALE=$DEPOSIT_SCALE
#   SURVEY_FULL     360-degree survey sweeps (6 stops) instead of the +-0.6 rad forward pan
#   SURVEY_OFFS     station offsets from each object section's centre (m, comma list)
#   SECTION_FIRST   1 = sweep every station of a section before picking anything in it
#                   (LBOX_DEFER=1 then defers the lunchbox within the section only;
#                   LBOX_DEFER=2 keeps deferring it while basket items lie in sections ahead)
[ -n "${SURVEY_FULL:-}" ] && export MISSION_SURVEY_FULL=$SURVEY_FULL
[ -n "${SURVEY_OFFS:-}" ] && export MISSION_SURVEY_OFFS=$SURVEY_OFFS
[ -n "${SECTION_FIRST:-}" ] && export MISSION_SURVEY_SECTION_FIRST=$SECTION_FIRST
[ -n "${PERCH_SKIP:-}" ] && export MISSION_PERCH_SKIP=$PERCH_SKIP
[ -n "${STAIR_HAND_VX:-}" ] && export MISSION_STAIR_HAND_VX=$STAIR_HAND_VX
#   LBOX_TRIES      grasp attempts per crouch for the lunchbox (RL first; see runner)
#   IK_LAST_CYCLE   1 = the scripted IK pinch only in an object's last cycle
#   EDGE_LAT        |lat| treated as the boundary rope for back-off/abandon (1.45)
#   OBJ_ATTEMPTS    full approach->crouch->grasp cycles per object (profile: 2, +1 hand classes)
#   UPCAN_RL_TRIES  N = a can standing on its end gets N RL tries then the scripted
#                   straddle, every cycle (seed 0's grass can; RL 7/44 on it)
[ -n "${LBOX_TRIES:-}" ] && export MISSION_LBOX_TRIES=$LBOX_TRIES
[ -n "${IK_LAST_CYCLE:-}" ] && export MISSION_IK_LAST_CYCLE=$IK_LAST_CYCLE
[ -n "${UPCAN_RL_TRIES:-}" ] && export MISSION_UPCAN_RL_TRIES=$UPCAN_RL_TRIES
[ -n "${EDGE_LAT:-}" ] && export MISSION_EDGE_LAT=$EDGE_LAT
[ -n "${OBJ_ATTEMPTS:-}" ] && export MISSION_OBJ_ATTEMPTS=$OBJ_ATTEMPTS
#   REAPP_MAX       re-approaches after a crouch that missed the FAR/LAT bars (1)
#   LBOX_NEAR_BINS  1 = when only the hand class is left, take the copy in the next
#                   unswept section (nearer the bins) instead of this section's
[ -n "${REAPP_MAX:-}" ] && export MISSION_REAPP_MAX=$REAPP_MAX
[ -n "${LBOX_NEAR_BINS:-}" ] && export MISSION_LBOX_NEAR_BINS=$LBOX_NEAR_BINS
#   RECENTRE_LAT    after a pick cycle at |lat| >= this, turn and walk back to the
#                   centreline before the search resumes (0.9; 0 = off)
#   BASKET_LIP_X    v3 basket: base-frame x of the rear lip's front face (-0.15)
#   HOVER_XMIN      basket unload: hover no further aft than this x (-0.10)
#   (v4 basket: BASKET_USD=eunha_rl/assets/a2_piperx_basket_v4.usd BASKET_LIP_X=-0.16)
[ -n "${RECENTRE_LAT:-}" ] && export MISSION_RECENTRE_LAT=$RECENTRE_LAT
[ -n "${BASKET_LIP_X:-}" ] && export MISSION_BASKET_LIP_X=$BASKET_LIP_X
[ -n "${HOVER_XMIN:-}" ] && export MISSION_HOVER_XMIN=$HOVER_XMIN
#   EDGE_TURN       never spin in place beyond this |lat| — walk straight in first (1.45)
#   CYCLE_RECENTRE_LAT  before a retry cycle's re-look, come back inside from |lat| >= this (1.3)
[ -n "${EDGE_TURN:-}" ] && export MISSION_EDGE_TURN=$EDGE_TURN
[ -n "${CYCLE_RECENTRE_LAT:-}" ] && export MISSION_CYCLE_RECENTRE_LAT=$CYCLE_RECENTRE_LAT
export MISSION_LAT_GAIN=${LAT_GAIN:-1.8} MISSION_CORRIDOR=${CORRIDOR:-1.10}
export MISSION_TURN_SCALE=${TURN_SCALE:-1.8} MISSION_CARRY_VX=${CARRY_VX:-0.32}
export MISSION_SKIP_REDBAG=${SKIP_REDBAG:-0} MISSION_BAG_HANDLE=1
export MISSION_NO_VIDEO=${NO_VIDEO:-0}
export MISSION_NAV_HONEST=${NAV_HONEST:-0}   # 1 = map-free navigation (실전형)
#   TORQUE_AUDIT=1  per-joint PD-demand / applied torque / joint speed vs the actuator
#                   limits, summary every 5000 steps + torque_audit.json in the out dir
[ -n "${TORQUE_AUDIT:-}" ] && export MISSION_TORQUE_AUDIT=$TORQUE_AUDIT
#   STAIR_CARRY_VX  carry speed on the stair section (0.28; CARRY_VX elsewhere)
#   SHAKE_OFF       1 = an object still riding the open fingers after the release over
#                   a bin is shaken off (wrist flicks, re-grip/re-open, jiggle) inside
#                   the mouth instead of withdrawing blind (svP lunchbox)
#   DEPOSIT_NUDGE   1 = a deposit that reads high (z 0.22..0.27) inside the basket
#                   footprint is pressed down with the open jaw and re-read before
#                   it is called IN/MISS (svP: a can propped on end was called MISS)
#   PICK_FREE       1 = basket unload: when the pose-locked servo stalls > 3 cm (front-
#                   outboard corner), retry with the position-only servo (tool tilt free)
#   PICK_HOLD_ROT   1 = basket unload servos pull the tool back to the orientation they
#                   started with (the relative IK otherwise lets it drift: probe p5 saw
#                   the tool go from straight down to horizontal between tries)
#   PICK_YAW        1 = basket unload rolls the wrist when a can/rind long axis lies
#                   within 45 deg of the closing axis (else the fingers land ON it);
#                   0 = the old every-second-try alternation only
#   LIP_ROLL_ANG    against the rear lip the roll is normally skipped (a finger would
#                   land on the lip); below this axis angle (deg, default 65) the unrolled
#                   grip cannot close either, so roll anyway and shift the reach target
#                   forward to BASKET_LIP_X + LIP_ROLL_X (m, default 0.055)
[ -n "${LIP_ROLL_ANG:-}" ] && export MISSION_LIP_ROLL_ANG=$LIP_ROLL_ANG
[ -n "${LIP_ROLL_X:-}" ] && export MISSION_LIP_ROLL_X=$LIP_ROLL_X
[ -n "${STAIR_CARRY_VX:-}" ] && export MISSION_STAIR_CARRY_VX=$STAIR_CARRY_VX
[ -n "${STAIR_VX:-}" ] && export MISSION_STAIR_VX=$STAIR_VX
[ -n "${SHAKE_OFF:-}" ] && export MISSION_SHAKE_OFF=$SHAKE_OFF
[ -n "${DEPOSIT_NUDGE:-}" ] && export MISSION_DEPOSIT_NUDGE=$DEPOSIT_NUDGE
[ -n "${PICK_FREE:-}" ] && export MISSION_PICK_FREE=$PICK_FREE
[ -n "${PICK_HOLD_ROT:-}" ] && export MISSION_PICK_HOLD_ROT=$PICK_HOLD_ROT
[ -n "${PICK_YAW:-}" ] && export MISSION_PICK_YAW=$PICK_YAW
#   BANK_PICK       far (default) | over | policy — grasp start config. policy = every
#                   bank config is scored through the grasp actor on the observation
#                   it would open with and the most object-directed start wins; needs
#                   BANK_EE, the per-config gripper quaternion dump from this sim
#                   (exp_runs/banks/pregrasp_bank_0.24_ee.npz, scratch bank_quat_dump.py)
#   PICK_DMAX       policy pick: max start |grasp vector| (m, default 0.16 — the RSI
#                   training start distribution)
#   RAMP_SETTLE     up to N extra steps after the joint ramp until the arm is within
#                   0.05 rad of the bank config (default 0)
#   DIVERGE_AT/DIVERGE  abort the attempt at step DIVERGE_AT (40) when the policy's own
#                   estimated TCP-target distance is still > DIVERGE (0.20 m); 0 = off
#   PICK_CLEAR      policy pick: the lower fingertip pad must clear the object's top by
#                   this much when over its footprint (m, default 0.03; +0.03 per
#                   blocked ramp in the same squat)
#   PICK_LAT        policy pick: prefer starts whose TCP is within this of the grasp
#                   point laterally (m, default 0.04 — the RSI +-3 cm drop)
#   RAMP_FAIL       skip the attempt when the arm ends the ramp more than this off its
#                   config (rad, default 0.12: the hand landed on the object); 0 = off
#   HOVER_AT/HOVER  abort at step HOVER_AT (150) when the policy's estimated distance
#                   was never under HOVER (0.10 m) and it never closed near; 0 = off
#   PICK_LATMIN     policy pick: lower end of the preferred lateral window (m, 0)
#   PICK_LAT_LUNCHBOX / PICK_LATMIN_LUNCHBOX / PICK_CLEAR_LUNCHBOX
#                   the same three for the lunchbox only (the holds all came from SIDE
#                   starts 0.14-0.16 m out at lid height, e.g. LATMIN 0.10 LAT 0.20)
#   PICK_GROUND     policy pick: the lower fingertip pad must stay this far above the
#                   ground the robot stands on (its own stance height; m, default 0.02)
#   RAMP_MASK       policy pick: npz with a bool "ok" per kept bank config — whether the
#                   joint ramp reaches it in the squat with no object (scratch
#                   ramp_sweep.py); unreachable configs are never picked
[ -n "${BANK_PICK:-}" ] && export MISSION_BANK_PICK=$BANK_PICK
[ -n "${BANK_EE:-}" ] && export MISSION_BANK_EE=$BANK_EE
[ -n "${PICK_DMAX:-}" ] && export MISSION_PICK_DMAX=$PICK_DMAX
[ -n "${RAMP_SETTLE:-}" ] && export MISSION_RAMP_SETTLE=$RAMP_SETTLE
[ -n "${DIVERGE_AT:-}" ] && export MISSION_GRASP_DIVERGE_AT=$DIVERGE_AT
[ -n "${DIVERGE:-}" ] && export MISSION_GRASP_DIVERGE=$DIVERGE
[ -n "${PICK_CLEAR:-}" ] && export MISSION_PICK_CLEAR=$PICK_CLEAR
[ -n "${PICK_LAT:-}" ] && export MISSION_PICK_LAT=$PICK_LAT
[ -n "${RAMP_FAIL:-}" ] && export MISSION_RAMP_FAIL=$RAMP_FAIL
[ -n "${HOVER_AT:-}" ] && export MISSION_GRASP_HOVER_AT=$HOVER_AT
[ -n "${HOVER:-}" ] && export MISSION_GRASP_HOVER=$HOVER
[ -n "${PICK_LATMIN:-}" ] && export MISSION_PICK_LATMIN=$PICK_LATMIN
[ -n "${PICK_LAT_LUNCHBOX:-}" ] && export MISSION_PICK_LAT_LUNCHBOX=$PICK_LAT_LUNCHBOX
[ -n "${PICK_LATMIN_LUNCHBOX:-}" ] && export MISSION_PICK_LATMIN_LUNCHBOX=$PICK_LATMIN_LUNCHBOX
[ -n "${PICK_CLEAR_LUNCHBOX:-}" ] && export MISSION_PICK_CLEAR_LUNCHBOX=$PICK_CLEAR_LUNCHBOX
[ -n "${PICK_GROUND:-}" ] && export MISSION_PICK_GROUND=$PICK_GROUND
[ -n "${RAMP_MASK:-}" ] && export MISSION_BANK_RAMP_MASK=$RAMP_MASK
# OBJECT-HONEST mode: no simulator pose of any OBJECT on a control path — the grasp
# target comes from the detection, HELD/carried/released from the gripper (width +
# effort + the TCP's own rise), the basket from the slot the hand released over.
# Truth stays in the report and in the [TRUTH] log lines, which is what the venue
# judges score anyway. OFF by default; NAV_HONEST is the separate map-prior switch.
[ -n "${OBJ_HONEST:-}" ] && export MISSION_OBJ_HONEST=$OBJ_HONEST
[ -n "${HOLD_GW_LO:-}" ] && export MISSION_HOLD_GW_LO=$HOLD_GW_LO
[ -n "${HOLD_GW_HI:-}" ] && export MISSION_HOLD_GW_HI=$HOLD_GW_HI
[ -n "${HOLD_EFFORT:-}" ] && export MISSION_HOLD_EFFORT=$HOLD_EFFORT
[ -n "${HOLD_LIFT:-}" ] && export MISSION_HOLD_LIFT=$HOLD_LIFT
[ -n "${DET_CENTRE_LUNCHBOX:-}" ] && export MISSION_DET_CENTRE_LUNCHBOX=$DET_CENTRE_LUNCHBOX
[ -n "${DET_CENTRE_CAN:-}" ] && export MISSION_DET_CENTRE_CAN=$DET_CENTRE_CAN
[ -n "${DET_CENTRE_RIND:-}" ] && export MISSION_DET_CENTRE_RIND=$DET_CENTRE_RIND
[ -n "${DET_TOP_LUNCHBOX:-}" ] && export MISSION_DET_TOP_LUNCHBOX=$DET_TOP_LUNCHBOX
[ -n "${DET_TOP_CAN:-}" ] && export MISSION_DET_TOP_CAN=$DET_TOP_CAN
[ -n "${DET_TOP_RIND:-}" ] && export MISSION_DET_TOP_RIND=$DET_TOP_RIND
[ -n "${BASKET_FLOOR_B:-}" ] && export MISSION_BASKET_FLOOR_B=$BASKET_FLOOR_B
# drop aim: how far the bin memory may move the hand off the approach's own aim
[ -n "${RECENTRE_MAX:-}" ] && export MISSION_RECENTRE_MAX=$RECENTRE_MAX
[ -n "${RECENTRE_MAX_AGE:-}" ] && export MISSION_RECENTRE_MAX_AGE=$RECENTRE_MAX_AGE
[ -n "${RECENTRE_STALE_CAP:-}" ] && export MISSION_RECENTRE_STALE_CAP=$RECENTRE_STALE_CAP
[ -n "${BIN_STOP_LAT:-}" ] && export MISSION_BIN_STOP_LAT=$BIN_STOP_LAT
# back out and look at the bin row again when its sighting has gone stale (the second
# drop of a stop is always the one that misses)
[ -n "${BIN_RESIGHT:-}" ] && export MISSION_BIN_RESIGHT=$BIN_RESIGHT
[ -n "${BIN_RESIGHT_AGE:-}" ] && export MISSION_BIN_RESIGHT_AGE=$BIN_RESIGHT_AGE
# deck-edge guard: forward speed cap once |lat| is past EDGE_SAFE (a fall ends the run)
[ -n "${EDGE_SAFE:-}" ] && export MISSION_EDGE_SAFE=$EDGE_SAFE
# how much clock a "sweep again instead of finishing early" needs beyond the reserves
[ -n "${PERSIST_MARGIN:-}" ] && export MISSION_PERSIST_MARGIN=$PERSIST_MARGIN
[ -n "${RELEASE_LOAD_CHECK:-}" ] && export MISSION_RELEASE_LOAD_CHECK=$RELEASE_LOAD_CHECK
[ -n "${DISP_LOADED_N:-}" ] && export MISSION_DISP_LOADED_N=$DISP_LOADED_N
[ -n "${EDGE_HOLD:-}" ] && export MISSION_EDGE_HOLD=$EDGE_HOLD
[ -n "${BIN_OBS_AGE:-}" ] && export NAVEST_BIN_OBS_AGE=$BIN_OBS_AGE
[ -n "${BIN_OBS_MOVE:-}" ] && export NAVEST_BIN_OBS_MOVE=$BIN_OBS_MOVE
TAG=${TAG:-mission}
OUT="$RW/deploy/logs/$(date +%m%d_%H%M)_$TAG"
mkdir -p "$OUT"
# SNAPSHOT THE SOURCE THAT RUNS. course_mission/ is not tracked by git and the tree
# moves constantly during a round: hs16a scored 120/160 and by the time the score line
# printed, six of the files that produced it had already been edited under it. A few
# hundred kB per run buys the ability to go back to whatever scored.
mkdir -p "$OUT/src"
cp -a "$RW/course_mission" "$OUT/src/" 2>/dev/null
cp -a "$RW/deploy/run_mission.sh" "$OUT/src/" 2>/dev/null
CUDA_VISIBLE_DEVICES=${GPU:-0} $PY course_mission/solution_v5_venue.py --headless \
  --device cuda:0 --walk_ckpt "$RW/${WALK_CKPT:-from_eunha/0824/model_22998.pt}" \
  --seed ${SEED:-0} --out_dir "$OUT" ${EXTRA:-}
grep -ao "score (과제2) = [0-9]* / [0-9]*" "$OUT/mission.log" | tail -1
exit
}
