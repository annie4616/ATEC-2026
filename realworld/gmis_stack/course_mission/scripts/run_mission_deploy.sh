#!/bin/bash
# DEPLOY PROFILE: the mission under every real-robot constraint we can enforce in sim,
# !! 이 파일을 실행 중에 절대 in-place로 고치지 말 것: bash는 바이트 오프셋으로 읽는다.
# !! 수정은 반드시 temp에 쓰고 mv로 교체(inode 교체 → 실행 중 셸은 옛 파일을 계속 읽음).
# !! 2026-08-24 19:09 실측: in-place 수정된 스크립트가 미션 종료 후 어긋난 오프셋에서
# !! 재실행되어 같은 OUT에 유령 미션을 띄우고 mission.log를 덮어썼다.
# configured for maximum score under those constraints. 30 minutes, GPU 0 only.
#
#   bash course_mission/scripts/run_mission_deploy.sh                # deploy profile
#   STRICT=1 bash course_mission/scripts/run_mission_deploy.sh      # + 0 Hz grasp target
#
# WHY EACH SWITCH (all measured, see DEPLOY.md "실험 결과"):
#   MISSION_DETECT_HZ=2        honest N150 detector rate (measured 0.5-1.0 s/frame there;
#                              the default 5 Hz is the stale deploy.py estimate)
#   MISSION_ARM_RATE=0.2       10 rad/s arm limit in software; on hardware the driver
#                              faults without it. Cost: none (4/12 vs base 7/17, p=0.70)
#   MISSION_GRIP_RATE=0.02     same for the gripper
#   MISSION_ARM_DELAY=1        arm/gripper share the 20 ms command latency the legs
#                              already model. Cost: none (p=1.00)
#   MISSION_GRASP_DET_SEED=1   grasp seeded from the DETECTION, cam_err=0 — removes a
#                              BLOCKER-class truth read. Cost: none (3/6, p=1.00)
#   MISSION_SKIP_REDBAG=1      bags are negative EV: 0 held in 60+ attempts AND the bag
#                              orbit walked a run 54 m off the course (0/160). The ~40
#                              points they theoretically carry were never once realised.
#   MISSION_FOOT_GUARD=1       lateral guard derived from the FEET (venue judges feet):
#                              fires at 1.645 m, before the outer foot reaches the
#                              stanchion base at 1.8675 m. The body-centre guard fired
#                              only once the robot was already standing on the base.
#   MISSION_PERSIST_BUDGET=1   if the search stops only because of pass/attempt caps and
#                              14k+ steps remain past the walk home, wipe the failure
#                              records and sweep again — crossing the finish line ends
#                              collection (guide 125), so banked budget is worth zero
#   MISSION_APRON_FALLBACK=1   bin unreachable -> put the object down where we stand:
#                              25 % (5 pts) instead of the guaranteed 0 of carrying it home
#   MISSION_GRASP_STEPS=500    the rate-limited + delayed arm tracks the same trajectory
#                              slower; 340 steps was tuned with neither. Measured under
#                              the full constraint set: 340 steps 1/6, 500 steps 3/6.
#                              Cost: ~3 s per attempt out of a 30-min budget
#   STRICT adds MISSION_GRASP_FREEZE=1: the grasp target never refreshes during the
#   attempt, which is what the head camera actually delivers (NEAR_CUT + arm occlusion).
#   Grasp rate collapses to ~7 % (2/30, p=0.0068) — this is the honest floor without a
#   wrist camera, and the number that argues for buying one.
set -u
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=/home/yewon/.conda/envs/isaaclab/bin/python
STRICT=${STRICT:-0}
TAG=${TAG:-deploy$([ "$STRICT" = "1" ] && echo _strict)}
OUT=${OUT:-$RW/course_mission/videos_venue/$(date +%m%d_%H%M)_$TAG}
if [ -n "$(ls "$OUT"/report_*.json 2>/dev/null)" ]; then
  echo "REFUSING to overwrite $OUT — it already holds a finished run"
  OUT="${OUT}_$(date +%H%M%S)"
fi
mkdir -p "$OUT"
LOG=$OUT/mission.log
ln -sfn "$LOG" "$RW/course_mission/logs/mission_${TAG}_latest.log"
source $RW/eunha_rl/scripts/eunha_env.sh ${EUNHA_TAG:-s0821}
export MISSION_NO_VIDEO=${NO_VIDEO:-1}
# THE VENUE IS THE PHOTO WORLD. Every scoring run before 0826 07:20 silently ran the
# plain procedural textures because this line was missing — the user's photographed
# backgrounds/props (and the photo-fitted perception prototypes) were never exercised.
# Default ON; VENUE_PHOTO=0 opts out for texture-independent mechanism probes only.
export MISSION_VENUE_PHOTO=${VENUE_PHOTO:-1}
export MISSION_HARVEST=${HARVEST:-all}
export MISSION_DETECT_HZ=${DETECT_HZ:-2}
export MISSION_ARM_RATE=0.2
export MISSION_GRIP_RATE=0.02
export MISSION_ARM_DELAY=1
export MISSION_GRASP_DET_SEED=1
# Overridable since 0827: the HANDLE bag (MISSION_BAG_HANDLE=1) converted its first
# HELD — pass REDBAG=1 to attempt bags (sets SKIP_REDBAG=0 and BAG_HANDLE=1 together).
if [ "${REDBAG:-0}" = "1" ]; then
  export MISSION_SKIP_REDBAG=0 MISSION_BAG_HANDLE=1
else
  export MISSION_SKIP_REDBAG=1
fi
export MISSION_FOOT_GUARD=1
export MISSION_APRON_FALLBACK=1
export MISSION_PERSIST_BUDGET=1
export MISSION_GRASP_STEPS=500
[ "${BASKET:-0}" = "1" ] && export MISSION_USE_BASKET=1
[ "$STRICT" = "1" ] && export MISSION_GRASP_FREEZE=1
# code snapshot BEFORE the run, and launch.py writes config.json (all 47 vars + ckpt
# hashes) into OUT at startup — between them every run is reconstructible.
bash "$RW/course_mission/scripts/snapshot_src.sh" "$OUT"
bash "$RW/course_mission/scripts/checkpoint_code.sh" "run_$TAG" > /dev/null
cd $RW
CUDA_VISIBLE_DEVICES=${GPU:-0} timeout --signal=KILL ${TMO:-43200} $PY \
  course_mission/solution_v5_venue.py --headless --device cuda:0 \
  --out_dir "$OUT" ${EXTRA:-} > $LOG 2>&1
SCORE=$(grep -ao "score (과제2) = [0-9]* / [0-9]*" "$LOG" | tail -1 | grep -o "[0-9]* / [0-9]*")
python3 - "$OUT" <<'PYEOF'
import json, sys, glob
for f in glob.glob(sys.argv[1] + "/report_*.json"):
    r = json.load(open(f))
    print("  task1 %s/%s  total %s  oob(body/feet) %s/%s  finished %s (%s min)  steps %s" % (
        r.get("score_task1"), r.get("task1_max"), r.get("score_total"),
        r.get("oob_count"), r.get("oob_feet_count"), r.get("finished"),
        r.get("finish_min"), r.get("steps")))
PYEOF
echo "$(basename "$OUT")  과제2 ${SCORE:-unfinished}" > "$OUT/RESULT.txt"
cat "$OUT/RESULT.txt"
grep -aE "orbit abandoned|off the side|OUT OF BOUNDS|apron fallback|cannot hold" $LOG | tail -8
grep -a -m1 -A14 "Traceback" $LOG | head -16
echo "MISSION_DONE_$TAG"
