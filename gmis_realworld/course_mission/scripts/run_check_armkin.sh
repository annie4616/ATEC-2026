#!/bin/bash
# Compare armkin.py's URDF kinematics against the simulator's PhysX ones.
set -u
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=/home/yewon/.conda/envs/isaaclab/bin/python
LOG=$RW/course_mission/logs/check_armkin.log
mkdir -p "$RW/course_mission/logs"
if [ -s "$LOG" ]; then echo "REFUSING: $LOG already holds a run"; exit 1; fi
source $RW/eunha_rl/scripts/eunha_env.sh ${EUNHA_TAG:-s0821}
cd $RW
CUDA_VISIBLE_DEVICES=0 timeout --signal=KILL ${TMO:-3600} $PY \
  course_mission/scripts/check_armkin.py --headless --device cuda:0 \
  --poses ${POSES:-25} > "$LOG" 2>&1
grep -a "\[ARMKIN\]" "$LOG" | tail -35
grep -a -m1 -A14 "Traceback" "$LOG" | head -16
echo "ARMKIN_DONE"
