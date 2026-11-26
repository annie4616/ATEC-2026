#!/bin/bash
set -u
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=/home/yewon/.conda/envs/isaaclab/bin/python
export CUDA_MPS_PIPE_DIRECTORY=/tmp/yewon-no-mps
cd $RW
echo "COURSE_SMOKE start $(date '+%F %T') on GPU5"
for task in ATEC-Course-A2Piper; do
  CUDA_VISIBLE_DEVICES=5 timeout --signal=KILL 1500 $PY course_sim/scripts/view_course.py --headless --device cuda:0 \
    --task $task > course_sim/videos/smoke_$task.log 2>&1
  grep -E "\[COURSE\]|Traceback|Error" course_sim/videos/smoke_$task.log | tail -4
  mv course_sim/videos/course-step-0.mp4 course_sim/videos/${task}.mp4 2>/dev/null && echo "COURSE_VIDEO ${task}.mp4"
done
echo "COURSE_SMOKE_DONE"
