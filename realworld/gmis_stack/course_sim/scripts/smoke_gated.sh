#!/bin/bash
# Course env smoke test — waits for a free slot on GPU 0 (our jobs < 4), then records
# a video for both robot variants. Never exceeds the 4-jobs/GPU budget.
set -u
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=/home/yewon/.conda/envs/isaaclab/bin/python
export CUDA_MPS_PIPE_DIRECTORY=/tmp/yewon-no-mps
cd $RW
while true; do
  uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1==0 {print $2}')
  n=$(nvidia-smi --query-compute-apps=gpu_uuid,process_name --format=csv,noheader | grep "$uuid" | grep -c yewon)
  [ "$n" -lt 4 ] && break
  sleep 120
done
echo "COURSE_SMOKE start $(date '+%F %T')"
for task in ATEC-Course-A2Piper ATEC-Course-B2Piper; do
  CUDA_VISIBLE_DEVICES=0 timeout 1500 $PY course_sim/scripts/view_course.py --headless --device cuda:0 \
    --task $task > course_sim/videos/smoke_$task.log 2>&1
  grep -E "\[COURSE\]|Traceback|Error" course_sim/videos/smoke_$task.log | tail -4
  mv course_sim/videos/course-step-0.mp4 course_sim/videos/${task}.mp4 2>/dev/null && echo "COURSE_VIDEO ${task}.mp4"
done
echo "COURSE_SMOKE_DONE"
