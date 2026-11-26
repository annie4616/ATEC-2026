#!/bin/bash
# Kill a detached run PROPERLY.
#
# launch_detached.sh starts the job under setsid, so the pid file holds the wrapper —
# the session leader. `kill $(cat x.pid)` reaps only that shell and leaves the Isaac Sim
# child alive holding ~7-10 GB of GPU memory. Twelve of those accumulated over one
# afternoon of restarts and filled a 98 GB card, after which every new run died with
# "Failed to create stream on device cuda:0". Kill the whole process group instead.
set -u
CM=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
for name in "$@"; do
  f=$CM/logs/$name.pid
  [ -f "$f" ] || { echo "  $name: no pid file"; continue; }
  pid=$(cat "$f")
  pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
  if [ -n "$pgid" ]; then
    kill -TERM -- "-$pgid" 2>/dev/null
    sleep 3
    kill -KILL -- "-$pgid" 2>/dev/null
    echo "  $name: killed process group $pgid"
  else
    echo "  $name: not running"
  fi
done
# Isaac spawns grandchildren (renderer, imageio/ffmpeg writers) that escape the group.
# Sweep only TRUE orphans — processes reparented to init (PPID 1). An earlier version
# swept every course_mission process older than 120 s and killed a healthy run that was
# mid-mission in another slot.
# Children are reparented to init only AFTER their parent dies, so one pass misses the
# grandchildren: a straggler survived 4300 s and kept writing to the log of a run that had
# supposedly been stopped, corrupting the next run's numbers. Sweep repeatedly.
for _round in 1 2 3; do
  found=0
  for pid in $(ps -eo pid,ppid,cmd \
               | grep -E "[p]ython.*course_mission|[f]fmpeg.*course_mission" \
               | awk '$2==1 {print $1}'); do
    kill -9 "$pid" 2>/dev/null && echo "  swept orphan $pid (ppid 1)" && found=1
  done
  [ "$found" = "0" ] && break
  sleep 2
done
sleep 3
echo "GPU0 now: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 0)"
