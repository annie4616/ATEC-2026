#!/bin/bash
# Launch a command fully DETACHED from the calling session (survives VSCode/session exit).
# Usage: launch_detached.sh <name> <command...>
# Log:   grasp_rl/detached/<name>.log   PID: grasp_rl/detached/<name>.pid
# Done:  grasp_rl/detached/<name>.done  (written on exit with the exit code)
set -u
RW=/hdd0/yewon/ATEC2026_Simulation_Challenge/real_world_preliminary/ATEC2026_Simulation_Challenge
NAME=${1:?name required}; shift
DIR=${GRASP_DETACH_DIR:-$RW/grasp_rl/detached}
mkdir -p "$DIR"
setsid nohup bash -c "\"\$@\"; echo \$? > '$DIR/$NAME.done'" bash "$@" > "$DIR/$NAME.log" 2>&1 < /dev/null &
PID=$!
echo $PID > "$DIR/$NAME.pid"
disown $PID 2>/dev/null
echo "detached: $NAME pid=$PID log=$DIR/$NAME.log"
