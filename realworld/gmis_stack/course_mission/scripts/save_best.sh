#!/bin/bash
# Promote a finished run's snapshotted source into course_mission/snapshots/ under a name
# that records its score, so the best-scoring code is kept even as the tree moves on.
#
#   save_best.sh videos_venue/0817_1830_c2 [label]
set -u
RUN=${1:?usage: save_best.sh <run_dir> [label]}
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$RW/course_mission"
[ -d "$RUN/src" ] || { echo "no $RUN/src — run predates snapshotting, source is NOT recoverable"; exit 1; }
REP=$(ls "$RUN"/report_*.json 2>/dev/null | head -1)
SCORE=$(~/.conda/envs/isaaclab/bin/python -c "
import json,sys
d=json.load(open(sys.argv[1])); print(f\"{d['score_task2']}pts_{d['steps']}steps\")" "$REP" 2>/dev/null)
NAME="${SCORE:-unscored}_$(basename "$RUN")${2:+_$2}"
mkdir -p "snapshots/$NAME"
cp -rp "$RUN/src"/* "snapshots/$NAME"/
[ -n "$REP" ] && cp -p "$REP" "snapshots/$NAME"/
echo "saved -> snapshots/$NAME"
echo "remember to add a row to snapshots/README.md"
