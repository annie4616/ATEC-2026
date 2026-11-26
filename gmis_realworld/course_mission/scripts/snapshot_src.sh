#!/bin/bash
# Copy the exact source that a run is about to execute into that run's output folder.
#
# WHY THIS EXISTS: course_mission has zero files tracked in git, and the two runs that
# scored 100/160 (videos_persist/0817_0153 and 0817_0405) were followed by ~15 edits to
# runner.py. Their reports and videos survive; the code that produced them does not. A
# score you cannot reproduce is not a result. Every run now carries its own source, so
# `videos_*/<run>/src/` is always the answer to "what exactly scored this?".
#
# Usage: snapshot_src.sh <out_dir>
set -u
OUT=${1:?usage: snapshot_src.sh <out_dir>}
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SRC=$OUT/src
mkdir -p "$SRC"

# the mission itself
cp -p "$RW"/course_mission/*.py "$SRC"/ 2>/dev/null
mkdir -p "$SRC/scripts"
cp -p "$RW"/course_mission/scripts/*.sh "$SRC/scripts"/ 2>/dev/null
# the arena it runs on
mkdir -p "$SRC/course_sim"
cp -p "$RW"/course_sim/*.py "$SRC/course_sim"/ 2>/dev/null

# Checkpoints are far too big to copy, so record WHICH ones plus a hash, and the env that
# selects them. A policy silently swapped under the same path is the other way a score
# becomes unreproducible.
{
  echo "run_dir : $OUT"
  echo "host    : $(hostname)"
  echo "git(rw) : $(cd "$RW" && git rev-parse --short HEAD 2>/dev/null || echo none)"
  echo
  echo "== environment =="
  for v in MISSION_HARVEST MISSION_ORACLE_DETECT MISSION_NO_BASKET MISSION_NO_VIDEO \
           MISSION_REAL_NOISE MISSION_COMPUTE MISSION_ACTION_DELAY MISSION_DETECT_HZ \
           CUDA_VISIBLE_DEVICES; do
    echo "$v=${!v:-}"
  done
  echo
  echo "== checkpoints (sha256, first 16) =="
  find "$RW/eunha_rl" "$RW/grasp_rl" -name "*.pt" -newermt "-30 days" 2>/dev/null \
    | head -40 | while read -r f; do
        printf "%s  %s\n" "$(sha256sum "$f" | cut -c1-16)" "${f#$RW/}"
      done
  echo
  echo "== mission source sha256 =="
  (cd "$SRC" && find . -name "*.py" -o -name "*.sh" | sort | xargs sha256sum 2>/dev/null \
     | sed 's/\(^.\{16\}\).*  /\1  /')
} > "$SRC/PROVENANCE.txt" 2>/dev/null

echo "  source snapshot -> $SRC"
