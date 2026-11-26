#!/bin/bash
# Code checkpoint: every behaviour-affecting edit gets a restorable copy BEFORE it can be
# lost. Lesson learned the hard way (memory: save-scoring-code-snapshots): course_mission
# is untracked in git, and the 100-point mission code was once actually lost.
#
#   bash course_mission/scripts/checkpoint_code.sh "<tag>"
#
# Restore = copy the checkpoint back over course_mission/*.py and course_sim/.
set -u
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${1:-manual}
DST=$RW/course_mission/snapshots/ckpts/$(date +%m%d_%H%M%S)_$TAG
mkdir -p "$DST"
cp $RW/course_mission/*.py "$DST/"
mkdir -p "$DST/scripts" "$DST/course_sim"
cp $RW/course_mission/scripts/*.py $RW/course_mission/scripts/*.sh "$DST/scripts/" 2>/dev/null
cp $RW/course_sim/*.py "$DST/course_sim/"
mkdir -p "$DST/graspenv/mdp"
cp $RW/grasp_rl/graspenv/*.py "$DST/graspenv/" 2>/dev/null
cp $RW/grasp_rl/graspenv/mdp/*.py "$DST/graspenv/mdp/" 2>/dev/null
{
  echo "tag: $TAG"
  echo "when: $(date -u +%FT%TZ)"
  echo "md5:"
  md5sum $RW/course_mission/*.py $RW/course_sim/*.py | sed 's|.*/||' | sort
} > "$DST/CHECKPOINT.txt"
echo "checkpoint -> ${DST#$RW/}"
