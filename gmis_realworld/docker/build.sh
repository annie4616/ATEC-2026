#!/usr/bin/env bash
# Build the on-board runtime image.
#
#   bash docker/build.sh                 # build, tagged with the short git rev
#   WITH_SDK=0 bash docker/build.sh      # skip the robot SDKs (offline / CI)
#   TAG=escrow-0912 bash docker/build.sh # a name you can cite in the escrow submission
#
# The build context is the repository root and the ROOT .dockerignore whitelists the 13 MB
# the runtime actually imports (measured: 90 files).  That file has to sit at the context
# root -- it lived in docker/ until 2026-09-09, where Docker never reads it, and the context
# was the whole 485 MB repo with untracked __pycache__ baked into the escrow image.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

REV="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
# --porcelain, not `git diff --quiet`: the latter sees neither STAGED nor UNTRACKED files,
# so a new file under onboard/ shipped inside a clean-looking <rev> tag.  The whole point of
# the tag is to say which source produced the image.
DIRTY=""
[ -n "$(git status --porcelain 2>/dev/null)" ] && DIRTY="-dirty"
TAG="${TAG:-${REV}${DIRTY}}"
IMAGE="${IMAGE:-atec-onboard}"
WITH_SDK="${WITH_SDK:-1}"
DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "== building ${IMAGE}:${TAG} =="
echo "   git rev  : ${REV}${DIRTY}"
echo "   with SDK : ${WITH_SDK}"
if [ -n "$DIRTY" ]; then
  echo "   WARNING: the working tree is dirty; this tag is not reproducible"
  git status --porcelain | sed 's/^/            /'
fi

docker build \
  -f docker/Dockerfile \
  --platform linux/amd64 \
  --build-arg "WITH_SDK=${WITH_SDK}" \
  --build-arg "GIT_REV=${REV}${DIRTY}" \
  --build-arg "BUILD_DATE=${DATE}" \
  -t "${IMAGE}:${TAG}" \
  -t "${IMAGE}:latest" \
  "$HERE"

echo
docker images "${IMAGE}" --format '   {{.Repository}}:{{.Tag}}  {{.Size}}  {{.ID}}'
echo
echo "   digest (cite this in the escrow, not the tag):"
docker image inspect "${IMAGE}:${TAG}" --format '   {{.Id}}'
echo
echo "   next: bash docker/run.sh doctor"
