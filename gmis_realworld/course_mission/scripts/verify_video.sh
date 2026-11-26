#!/bin/bash
# Verify every mp4 in a run folder actually decodes, and repair it in place if not.
#
# WHY: 3 of the 30 recorded mp4s were unplayable — and they were the three LONGEST
# recordings, including BOTH 100/160 runs. The container header was fine (correct
# resolution and frame count) so nothing looked wrong until someone tried to play it;
# the H.264 stream failed on the first frame. A forced re-encode with error tolerance
# recovered 42,246 of 43,081 frames (98%), so the pixel data was almost entirely intact —
# it was the container index that was broken. Catch it while the run is fresh.
#
# Usage: verify_video.sh <run_dir>
set -u
DIR=${1:?usage: verify_video.sh <run_dir>}
PY=/home/yewon/.conda/envs/isaaclab/bin/python
FF=$($PY -c "import imageio_ffmpeg,sys; sys.stdout.write(imageio_ffmpeg.get_ffmpeg_exe())" 2>/dev/null)
shopt -s nullglob
for f in "$DIR"/*.mp4; do
  ok=$($PY - "$f" <<'PYEOF' 2>/dev/null
import cv2, sys
c = cv2.VideoCapture(sys.argv[1]); r, _ = c.read(); c.release()
print("1" if r else "0")
PYEOF
)
  if [ "$ok" = "1" ]; then
    echo "  video OK: $(basename "$f")"
    continue
  fi
  echo "  !! UNPLAYABLE: $(basename "$f") — repairing"
  [ -n "$FF" ] || { echo "     no ffmpeg, cannot repair"; continue; }
  mkdir -p "$DIR/repaired"
  "$FF" -y -v error -err_detect ignore_err -fflags +genpts+discardcorrupt \
        -i "$f" -c:v libx264 -preset veryfast -crf 26 \
        "$DIR/repaired/$(basename "$f" .mp4)_repaired.mp4" 2>/dev/null
  n=$($PY - "$DIR/repaired/$(basename "$f" .mp4)_repaired.mp4" <<'PYEOF' 2>/dev/null
import cv2, sys, os
p = sys.argv[1]
if not os.path.exists(p):
    print(0); raise SystemExit
c = cv2.VideoCapture(p); k = 0
while c.read()[0]:
    k += 1
c.release(); print(k)
PYEOF
)
  echo "     repaired -> repaired/$(basename "$f" .mp4)_repaired.mp4  ($n frames)"
done
