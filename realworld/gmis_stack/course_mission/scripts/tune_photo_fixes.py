"""Score the photo-world detector on the probe_photo_range frames — the PRODUCTION
detect_arrays, no simulator. Sweeps the rind AREA_LO ladder; the lunchbox height carve
is whatever perception.py currently ships. hit = right class within 0.6 m of truth
(4 noise draws); fp = any other report in the frame (per draw)."""
import glob
import os
import sys
from pathlib import Path

import numpy as np

RW = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RW))
os.environ["MISSION_VENUE_PHOTO"] = "1"
from course_mission.perception import Perception  # noqa: E402

frames = sorted(glob.glob(str(RW / "course_mission" / "videos_deploy" / "photo_range" / "*.npz")))


def score(area_lo_rind):
    Perception.AREA_LO = {**Perception.AREA_LO, "rind": area_lo_rind}
    rows, fp_cls = [], {}
    for f in frames:
        z = np.load(f, allow_pickle=True)
        rgb = z["rgb"].astype(np.float32)
        truth, cls, gz = z["truth"], str(z["cls"]), float(z["gz"])
        np.random.seed(0)
        hit = fp = 0
        for _ in range(4):
            dets = Perception.detect_arrays(rgb, z["depth"], z["K"], z["cam_p"],
                                            z["cam_R"], gz, add_noise=True)
            got = False
            for d in dets:
                if d["cls"] == cls and np.linalg.norm(np.asarray(d["pos_w"])[:2] - truth[:2]) < 0.6:
                    got = True
                else:
                    fp += 1
                    fp_cls[d["cls"]] = fp_cls.get(d["cls"], 0) + 1
            hit += got
        rows.append((Path(f).stem, hit, fp))
    return rows, fp_cls


for alo in (0.5, 0.3, 0.2, 0.12):
    rows, fp_cls = score(alo)
    tgt = {}
    for name, hit, fp in rows:
        c = name.split("_")[0]
        a, b = tgt.get(c, (0, 0)); tgt[c] = (a + hit, b + 4)
    tot_fp = sum(fp for _, _, fp in rows)
    print(f"AREA_LO(rind)={alo}: " +
          "  ".join(f"{c} {a}/{b}" for c, (a, b) in sorted(tgt.items())) +
          f"  | false reports {tot_fp} {fp_cls}")
    if alo == 0.2:
        for name, hit, fp in rows:
            print(f"    {name:24s} hit {hit}/4 fp {fp}")
