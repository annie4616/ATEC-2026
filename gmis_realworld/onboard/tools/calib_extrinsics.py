#!/usr/bin/env python3
"""Camera intrinsics and mount pose from a checkerboard.

Real-hardware confirmations #2 and #3 in one sitting.  Both matter more than they look:

  * The detector's constants were fitted at fx = fy = 488.7 over a 66.4 deg horizontal
    field.  If the real lens is wider -- and Unitree forward cameras usually are -- the
    remap into that virtual pinhole needs the true K and distortion, or every span and
    lateral threshold in `perception.py` is being applied to the wrong geometry.
  * The mount pose is worse.  `cam_p`/`cam_rot` enter every unprojection, and the ground
    intersection is a lever arm: 2 deg of pitch error is 0.31 m of range error at 2 m,
    which is larger than the entire grasp tolerance.  The target is under 0.3 deg.

Procedure (about 30 minutes, needs a printed checkerboard and a tape measure):

  1. --capture   : hold the board at varied distances and angles; ~20 views.
  2. --intrinsics: solve K and the distortion coefficients.
  3. --extrinsics: lay the board FLAT on the ground in front of the standing robot with its
                   origin corner a measured distance ahead on the centreline; the solve then
                   gives the camera pose in the body frame directly, because a board on the
                   ground and a robot standing on the same ground share it.

Writes yaml ready to paste into configs/sensors.yaml.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _cv():
    try:
        import cv2
        return cv2
    except ImportError:
        print("opencv-python-headless is required for calibration", file=sys.stderr)
        raise SystemExit(2)


def capture(out: str, device: int, n: int, cols: int, rows: int) -> int:
    cv2 = _cv()
    Path(out).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(device)
    got = 0
    print(f"press Enter to keep a view ({cols}x{rows} inner corners), q to stop")
    while got < n:
        ok, frame = cap.read()
        if not ok:
            continue
        found, _ = cv2.findChessboardCorners(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (cols, rows),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK)
        print(f"  board {'VISIBLE' if found else 'not found'}   kept {got}/{n}",
              end="\r", flush=True)
        k = input() if found else ""
        if k.strip().lower() == "q":
            break
        if found:
            cv2.imwrite(str(Path(out) / f"view_{got:02d}.png"), frame)
            got += 1
    cap.release()
    print(f"\nwrote {got} views to {out}")
    return 0


def intrinsics(images: str, cols: int, rows: int, square: float, fisheye: bool) -> int:
    cv2 = _cv()
    files = sorted(glob.glob(str(Path(images) / "*.png")))
    if len(files) < 8:
        print(f"need at least 8 views, found {len(files)}", file=sys.stderr)
        return 1
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    obj, img, shape = [], [], None
    for f in files:
        g = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
        shape = g.shape[::-1]
        ok, c = cv2.findChessboardCorners(g, (cols, rows))
        if not ok:
            continue
        c = cv2.cornerSubPix(g, c, (11, 11), (-1, -1),
                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
        obj.append(objp)
        img.append(c)
    print(f"using {len(obj)}/{len(files)} views")
    if fisheye:
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rms, K, D, _, _ = cv2.fisheye.calibrate(
            [o.reshape(-1, 1, 3) for o in obj], img, shape, K, D,
            flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW)
        dist = D.ravel()
    else:
        rms, K, D, _, _ = cv2.calibrateCamera(obj, img, shape, None, None)
        dist = D.ravel()
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    hfov = 2 * np.degrees(np.arctan(shape[0] / 2 / fx))
    print(f"\nRMS reprojection error {rms:.3f} px  (below ~0.5 is good)")
    print(f"K: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}   HFOV {hfov:.1f} deg")
    print(f"dist: {np.round(dist, 5).tolist()}")
    print("\n-- paste into configs/sensors.yaml under camera.source --")
    print(f"    model: {'fisheye' if fisheye else 'pinhole'}")
    print(f"    width: {shape[0]}\n    height: {shape[1]}")
    print(f"    K: [{fx:.3f}, {fy:.3f}, {cx:.3f}, {cy:.3f}]")
    print(f"    dist: {np.round(dist, 6).tolist()}")
    if rms > 1.0:
        print("\nWARNING: RMS above 1 px -- recapture with more varied board angles")
    return 0


def extrinsics(image: str, cols: int, rows: int, square: float,
               ahead: float, body_height: float) -> int:
    """Board flat on the ground, origin corner `ahead` metres in front of the base."""
    cv2 = _cv()
    from onboard import cfg as C
    src = (C.load("sensors")["camera"].get("source") or {})
    if not src.get("K"):
        print("run --intrinsics first and paste the result into sensors.yaml",
              file=sys.stderr)
        return 1
    fx, fy, cx, cy = (float(v) for v in src["K"])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    D = np.asarray(src.get("dist") or [0, 0, 0, 0, 0], float)

    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    # Board frame -> body frame: x forward from `ahead`, y left, z at the ground.
    objp_body = np.stack([objp[:, 0] + ahead, objp[:, 1], np.full(len(objp), -body_height)], 1)

    g = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY)
    ok, c = cv2.findChessboardCorners(g, (cols, rows))
    if not ok:
        print("no board found in that image", file=sys.stderr)
        return 1
    c = cv2.cornerSubPix(g, c, (11, 11), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
    ok, rvec, tvec = cv2.solvePnP(objp_body.astype(np.float32), c, K, D)
    if not ok:
        print("solvePnP failed", file=sys.stderr)
        return 1
    R_cb, _ = cv2.Rodrigues(rvec)              # body -> camera (ROS axes)
    R_bc = R_cb.T
    p_cam = (-R_bc @ tvec).ravel()             # camera position in the body frame

    from onboard.perceive.camadapter import ROS_IN_BODY
    R_mount = R_bc @ ROS_IN_BODY.T             # strip the ROS-axis convention
    pitch = np.degrees(np.arcsin(-R_mount[2, 0]))
    yaw = np.degrees(np.arctan2(R_mount[1, 0], R_mount[0, 0]))
    roll = np.degrees(np.arctan2(R_mount[2, 1], R_mount[2, 2]))

    print("\n-- paste into configs/sensors.yaml under camera.extrinsics --")
    print(f"    xyz: [{p_cam[0]:.4f}, {p_cam[1]:.4f}, {p_cam[2]:.4f}]")
    print(f"    pitch_deg: {pitch:.2f}\n    yaw_deg: {yaw:.2f}\n    roll_deg: {roll:.2f}")
    print(f"\n  nominal from the catalogue: xyz [0.3381, 0.0336, 0.0525], pitch 22.00")
    d = np.linalg.norm(p_cam - np.array([0.3381, 0.0336, 0.0525]))
    print(f"  difference from nominal: {d * 1000:.0f} mm, {abs(pitch - 22.0):.2f} deg pitch")
    if abs(pitch - 22.0) > 2.0:
        print("  NOTE: more than 2 deg from nominal -- that is 0.31 m of range error at 2 m,"
              "\n        so this measurement matters more than the catalogue value")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--intrinsics", action="store_true")
    ap.add_argument("--extrinsics", metavar="IMAGE")
    ap.add_argument("--dir", default="onboard/logs/calib")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--views", type=int, default=20)
    ap.add_argument("--cols", type=int, default=9, help="inner corners across")
    ap.add_argument("--rows", type=int, default=6, help="inner corners down")
    ap.add_argument("--square", type=float, default=0.025, help="square size, metres")
    ap.add_argument("--fisheye", action="store_true")
    ap.add_argument("--ahead", type=float, default=0.8,
                    help="board origin corner, metres ahead of the base")
    ap.add_argument("--body-height", type=float, default=0.413)
    a = ap.parse_args()

    if a.capture:
        raise SystemExit(capture(a.dir, a.device, a.views, a.cols, a.rows))
    if a.intrinsics:
        raise SystemExit(intrinsics(a.dir, a.cols, a.rows, a.square, a.fisheye))
    if a.extrinsics:
        raise SystemExit(extrinsics(a.extrinsics, a.cols, a.rows, a.square,
                                    a.ahead, a.body_height))
    ap.print_help()
    raise SystemExit(1)
