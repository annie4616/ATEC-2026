"""Validate the map-free venue navigation signals against simulator ground truth.

At the venue the course shape is unknown: the robot must drive forward, keep itself
between the red belts, and recognise the end of the course as the moment it SEES the bin
row. This script drives the course in two halves:

  phase A (steps 0..N/2)   MAP-based steering (the sim's known centreline), while the
                           belt/bin estimators run open-loop -> measures estimator ERROR
                           against ground truth without risking the walk.
  phase B (steps N/2..N)   BELT-based steering only: heading and lateral offset come
                           from perception.detect_belt, the map is not consulted ->
                           measures whether the estimator is good enough to DRIVE on.

Logged per sample: belt lat estimate vs true lat, heading estimate vs true course
tangent error, and the first bin-row sighting (estimated position vs the true bins).
"""
import argparse
import os as _os
import sys
from pathlib import Path as _P
_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
_RW = _P(__file__).resolve().parents[2]
for _p in (str(_RW / "source" / "atec_rl_lab"), str(_RW / "eunha_rl"),
           str(_RW / "a2pro_rl"), str(_RW)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=5200)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--speed", type=str, default="s075")  # resolve_ckpts expects these
parser.add_argument("--depth", type=str, default="0.24")
parser.add_argument("--per_type", type=int, default=None)
parser.add_argument("--walk_ckpt", type=str, default="")
parser.add_argument("--grasp_ckpt", type=str, default="")
parser.add_argument("--out", type=str, default="")
parser.add_argument("--fuse", action="store_true",
                    help="phase B steers on the belt+odometry fusion (navfuse) instead of "
                         "the raw EMA; the fused estimate is scored in BOTH phases")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app = AppLauncher(args_cli).app

import gymnasium as gym          # noqa: E402
import numpy as np               # noqa: E402
import torch                     # noqa: E402
from pathlib import Path         # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import course_sim.terrain as ct  # noqa: E402
from course_mission.launch import head_cam_cfg, HEAD_POS, HEAD_ROT, TASK, resolve_ckpts  # noqa: E402
from course_mission.mission import (  # noqa: E402
    ArmDriver, HeadCamMount, LegDriver, arc_of,
)
from course_mission.runner import ground_z_world  # noqa: E402
from course_mission import perception as pc  # noqa: E402
from course_mission.navfuse import BeltOdomFuser  # noqa: E402


def main():
    device = args_cli.device or "cuda:0"
    ck = resolve_ckpts(args_cli)
    env_cfg = parse_env_cfg(TASK, device=device, num_envs=1)
    env_cfg.layout_seed = args_cli.seed
    if args_cli.per_type is not None:
        env_cfg.per_type = args_cli.per_type
    env_cfg.scene.head_cam = head_cam_cfg()
    env = gym.make(TASK, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    env.reset()
    robot = uenv.scene["robot"]
    cam = uenv.scene["head_cam"]
    legs = LegDriver(robot, device, ck["walk"], ck["crouch"], ck["adapt"])
    arm = ArmDriver(robot, device)
    head_mount = HeadCamMount(cam, robot, device, HEAD_POS, HEAD_ROT)
    travel_q = torch.tensor([0.0, 1.2, -1.2, 0.0, 0.6, 0.0], device=device)
    act = torch.zeros(1, uenv.action_manager.total_action_dim, device=device)

    bins_true = np.array(ct.curve_map(ct.SECTION_CENTERS["bins"], 0.0))
    lat_err, head_err = [], []
    lat_err_b, drove_b = [], 0
    # The fusion runs in BOTH phases so its error can be scored against truth even while
    # the map is doing the driving; --fuse additionally hands it the wheel in phase B.
    fuser = BeltOdomFuser()
    fus_err, fus_err_b = [], []
    rng = np.random.default_rng(args_cli.seed)
    first_bins = None
    belt = None
    smooth_lat, smooth_head = 0.0, 0.0
    half = args_cli.steps // 2

    with torch.inference_mode():
        for i in range(args_cli.steps):
            d = robot.data
            X = float(d.root_pos_w[0, 0]); Y = float(d.root_pos_w[0, 1])
            ax, w = arc_of(X, Y)
            yaw = float(torch.atan2(
                2 * (d.root_quat_w[0, 0] * d.root_quat_w[0, 3]
                     + d.root_quat_w[0, 1] * d.root_quat_w[0, 2]),
                1 - 2 * (d.root_quat_w[0, 2] ** 2 + d.root_quat_w[0, 3] ** 2)))
            th = float(ct.curve_theta(ax))

            # Proprioception for the fusion: leg-odometry body velocity and the IMU yaw
            # rate, both with the error a real estimator has. Never the map, never truth.
            _v = d.root_lin_vel_b[0]
            fuser.predict(float(_v[0]) + float(rng.normal(0, 0.04)),
                          float(_v[1]) + float(rng.normal(0, 0.04)),
                          float(d.root_ang_vel_b[0, 2]) + float(rng.normal(0, 0.03)),
                          0.02)

            if i % 10 == 0:                      # 5 Hz, the deployed detect rate
                rgb = cam.data.output["rgb"][0, ..., :3].cpu().numpy()
                K = cam.data.intrinsic_matrices[0].cpu().numpy()
                cp, cR = head_mount.pos_w, head_mount.R_ros
                gz = float(ground_z_world(X, Y))
                belt = pc.detect_belt(rgb, K, cp, cR, gz)
                if belt is not None:
                    true_head_err = (th - yaw + np.pi) % (2 * np.pi) - np.pi
                    lat_err.append(belt["lat"] - w)
                    head_err.append(belt["heading"] - true_head_err)
                    if i >= half:
                        lat_err_b.append(belt["lat"] - w)
                    fuser.update(belt["lat"], belt["heading"],
                                 halfwidth=belt.get("halfwidth"))
                if first_bins is None:
                    br = pc.detect_bins(rgb, K, cp, cR, gz)
                    if br is not None:
                        est = br["centre_w"][:2]
                        first_bins = (i, float(np.hypot(*(est - bins_true))),
                                      float(np.hypot(X - bins_true[0], Y - bins_true[1])),
                                      br["n_seen"])
                        print(f"[NAV] bins first seen at step {i}: est-vs-true err "
                              f"{first_bins[1]:.2f} m, robot {first_bins[2]:.1f} m away, "
                              f"{first_bins[3]} colors", flush=True)

            # Score the FUSED estimate every step, fix or no fix -- the point of the
            # filter is that it always has an answer, so it must be judged continuously
            # rather than only on the frames where the detector happened to return.
            fus_err.append(abs(fuser.lat - w))
            if i >= half:
                fus_err_b.append(abs(fuser.lat - w))

            cmd = torch.zeros(1, 3, device=device)
            if i < half:
                err = (th - 0.8 * np.clip(w, -0.8, 0.8) - yaw + np.pi) % (2 * np.pi) - np.pi
                cmd[0, 0] = 0.55
                cmd[0, 1] = float(np.clip(-0.5 * w, -0.3, 0.3))
                cmd[0, 2] = float(np.clip(1.2 * err, -0.7, 0.7))
            else:
                # MAP-FREE steering, honestly this time. The first run quietly fell back
                # to the map whenever the belt fix dropped out, so only 38 % of "phase B"
                # was actually belt-driven. Here the fallback is what a real robot has:
                # the LAST belt-derived command, decayed toward straight-ahead -- 262 mm /
                # 5.4 deg median fixes at ~2 Hz effective are what must carry the walk.
                drove_b += 1
                if args_cli.fuse:
                    _vx, _vy, _wz = fuser.steer()
                    cmd[0, 0], cmd[0, 1], cmd[0, 2] = _vx, _vy, _wz
                    if first_bins is not None and i - first_bins[0] > 150:
                        print(f"[NAV] STOPPING at the sighted bins (arc {ax:+.1f} "
                              f"lat {w:+.2f})", flush=True)
                        break
                    arm.set_joints(travel_q)
                    arm.set_gripper(False)
                    arm.write_gripper()
                    head_mount.sync()
                    act[:, :12] = legs.step(cmd)
                    env.step(act)
                    if i % 500 == 0:
                        print(f"[NAV] step {i} arc {ax:+.1f} lat {w:+.2f} "
                              f"fused {fuser.lat:+.2f}+-{fuser.lat_sigma:.2f} "
                              f"c={fuser.curvature:+.3f} "
                              f"belt={'ok' if belt else '--'}", flush=True)
                    continue
                if belt is not None:
                    smooth_lat = 0.6 * smooth_lat + 0.4 * belt["lat"]
                    smooth_head = 0.6 * smooth_head + 0.4 * belt["heading"]
                else:
                    smooth_head *= 0.98            # keep heading, bleed the turn
                cmd[0, 0] = 0.5
                cmd[0, 1] = float(np.clip(-0.5 * smooth_lat, -0.3, 0.3))
                cmd[0, 2] = float(np.clip(1.2 * (smooth_head
                                                 - 0.8 * np.clip(smooth_lat, -0.8, 0.8)),
                                          -0.7, 0.7))
                if first_bins is not None and i - first_bins[0] > 150:
                    print(f"[NAV] STOPPING at the sighted bins (arc {ax:+.1f} "
                          f"lat {w:+.2f})", flush=True)
                    break
            arm.set_joints(travel_q)
            arm.set_gripper(False)
            arm.write_gripper()
            head_mount.sync()
            act[:, :12] = legs.step(cmd)
            env.step(act)
            if i % 500 == 0:
                print(f"[NAV] step {i} arc {ax:+.1f} lat {w:+.2f} "
                      f"belt={'ok' if belt else '--'}", flush=True)

    ae = np.abs(np.array(lat_err)); he = np.degrees(np.abs(np.array(head_err)))
    print(f"[NAV] belt fixes: {len(lat_err)} samples", flush=True)
    if len(ae):
        print(f"[NAV] lat error   median {np.median(ae)*1000:.0f} mm  "
              f"p90 {np.percentile(ae, 90)*1000:.0f} mm", flush=True)
        print(f"[NAV] head error  median {np.median(he):.1f} deg  "
              f"p90 {np.percentile(he, 90):.1f} deg", flush=True)
    X = float(robot.data.root_pos_w[0, 0]); Y = float(robot.data.root_pos_w[0, 1])
    ax, w = arc_of(X, Y)
    if len(fus_err):
        fe = np.array(fus_err); feb = np.array(fus_err_b)
        print(f"[NAV] FUSED lat error (every step, {len(fe)} samples): "
              f"median {np.median(fe)*1000:.0f} mm  p90 {np.percentile(fe, 90)*1000:.0f} mm",
              flush=True)
        if len(feb):
            print(f"[NAV] FUSED lat error phase B only: median {np.median(feb)*1000:.0f} mm "
                  f"p90 {np.percentile(feb, 90)*1000:.0f} mm", flush=True)
        print(f"[NAV] fusion gate: {fuser.n_acc} accepted / {fuser.n_rej} rejected", flush=True)
    print(f"[NAV] belt-only drove {drove_b} steps; final arc {ax:+.1f} lat {w:+.2f} "
          f"(|lat|<1.98 = stayed in) ", flush=True)
    if first_bins is None:
        print("[NAV] bins NEVER seen", flush=True)
    print("[NAV] DONE", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
