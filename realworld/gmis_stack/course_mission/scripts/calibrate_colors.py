"""Measure the real HSV of every competition object (and of the confusers: red boundary
strips, grass, pebbles, stair treads) as the head camera actually sees them.

The detector thresholds in mission.py were first written from the configured diffuse
colours; rendering adds lighting/shading, and the lunchbox comes from a USD whose colour
is not in the cfg at all. This teleports the robot next to each object, projects the
object's world position into the head camera and reports the HSV of the patch there.
"""

import argparse
import sys
from pathlib import Path

_S = Path(__file__).resolve()
RW = _S.parents[2]
for _p in (str(RW / "source" / "atec_rl_lab"), str(RW / "eunha_rl"), str(RW / "a2pro_rl"), str(RW)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

# Shared machine: GPUs 1-7 are other people's training runs. Pin to GPU 0 BEFORE any
# CUDA-aware import so forgetting the CUDA_VISIBLE_DEVICES prefix cannot land this on
# someone else's device.
import os as _os
_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Calibrate object colours in the head camera.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--per_type", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import imageio.v2 as iio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as mu  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import course_mission  # noqa: F401, E402
from course_mission.launch import head_cam_cfg, HEAD_POS, HEAD_ROT, TASK  # noqa: E402
from course_mission.mission import Perception, _quat_to_mat  # noqa: E402
from course_mission.runner import ground_z_arc  # noqa: E402
import course_sim.terrain as ct  # noqa: E402


def patch_near(cam, p_w, radius=0.09):
    """RGB of every pixel whose UNPROJECTED world point lies within `radius` of p_w.

    Uses only the pixel->world direction that Perception.detect already relies on, so a
    sign/convention error cannot silently sample the sky (the first calibration pass,
    which projected world->pixel, did exactly that).
    """
    rgb = cam.data.output["rgb"][0, ..., :3].cpu().numpy()
    dep = cam.data.output["distance_to_image_plane"][0].cpu().numpy()
    dep = dep.reshape(dep.shape[0], dep.shape[1])
    K = cam.data.intrinsic_matrices[0].cpu().numpy()
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    cp = cam.data.pos_w[0].cpu().numpy()
    R = _quat_to_mat(cam.data.quat_w_ros[0].cpu().numpy())
    H, W = dep.shape
    vv, uu = np.mgrid[0:H, 0:W]
    ok = np.isfinite(dep) & (dep > 0.2) & (dep < 6.0)
    if ok.sum() == 0:
        return None
    d = dep[ok]
    pts = np.stack([(uu[ok] - cx) / fx * d, (vv[ok] - cy) / fy * d, d], axis=1)
    pw = cp[None, :] + pts @ R.T
    near = np.linalg.norm(pw - p_w[None, :], axis=1) < radius
    if near.sum() < 8:
        return None
    return rgb[ok][near]


def main():
    device = args_cli.device or "cuda:0"
    out = RW / "course_mission" / "videos"
    out.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(TASK, device=device, num_envs=1)
    env_cfg.layout_seed = args_cli.seed
    env_cfg.per_type = args_cli.per_type
    env_cfg.scene.head_cam = head_cam_cfg()

    env = gym.make(TASK, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    env.reset()
    robot = uenv.scene["robot"]
    act = torch.zeros(1, uenv.action_manager.total_action_dim, device=device)

    # let everything fall and settle
    for _ in range(150):
        env.step(act)

    layout = env_cfg.mission_layout
    stats = {}
    cam = uenv.scene["head_cam"]

    def sample(px_rgb, vs, us, tag, store):
        h, s, v = Perception._rgb_to_hsv(px_rgb[None, ...] if px_rgb.ndim == 1 else px_rgb)
        store.setdefault(tag, []).append((float(np.median(h)), float(np.median(s)), float(np.median(v))))

    with torch.inference_mode():
        for name, cls, sec, ax, w, _z in layout:
            obj = uenv.scene[name]
            p = obj.data.root_pos_w[0].cpu().numpy()
            # stand the robot 1.1 m behind the object along the course tangent
            th = float(ct.curve_theta(ax))
            eye_x, eye_y = ct.curve_map(ax - 1.15, w)
            gz = 0.05
            root = torch.tensor([[float(eye_x), float(eye_y), gz + 0.45]], device=device)
            quat = mu.quat_from_euler_xyz(
                torch.zeros(1, device=device), torch.zeros(1, device=device),
                torch.tensor([th], device=device))
            robot.write_root_pose_to_sim(torch.cat([root, quat], dim=1))
            robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))
            for _ in range(4):
                env.step(act)

            patch = patch_near(cam, p, radius=0.09)
            if patch is None:
                continue
            h, s, val = Perception._rgb_to_hsv(patch[None, ...])
            stats.setdefault(cls, []).append(
                (float(np.median(h)), float(np.median(s)), float(np.median(val)),
                 float(np.median(patch[:, 0])), float(np.median(patch[:, 1])),
                 float(np.median(patch[:, 2])), float(len(patch))))
            if len(stats[cls]) == 1:
                iio.imwrite(str(out / f"calib_{cls}.png"),
                            cam.data.output["rgb"][0, ..., :3].cpu().numpy())

        # confusers: sample the red strip, grass, pebble, stair, bin pad by looking at
        # known ground points from a fixed pose
        env_stats = {}
        probes = [("strip", ct.SECTION_CENTERS["grass"], 1.80),
                  ("grass_gnd", ct.SECTION_CENTERS["grass"], 0.0),
                  ("mud_gnd", ct.SECTION_CENTERS["mud"], 0.0),
                  ("pebble_gnd", ct.SECTION_CENTERS["pebble"], 0.0),
                  ("stair_gnd", ct.SECTION_CENTERS["stairs"] + 2.0, 0.0)]
        if "mixed" in ct.SECTION_CENTERS:            # legacy course only; KeyError on HK
            probes.insert(1, ("mixed_gnd", ct.SECTION_CENTERS["mixed"], 0.0))
        for tag, ax, w in probes:
            th = float(ct.curve_theta(ax))
            eye_x, eye_y = ct.curve_map(ax - 1.3, w)
            root = torch.tensor([[float(eye_x), float(eye_y), 0.05 + 0.45]], device=device)
            quat = mu.quat_from_euler_xyz(
                torch.zeros(1, device=device), torch.zeros(1, device=device),
                torch.tensor([th], device=device))
            robot.write_root_pose_to_sim(torch.cat([root, quat], dim=1))
            robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))
            for _ in range(4):
                env.step(act)
            gx, gy = ct.curve_map(ax, w)
            p = np.array([float(gx), float(gy), ground_z_arc(ax) + 0.005])
            patch = patch_near(cam, p, radius=0.12)
            if patch is None:
                print(f"[CAL] {tag}: no pixels near the probe point", flush=True)
                continue
            h, s_, v_ = Perception._rgb_to_hsv(patch[None, ...])
            env_stats[tag] = (float(np.median(h)), float(np.median(s_)), float(np.median(v_)),
                              float(np.median(patch[:, 0])), float(np.median(patch[:, 1])),
                              float(np.median(patch[:, 2])))
            iio.imwrite(str(out / f"calib_{tag}.png"),
                        cam.data.output["rgb"][0, ..., :3].cpu().numpy())

    print("\n[CAL] ===== object HSV as rendered (median over samples) =====", flush=True)
    for cls, rows in stats.items():
        a = np.array(rows)
        print(f"[CAL] {cls:10s} n={len(rows)} H={np.median(a[:,0]):6.1f} "
              f"[{a[:,0].min():.0f}-{a[:,0].max():.0f}]  S={np.median(a[:,1]):.3f} "
              f"[{a[:,1].min():.2f}-{a[:,1].max():.2f}]  V={np.median(a[:,2]):.3f} "
              f"[{a[:,2].min():.2f}-{a[:,2].max():.2f}]  rgb=({np.median(a[:,3]):.0f},"
              f"{np.median(a[:,4]):.0f},{np.median(a[:,5]):.0f})", flush=True)
    print("[CAL] ===== confusers =====", flush=True)
    for tag, r in env_stats.items():
        print(f"[CAL] {tag:10s} H={r[0]:6.1f} S={r[1]:.3f} V={r[2]:.3f} rgb=({r[3]:.0f},{r[4]:.0f},{r[5]:.0f})", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
