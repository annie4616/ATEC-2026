"""Diagnose the head-camera detector on a known scene.

Places the robot a fixed distance behind a chosen object and reports, in order:
  1. is the UNPROJECTION correct?   min distance from any unprojected pixel to the
     object's true position (should be ~1-3 cm; >0.3 m means the camera extrinsics or
     the depth convention are wrong)
  2. is the GROUND HEIGHT right?    percentiles of (unprojected z - assumed ground z)
  3. what does the FOREGROUND mask contain?  per-blob span / height / hue / class
  4. what colour are the object's own pixels?  (pixels within 8 cm of the true position)
Dumps rgb + a foreground-mask png for eyeballing.
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

parser = argparse.ArgumentParser(description="Detector diagnosis.")
parser.add_argument("--speed", type=str, default="s075")
parser.add_argument("--depth", type=str, default="0.24")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--per_type", type=int, default=4)
parser.add_argument("--walk_ckpt", type=str, default="")
parser.add_argument("--grasp_ckpt", type=str, default="")
parser.add_argument("--stand_back", type=float, default=1.3)
parser.add_argument("--out_dir", type=str, default=str(RW / "course_mission" / "videos"))
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
import course_sim.terrain as ct  # noqa: E402
from course_mission.launch import head_cam_cfg, HEAD_POS, HEAD_ROT, TASK  # noqa: E402
from course_mission.mission import Perception, _blobs, _quat_to_mat, arc_of  # noqa: E402
from course_mission.runner import ground_z_arc  # noqa: E402


def main():
    device = args_cli.device or "cuda:0"
    out = Path(args_cli.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(TASK, device=device, num_envs=1)
    env_cfg.layout_seed = args_cli.seed
    env_cfg.per_type = args_cli.per_type
    env_cfg.scene.head_cam = head_cam_cfg()

    env = gym.make(TASK, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    env.reset()
    robot = uenv.scene["robot"]
    cam = uenv.scene["head_cam"]
    act = torch.zeros(1, uenv.action_manager.total_action_dim, device=device)
    for _ in range(200):
        env.step(act)

    targets = []
    for cls in ("can", "rind", "lunchbox", "redbag"):
        for name, c, sec, ax, w, _z in env_cfg.mission_layout:
            if c == cls:
                targets.append((name, cls, sec, ax, w))
                break

    with torch.inference_mode():
        for name, cls, sec, _ax, _w in targets:
            p = uenv.scene[name].data.root_pos_w[0].cpu().numpy()
            oax, ow = arc_of(float(p[0]), float(p[1]))
            th = float(ct.curve_theta(oax))
            ex, ey = ct.curve_map(oax - args_cli.stand_back, ow)
            root = torch.tensor([[float(ex), float(ey), ground_z_arc(oax) + 0.50]], device=device)
            quat = mu.quat_from_euler_xyz(torch.zeros(1, device=device),
                                          torch.zeros(1, device=device),
                                          torch.tensor([th], device=device))
            robot.write_root_pose_to_sim(torch.cat([root, quat], dim=1))
            robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))
            for _ in range(12):
                env.step(act)

            rgb = cam.data.output["rgb"][0, ..., :3].cpu().numpy()
            dep = cam.data.output["distance_to_image_plane"][0].cpu().numpy()
            dep = dep.reshape(dep.shape[0], dep.shape[1])
            K = cam.data.intrinsic_matrices[0].cpu().numpy()
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            cp = cam.data.pos_w[0].cpu().numpy()
            R = _quat_to_mat(cam.data.quat_w_ros[0].cpu().numpy())
            H, W = dep.shape
            vv, uu = np.mgrid[0:H, 0:W]
            fin = np.isfinite(dep) & (dep > 0.05) & (dep < 8.0)
            d = dep[fin]
            pts = np.stack([(uu[fin] - cx) / fx * d, (vv[fin] - cy) / fy * d, d], axis=1)
            pw = cp[None, :] + pts @ R.T

            gz = ground_z_arc(oax)
            dd = np.linalg.norm(pw - p[None, :], axis=1)
            j = int(np.argmin(dd))
            print(f"\n[DBG] ===== {name} {cls} on {sec} =====", flush=True)
            print(f"[DBG] cam pos={np.round(cp,3).tolist()} "
                  f"quat_ros={np.round(cam.data.quat_w_ros[0].cpu().numpy(),3).tolist()}", flush=True)
            print(f"[DBG] object true pos={np.round(p,3).tolist()} assumed ground z={gz:.3f}", flush=True)
            print(f"[DBG] 1) unprojection: closest unprojected point is {dd[j]:.3f} m away "
                  f"at {np.round(pw[j],3).tolist()} (fx={fx:.1f} K ok)", flush=True)
            zz = pw[:, 2] - gz
            print(f"[DBG] 2) height above assumed ground: p05={np.percentile(zz,5):.3f} "
                  f"p50={np.percentile(zz,50):.3f} p95={np.percentile(zz,95):.3f} "
                  f"max={zz.max():.3f}", flush=True)
            near = dd < 0.10
            print(f"[DBG] 4) object pixels: n={int(near.sum())}", flush=True)
            if near.sum() > 0:
                px = rgb[vv[fin][near], uu[fin][near]]
                h, s, v = Perception._rgb_to_hsv(px)
                print(f"[DBG]    rgb=({np.median(px[:,0]):.0f},{np.median(px[:,1]):.0f},"
                      f"{np.median(px[:,2]):.0f}) H={np.median(h):.1f} S={np.median(s):.3f} "
                      f"V={np.median(v):.3f} -> classify="
                      f"{Perception.classify(float(np.median(h)), float(np.median(s)))}", flush=True)
                print(f"[DBG]    their height above ground: "
                      f"{np.round(np.percentile(pw[near,2]-gz,[5,50,95]),3).tolist()}", flush=True)

            det = Perception(cam, device).detect(gz)
            print(f"[DBG] 3) detect() returned {len(det)} blobs:", flush=True)
            for b in det:
                err = float(np.linalg.norm(b["pos_w"][:2] - p[:2]))
                print(f"[DBG]    {b['cls']:10s} pos={np.round(b['pos_w'],2).tolist()} "
                      f"n={b['n_pix']} span={b['span']:.2f} above={b['above']:.3f} "
                      f"H={b['hue']:.0f} S={b['sat']:.2f} | err_to_target={err:.2f}", flush=True)

            fgimg = np.zeros((H, W), np.uint8)
            sel = (zz > Perception.MIN_ABOVE) & (zz < Perception.MAX_ABOVE) & (d > Perception.NEAR_CUT) & (d < Perception.FAR_CUT)
            fgimg[vv[fin][sel], uu[fin][sel]] = 255
            iio.imwrite(str(out / f"dbg_{cls}_rgb.png"), rgb)
            iio.imwrite(str(out / f"dbg_{cls}_fg.png"), fgimg)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
