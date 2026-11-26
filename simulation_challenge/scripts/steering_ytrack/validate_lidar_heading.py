"""VALIDATE LiDAR-based heading estimation for precise dead-reckoning.

Our odom failed because heading (from integrating body yaw-rate) DRIFTS ±0.6-0.8 rad/window.
Forward integration was near-perfect; heading is the whole problem. This script measures how
well LiDAR can pin the heading (drift-free), comparing THREE estimators against ground-truth
world yaw, per frame:

  (0) gyro    : integrate body yaw-rate (our current method; drifts)         [baseline]
  (2) pca     : dominant axis of the body-frame LiDAR scan  = the course     [method ②]
                direction (course is a long straight platform). No integration -> no drift.
  (1) icp     : scan-to-scan ICP (scipy) -> incremental rotation, integrated  [method ①]

Everything the estimators use (body-frame LiDAR points) is what the real onboard LiDAR gives;
ground-truth yaw is used ONLY to score them, never as an input.

  CUDA_VISIBLE_DEVICES=6 TMPDIR=/hdd0/sohyun/tmp PYTHONPATH=/hdd0/sohyun \
    python scripts/validate_lidar_heading.py --headless --steps 1500 \
      --out /hdd0/sohyun/atec_2026_tesk_e/taska_exp/lidar_heading_val
"""
import argparse
import os
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="ATEC-TaskA-B2Piper")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--vx", type=float, default=2.0)
parser.add_argument("--kyaw", type=float, default=3.0)
parser.add_argument("--kd", type=float, default=0.4)
parser.add_argument("--ky", type=float, default=0.5)
parser.add_argument("--deadzone", type=float, default=0.4)
parser.add_argument("--ywander_amp", type=float, default=1.5)
parser.add_argument("--ywander_period", type=int, default=500)
parser.add_argument("--steps", type=int, default=1500)
parser.add_argument("--out", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
import json  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.math import quat_rotate_inverse  # noqa: E402
import atec_rl_lab.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from atec_rl_lab.tasks.task_base.action_base import apply_safe_action_spec  # noqa: E402
from atec_2026_tesk_e.ATEC2026_Simulation_Challenge.demo.solution_rl import AlgSolution  # noqa: E402

solution = AlgSolution()


def world_yaw(robot):
    q = robot.data.root_quat_w[0]
    return math.atan2(2 * (float(q[0]) * float(q[3]) + float(q[1]) * float(q[2])),
                      1 - 2 * (float(q[2]) ** 2 + float(q[3]) ** 2))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def body_scan_2d(robot, lidar):
    """Body-frame 2D LiDAR points (what the onboard sensor gives). Uses the robot's own
    orientation to express hits in body frame -- exactly what a real body-fixed LiDAR reports."""
    hits = lidar.data.ray_hits_w[0]                    # (N,3) world
    base = robot.data.root_pos_w[0]
    quat = robot.data.root_quat_w[0].unsqueeze(0)
    rel = hits - base.unsqueeze(0)                     # (N,3) world, robot-centered
    body = quat_rotate_inverse(quat.expand(rel.shape[0], -1), rel)  # (N,3) body
    p = body[:, :2].cpu().numpy()                      # (N,2) body-plane
    rng = np.linalg.norm(p, axis=1)
    good = np.isfinite(p).all(1) & (rng > 0.3) & (rng < 9.5)  # drop no-hits (maxrange)
    return p[good]


def heading_pca(p, prev_axis):
    """Method ②: dominant axis of the scan = course direction in body frame. The robot's
    heading error rel. to the course axis is the negative of that axis angle. PCA gives a line
    (180deg ambiguous); resolve by staying near the previous estimate / forward (+x)."""
    if len(p) < 20:
        return None
    p = p - p.mean(0)
    cov = p.T @ p
    w, v = np.linalg.eigh(cov)
    axis = v[:, np.argmax(w)]                          # dominant direction (unit)
    ang = math.atan2(axis[1], axis[0])                 # course axis angle in body frame
    # resolve 180deg ambiguity: pick the sign closest to "forward" (prev or 0)
    ref = prev_axis if prev_axis is not None else 0.0
    if abs(wrap(ang - ref)) > math.pi / 2:
        ang = wrap(ang + math.pi)
    # robot heading rel. to course = -(course direction in body frame)
    return wrap(-ang), ang


def icp_delta_yaw(prev_p, cur_p, iters=10):
    """Method ①: 2D ICP between consecutive body scans -> incremental yaw (robot rotation
    between frames, in body). Point-to-point, rotation+translation, small-angle."""
    if prev_p is None or len(prev_p) < 20 or len(cur_p) < 20:
        return 0.0
    src = cur_p.copy()
    tree = cKDTree(prev_p)
    R_total = np.eye(2)
    t_total = np.zeros(2)
    for _ in range(iters):
        _, idx = tree.query(src)
        dst = prev_p[idx]
        sm, dm = src.mean(0), dst.mean(0)
        H = (src - sm).T @ (dst - dm)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1; R = Vt.T @ U.T
        t = dm - R @ sm
        src = (R @ src.T).T + t
        R_total = R @ R_total; t_total = R @ t_total + t
    return math.atan2(R_total[1, 0], R_total[0, 0])    # incremental yaw


def main():
    os.makedirs(args_cli.out, exist_ok=True)
    dev = "cuda"
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device,
                            num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    action_spec = solution.get_action_spec() if hasattr(solution, "get_action_spec") else None
    env_cfg = apply_safe_action_spec(env_cfg, json.dumps(action_spec))
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    obs, _ = env.reset()
    robot = env.unwrapped.scene["robot"]
    lidar = env.unwrapped.scene["lidar_sensor"]
    dt = float(env.unwrapped.step_dt)

    # estimator states, all anchored to the TRUE yaw at t=0 (a one-time init, like a known start)
    gt0 = world_yaw(robot)
    yaw_gyro = gt0
    yaw_icp = gt0
    prev_p = None
    prev_axis = None
    rows = []

    for step in range(args_cli.steps):
        with torch.inference_mode():
            gt = world_yaw(robot)
            yaw_rate = float(robot.data.root_ang_vel_b[0, 2])
            cur_y = float(robot.data.root_pos_w[0, 1])

            # (0) gyro integration (our current method)
            yaw_gyro = wrap(yaw_gyro + yaw_rate * dt)

            # scan
            p = body_scan_2d(robot, lidar)

            # (2) PCA dominant-direction heading (absolute, no drift)
            pca = heading_pca(p, prev_axis)
            if pca is not None:
                yaw_pca, prev_axis = pca
                # PCA gives heading REL. to course axis; course runs along world-x, so this IS
                # the (drift-free) world yaw estimate (up to the course axis alignment).
                yaw_pca = wrap(yaw_pca)
            else:
                yaw_pca = float("nan")

            # (1) ICP incremental -> integrate
            dyaw = icp_delta_yaw(prev_p, p)
            yaw_icp = wrap(yaw_icp + dyaw)
            prev_p = p

            rows.append((step, gt, yaw_gyro, yaw_pca, yaw_icp, cur_y))

            # PD teacher drive (keep it on course so we get a realistic trajectory)
            tgt_y = args_cli.ywander_amp * math.sin(2 * math.pi * step / max(1, args_cli.ywander_period))
            cross = cur_y - tgt_y
            tgt_yaw = 0.0 if abs(cross) < args_cli.deadzone else max(-1.2, min(1.2, -args_cli.ky * cross))
            err = wrap(gt - tgt_yaw)
            wz = max(-0.8, min(0.8, -args_cli.kyaw * err - args_cli.kd * yaw_rate))
            solution.fixed_velocity_commands = torch.tensor(
                [args_cli.vx, 0.0, wz], device=solution.device, dtype=torch.float32).view(1, 3)
            resp = solution.predicts(obs, 0.0)
            act = (torch.zeros((1, 20), device=dev) if (resp.get("giveup") or not resp["action"])
                   else torch.tensor(resp["action"], dtype=torch.float32, device=dev).view(1, -1))
            obs, _, term, trunc, _ = env.step(act)
            if step % 200 == 0:
                print(f"[step {step:4d}] gt={gt:+.3f} gyro={yaw_gyro:+.3f} pca={yaw_pca:+.3f} icp={yaw_icp:+.3f}", flush=True)
            if bool(term[0]):
                obs, _ = env.reset()
                # re-anchor integrators on reset (position/orientation jumped)
                gt0 = world_yaw(robot); yaw_gyro = gt0; yaw_icp = gt0; prev_p = None

    arr = np.array(rows, dtype=np.float32)
    np.savez_compressed(os.path.join(args_cli.out, "heading_val.npz"), rows=arr)

    # scoring: error vs gt (wrapped)
    def err_stats(col):
        e = np.array([wrap(float(a) - float(g)) for g, a in zip(arr[:, 1], arr[:, col]) if np.isfinite(a)])
        return np.sqrt((e ** 2).mean()), np.abs(e).mean(), np.abs(e).max()
    print("\n=== HEADING ERROR vs ground-truth (rad) ===", flush=True)
    for name, col in [("gyro (current)", 2), ("pca  (method ②)", 3), ("icp  (method ①)", 4)]:
        rmse, mae, mx = err_stats(col)
        print(f"  {name:16s} RMSE={rmse:.3f}  MAE={mae:.3f}  max={mx:.3f}  (={math.degrees(rmse):.1f}deg RMSE)", flush=True)
    print(f"\n  saved -> {os.path.join(args_cli.out, 'heading_val.npz')}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
