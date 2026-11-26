"""FALLACY-1 CHECK: does the student's camera / lidar actually CONTAIN the steering
cue the kyaw teacher uses?

The kyaw teacher steers with privileged world yaw. For IL to work, the student's
sensors (head RGB, head depth, lidar height-scan) must carry enough information to
predict the teacher's wz. This script drives Task A with the kyaw teacher and, every
few steps, records synchronized (head_rgb, head_depth, lidar_scan, teacher_wz,
world_yaw, x, y). It also dumps a handful of RGB/depth frames as PNGs for eyeballing.

Offline afterwards we (a) look at the PNGs for course boundary / terrain flow and
(b) run a linear probe from a pretrained encoder's features -> teacher_wz to quantify
whether the cue is actually present (see probe_steering_linprobe.py).

  CUDA_VISIBLE_DEVICES=1 python scripts/probe_taska_steering_cues.py \
      --vx 2.5 --kyaw 3.0 --wz_max 0.8 --steps 3000 --every 5 \
      --out /hdd0/sohyun/atec_2026_tesk_e/taska_steering_probe
"""
import argparse
import os
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Probe Task A steering cues in student sensors.")
parser.add_argument("--task", type=str, default="ATEC-TaskA-B2Piper")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--vx", type=float, default=2.5)
parser.add_argument("--kyaw", type=float, default=3.0)
parser.add_argument("--wz_max", type=float, default=0.8)
parser.add_argument("--target_yaw", type=float, default=0.0)
parser.add_argument("--dyaw_amp", type=float, default=0.0,
                    help="amplitude (rad) of a wandering target-yaw so the teacher steers BOTH ways "
                         "-> diverse wz labels (fixes the near-constant-wz problem). 0 disables.")
parser.add_argument("--dyaw_period", type=int, default=300,
                    help="steps per full left-right cycle of the wandering target yaw.")
parser.add_argument("--ytrack", action="store_true", default=False,
                    help="LANE-KEEPING teacher: steer to hold a target cross-track y (not a target yaw). "
                         "The wz label becomes 'turn back toward the course center-line', which is exactly "
                         "what a vision student must learn from the red boundary. Use with --ywander to "
                         "sweep the target y left/right so the boundary appears on both sides of the frame.")
parser.add_argument("--ywander_amp", type=float, default=2.0,
                    help="amplitude (m) of a wandering TARGET y so the robot drifts left/right across the "
                         "course and the red strips appear at varied positions in the camera. 0 = hold y=0.")
parser.add_argument("--ywander_period", type=int, default=400,
                    help="steps per full left-right cycle of the wandering target y.")
parser.add_argument("--ky", type=float, default=0.6,
                    help="lane-keeping gain: target_yaw = -ky * (y - target_y), i.e. face back toward center.")
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--every", type=int, default=5, help="record every N steps")
parser.add_argument("--png_every", type=int, default=150, help="dump a PNG every N recorded frames")
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

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
import atec_rl_lab.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from atec_rl_lab.tasks.task_base.action_base import apply_safe_action_spec  # noqa: E402
from atec_2026_tesk_e.ATEC2026_Simulation_Challenge.demo.solution_rl import AlgSolution  # noqa: E402

solution = AlgSolution()
_VX = args_cli.vx


def _world_yaw(robot):
    q = robot.data.root_quat_w[0]  # w,x,y,z
    return math.atan2(2 * (float(q[0]) * float(q[3]) + float(q[1]) * float(q[2])),
                      1 - 2 * (float(q[2]) ** 2 + float(q[3]) ** 2))


def _save_png(arr, path, is_depth=False):
    """Save an HxW(x3) array as PNG using matplotlib (no extra deps)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4, 4))
    if is_depth:
        d = np.asarray(arr, dtype=np.float32)
        d = np.clip(d, 0, np.percentile(d[np.isfinite(d)], 95) if np.isfinite(d).any() else 1)
        plt.imshow(d, cmap="viridis")
    else:
        plt.imshow(np.asarray(arr).astype(np.uint8))
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()


def main():
    os.makedirs(args_cli.out, exist_ok=True)
    png_dir = os.path.join(args_cli.out, "frames")
    os.makedirs(png_dir, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device,
                            num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    action_spec = solution.get_action_spec() if hasattr(solution, "get_action_spec") else None
    env_cfg = apply_safe_action_spec(env_cfg, json.dumps(action_spec))
    # never terminate so we roll the full length
    for _name in ("fall", "illegal_contact", "base_contact"):
        if hasattr(env_cfg.terminations, _name):
            setattr(env_cfg.terminations, _name, None)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    obs, _ = env.reset()
    robot = env.unwrapped.scene["robot"]
    head_cam = env.unwrapped.scene["head_camera"]
    try:
        lidar = env.unwrapped.scene["lidar_sensor"]
    except (KeyError, Exception):
        lidar = None
        print("[WARN] no lidar_sensor in scene; lidar cue skipped.", flush=True)

    rgb_buf, depth_buf, lidar_buf = [], [], []
    wz_buf, yaw_buf, xy_buf = [], [], []
    rec = 0

    for step in range(args_cli.steps):
        with torch.inference_mode():
            yaw = _world_yaw(robot)
            if args_cli.ytrack:
                # LANE-KEEPING teacher: hold a target cross-track y. The robot is let to
                # drift left/right (wandering target y) so the red strips show up on both
                # sides of the frame; the teacher steers to face back toward the center-
                # line, so wz labels = "turn back toward course center" — exactly the
                # reactive steering a vision student learns from the boundary. Straight
                # course is fine: the varied lateral offset teaches the asymmetric-
                # boundary response that also generalizes to curved courses.
                cur_y = float(robot.data.root_pos_w[0, 1])
                tgt_y = 0.0
                if args_cli.ywander_amp > 0.0:
                    tgt_y = args_cli.ywander_amp * math.sin(2 * math.pi * step / max(1, args_cli.ywander_period))
                tgt = max(-1.2, min(1.2, -args_cli.ky * (cur_y - tgt_y)))  # desired heading toward center
                err = math.atan2(math.sin(yaw - tgt), math.cos(yaw - tgt))
                wz = max(-args_cli.wz_max, min(args_cli.wz_max, -args_cli.kyaw * err))
            else:
                # kyaw teacher (privileged world yaw). Optionally WANDER the target yaw
                # left/right so the teacher produces a diverse range of wz labels (and the
                # red boundary appears on both sides), instead of always nudging one way.
                tgt = args_cli.target_yaw
                if args_cli.dyaw_amp > 0.0:
                    tgt = tgt + args_cli.dyaw_amp * math.sin(2 * math.pi * step / max(1, args_cli.dyaw_period))
                err = math.atan2(math.sin(yaw - tgt), math.cos(yaw - tgt))
                wz = max(-args_cli.wz_max, min(args_cli.wz_max, -args_cli.kyaw * err))
            solution.fixed_velocity_commands = torch.tensor(
                [_VX, 0.0, wz], device=solution.device, dtype=torch.float32).view(1, 3)

            if step % args_cli.every == 0:
                rgb = head_cam.data.output["rgb"][0].cpu().numpy()[:, :, :3].astype(np.uint8)
                depth = np.squeeze(head_cam.data.output["depth"][0].cpu().numpy()).astype(np.float32)
                rgb_buf.append(rgb)
                depth_buf.append(depth)
                if lidar is not None:
                    # Real 16-ch 360-deg lidar. Store range = distance from base to hit
                    # (finite, clipped to max), which is the sim2real-robust geometric cue.
                    hits = lidar.data.ray_hits_w[0]                 # (R,3) world
                    base = robot.data.root_pos_w[0]                 # (3,)
                    rng = torch.norm(hits - base.unsqueeze(0), dim=1)
                    rng = torch.where(torch.isfinite(rng), rng, torch.full_like(rng, 10.0))
                    rng = torch.clip(rng, 0.0, 10.0)
                    lidar_buf.append(rng.cpu().numpy().astype(np.float32))
                wz_buf.append(np.float32(wz))
                yaw_buf.append(np.float32(yaw))
                xy_buf.append(robot.data.root_pos_w[0, :2].cpu().numpy().astype(np.float32))

                if rec % args_cli.png_every == 0:
                    _save_png(rgb, os.path.join(png_dir, f"rgb_{rec:04d}_wz{wz:+.2f}.png"))
                    _save_png(depth, os.path.join(png_dir, f"depth_{rec:04d}_wz{wz:+.2f}.png"), is_depth=True)
                rec += 1

            resp = solution.predicts(obs, 0.0)
            act = (torch.zeros((1, 20), device="cuda") if (resp.get("giveup") or not resp["action"])
                   else torch.tensor(resp["action"], dtype=torch.float32, device="cuda").view(1, -1))
            obs, _, term, trunc, _ = env.step(act)

            if step % 300 == 0:
                _p = robot.data.root_pos_w[0]
                print(f"[step {step:4d}] x={float(_p[0]):+.1f} y={float(_p[1]):+.2f} "
                      f"yaw={math.degrees(yaw):+.0f}deg wz={wz:+.2f} recorded={rec}", flush=True)

    # save arrays
    out_npz = os.path.join(args_cli.out, "steering_data.npz")
    np.savez_compressed(
        out_npz,
        rgb=np.stack(rgb_buf) if rgb_buf else np.zeros((0,)),
        depth=np.stack(depth_buf) if depth_buf else np.zeros((0,)),
        lidar=np.stack(lidar_buf) if lidar_buf else np.zeros((0,)),
        wz=np.array(wz_buf, dtype=np.float32),
        yaw=np.array(yaw_buf, dtype=np.float32),
        xy=np.stack(xy_buf) if xy_buf else np.zeros((0,)),
    )
    print(f"[DONE] recorded {rec} frames -> {out_npz}", flush=True)
    print(f"  rgb shape {rgb_buf[0].shape if rgb_buf else None}, "
          f"depth {depth_buf[0].shape if depth_buf else None}, "
          f"lidar {lidar_buf[0].shape if lidar_buf else None}", flush=True)
    print(f"  wz range [{min(wz_buf):+.2f}, {max(wz_buf):+.2f}]  PNGs in {png_dir}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
