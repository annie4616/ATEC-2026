"""VALIDATE full-horizon dead-reckoned POSITION vs true position.

The heading-validation showed gyro-integrated heading is accurate (2.2deg RMSE over 1500
steps) -- my earlier "heading drifts" claim was wrong (bad ground truth). So dead-reckoning
should give an accurate absolute position if we integrate over a LONG horizon (our odom head
failed because reset-every-48-steps made the displacement tiny, not because integration is
inaccurate). This measures integrated (x,y) vs true (x,y) with NO periodic reset (reset only
on episode boundary), reporting position error over the whole course.

  CUDA_VISIBLE_DEVICES=6 TMPDIR=/hdd0/sohyun/tmp PYTHONPATH=/hdd0/sohyun \
    python scripts/validate_odom_position.py --headless --steps 4000 \
      --out /hdd0/sohyun/atec_2026_tesk_e/taska_exp/odom_pos_val
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
parser.add_argument("--steps", type=int, default=4000)
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


def world_yaw(robot):
    q = robot.data.root_quat_w[0]
    return math.atan2(2 * (float(q[0]) * float(q[3]) + float(q[1]) * float(q[2])),
                      1 - 2 * (float(q[2]) ** 2 + float(q[3]) ** 2))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


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
    dt = float(env.unwrapped.step_dt)

    # dead-reckoning state, anchored to the TRUE pose at t=0 (known start)
    p0 = robot.data.root_pos_w[0]
    est_x, est_y = float(p0[0]), float(p0[1])
    yaw = world_yaw(robot)                       # gyro-integrated heading
    rows = []

    for step in range(args_cli.steps):
        with torch.inference_mode():
            gt = world_yaw(robot)
            yaw_rate = float(robot.data.root_ang_vel_b[0, 2])
            vb = robot.data.root_lin_vel_b[0]
            vx_b, vy_b = float(vb[0]), float(vb[1])
            tx = float(robot.data.root_pos_w[0, 0]); ty = float(robot.data.root_pos_w[0, 1])

            # dead reckon: integrate heading (gyro), rotate body vel to world, integrate xy
            yaw = wrap(yaw + yaw_rate * dt)
            c, s = math.cos(yaw), math.sin(yaw)
            est_x += (vx_b * c - vy_b * s) * dt
            est_y += (vx_b * s + vy_b * c) * dt

            rows.append((step, tx, ty, est_x, est_y, gt, yaw))

            tgt_y = args_cli.ywander_amp * math.sin(2 * math.pi * step / max(1, args_cli.ywander_period))
            cross = ty - tgt_y
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
                print(f"[step {step:4d}] true=({tx:+.1f},{ty:+.2f}) est=({est_x:+.1f},{est_y:+.2f}) "
                      f"errY={est_y-ty:+.2f} errX={est_x-tx:+.2f}", flush=True)
            if bool(term[0]):
                obs, _ = env.reset()
                p0 = robot.data.root_pos_w[0]
                est_x, est_y = float(p0[0]), float(p0[1]); yaw = world_yaw(robot)

    arr = np.array(rows, dtype=np.float32)
    np.savez_compressed(os.path.join(args_cli.out, "pos_val.npz"), rows=arr)
    ex = arr[:, 3] - arr[:, 1]; ey = arr[:, 4] - arr[:, 2]
    print("\n=== DEAD-RECKONED POSITION ERROR vs true (whole run, no periodic reset) ===", flush=True)
    print(f"  X error: RMSE={np.sqrt((ex**2).mean()):.3f}m  MAE={np.abs(ex).mean():.3f}m  max={np.abs(ex).max():.3f}m", flush=True)
    print(f"  Y error: RMSE={np.sqrt((ey**2).mean()):.3f}m  MAE={np.abs(ey).mean():.3f}m  max={np.abs(ey).max():.3f}m", flush=True)
    print(f"  (true Y range [{arr[:,2].min():+.2f},{arr[:,2].max():+.2f}], so cross-track drift signal magnitude ~{arr[:,2].std():.2f}m)", flush=True)
    print(f"  corr(est_y, true_y) = {np.corrcoef(arr[:,4], arr[:,2])[0,1]:+.3f}", flush=True)
    print(f"  saved -> {os.path.join(args_cli.out, 'pos_val.npz')}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
