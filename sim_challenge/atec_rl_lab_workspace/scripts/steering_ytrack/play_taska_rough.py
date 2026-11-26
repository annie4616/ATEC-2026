"""Record the ROUGH walking policy (demo/policy.pt == unitree_b2_rough
2026-06-18_04-37-10) on the Task A off-road arena, using demo/solution_rl.py.

The rough policy is a plain velocity-command walker (fixed cmd [0.5,0,0] forward),
no scandots. We ignore the solution's giveup (Task A score climbs as it advances)
so the full clip records, and disable early termination.

  CUDA_VISIBLE_DEVICES=1 python scripts/play_taska_rough.py \
      --task ATEC-TaskA-B2Piper --headless --enable_cameras \
      --video --video_length 500 --no_terminate \
      --video_dir .../datasets/atec_task_a
"""
import argparse
import os
import time
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record rough walker on Task A.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=500)
parser.add_argument("--video_dir", type=str, default=None)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="ATEC-TaskA-B2Piper")
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument("--no_terminate", action="store_true", default=False)
parser.add_argument("--vx", type=float, default=None,
                    help="Override the solution's forward velocity command (default 0.5).")
parser.add_argument("--vy", type=float, default=0.0,
                    help="Lateral velocity command (+ = strafe left/+y) to counter drift.")
parser.add_argument("--kyaw", type=float, default=0.0,
                    help="Heading P-gain: wz = clip(-kyaw*yaw, -wz_max, wz_max). 0 disables (fixed wz).")
parser.add_argument("--wz_max", type=float, default=1.0,
                    help="Max |yaw-rate command| for the heading feedback.")
parser.add_argument("--target_yaw", type=float, default=0.0,
                    help="Desired world heading in radians (+x forward = 0).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402

import atec_rl_lab.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from atec_2026_tesk_e.ATEC2026_Simulation_Challenge.scripts.rl_utils import camera_follow  # noqa: E402
from atec_rl_lab.tasks.task_base.action_base import apply_safe_action_spec  # noqa: E402

from atec_2026_tesk_e.ATEC2026_Simulation_Challenge.demo.solution_rl import AlgSolution  # noqa: E402

solution = AlgSolution()

# Forward velocity used every step (default in solution is 0.5).
_VX = args_cli.vx if args_cli.vx is not None else 0.5
_VY = args_cli.vy
if (args_cli.vx is not None or _VY != 0.0) and hasattr(solution, "fixed_velocity_commands"):
    solution.fixed_velocity_commands = torch.tensor(
        [_VX, _VY, 0.0], device=solution.device, dtype=torch.float32).view(1, 3)
    print(f"[INFO] velocity command vx={_VX} vy={_VY}", flush=True)
if args_cli.kyaw != 0.0:
    print(f"[INFO] heading feedback ON: wz=clip(-{args_cli.kyaw}*(yaw-{args_cli.target_yaw}), "
          f"+-{args_cli.wz_max}); target_yaw={args_cli.target_yaw}", flush=True)


def _yaw_of(env) -> float:
    """World yaw of the robot base (rad), +x forward = 0."""
    import math as _m
    q = env.unwrapped.scene["robot"].data.root_quat_w[0]   # w,x,y,z
    return _m.atan2(2 * (float(q[0]) * float(q[3]) + float(q[1]) * float(q[2])),
                    1 - 2 * (float(q[2]) ** 2 + float(q[3]) ** 2))


def _update_heading_cmd(env):
    """P-control the yaw-rate command so the robot holds target_yaw (straight)."""
    if args_cli.kyaw == 0.0 or not hasattr(solution, "fixed_velocity_commands"):
        return
    import math as _m
    yaw = _yaw_of(env)
    err = yaw - args_cli.target_yaw
    err = _m.atan2(_m.sin(err), _m.cos(err))               # wrap to [-pi, pi]
    wz = max(-args_cli.wz_max, min(args_cli.wz_max, -args_cli.kyaw * err))
    solution.fixed_velocity_commands = torch.tensor(
        [_VX, _VY, wz], device=solution.device, dtype=torch.float32).view(1, 3)


def play() -> tuple[float, float]:
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device,
        num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric,
    )
    action_spec = solution.get_action_spec() if hasattr(solution, "get_action_spec") else None
    env_cfg = apply_safe_action_spec(env_cfg, json.dumps(action_spec))

    if args_cli.no_terminate:
        for _name in ("fall", "illegal_contact", "base_contact"):
            if hasattr(env_cfg.terminations, _name):
                setattr(env_cfg.terminations, _name, None)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_dir = args_cli.video_dir or os.path.abspath(
            os.path.join("logs", "videos", args_cli.task, "play"))
        os.makedirs(video_dir, exist_ok=True)
        video_kwargs = {
            "video_folder": video_dir,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "name_prefix": "taska_rough_b2",
        }
        print("[INFO] Recording video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    obs, _ = env.reset()
    dt = env.unwrapped.step_dt if hasattr(env.unwrapped, "step_dt") else None
    timestep = 0
    total_episode_reward = 0.0
    total_elapsed_time = 0.0

    while simulation_app.is_running():
        with torch.inference_mode():
            start_time = time.time()
            _update_heading_cmd(env)                        # steer wz to hold heading
            # Pass score=0 so solution_rl.py never hits its `current_score > 1` giveup
            # (which returns an empty action -> the driver zeroed all joints -> the robot
            #  collapsed in place on flat ground). For recording we always want to walk.
            resp = solution.predicts(obs, 0.0)
            if resp.get("giveup") or not resp["action"]:
                actions = torch.zeros((1, 20), dtype=torch.float32, device='cuda')
            else:
                actions = torch.tensor(resp["action"], dtype=torch.float32, device='cuda').view(1, -1)
            obs, reward, terminated, truncated, info = env.step(actions)
            camera_follow(env)

            if timestep % 50 == 0:
                _rob = env.unwrapped.scene["robot"]
                _p = _rob.data.root_pos_w[0]
                _q = _rob.data.root_quat_w[0]      # w,x,y,z
                import math as _m
                _yaw = _m.atan2(2*(float(_q[0])*float(_q[3]) + float(_q[1])*float(_q[2])),
                                1 - 2*(float(_q[2])**2 + float(_q[3])**2))
                _wz = float(solution.fixed_velocity_commands[0, 2]) \
                    if hasattr(solution, "fixed_velocity_commands") else 0.0
                print(f"[TRAJ t={timestep:3d}] x={float(_p[0]):+.2f} y={float(_p[1]):+.2f} "
                      f"z={float(_p[2]):+.2f} yaw={_yaw:+.2f}rad wz={_wz:+.2f}", flush=True)

            sim_dt = info["Step_dt"]
            r = reward.mean().item() if isinstance(reward, torch.Tensor) else float(reward)
            total_episode_reward += r / sim_dt
            if isinstance(info, dict) and "Elapsed_Time" in info:
                elapsed = info["Elapsed_Time"]
                total_elapsed_time = elapsed.item() if hasattr(elapsed, "item") else float(elapsed)
            elif dt is not None:
                total_elapsed_time += dt

            if not args_cli.no_terminate and (terminated.item() or truncated.item()):
                print(f"[INFO] episode done at step {timestep}.", flush=True)
                break
            timestep += 1
            if args_cli.video and timestep >= args_cli.video_length:
                break
            if args_cli.real_time and dt is not None:
                st = dt - (time.time() - start_time)
                if st > 0:
                    time.sleep(st)

    env.close()
    return total_episode_reward, total_elapsed_time


if __name__ == "__main__":
    score, elapsed_time = play()
    print(f"score: {score:.2f}, elapsed_time: {elapsed_time:.2f} seconds")
    print("Closing simulation app...")
    simulation_app.close()
