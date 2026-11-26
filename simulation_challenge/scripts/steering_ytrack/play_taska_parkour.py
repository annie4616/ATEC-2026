"""Play/record the B2Piper PARKOUR teacher on the Task A off-road arena.

Based on scripts/play_atec_task.py, but:
  * uses demo/solution_taska_parkour.py (parkour policy, no box/push),
  * every step raycasts the REAL Task A ground mesh at the teacher's 132 scan-grid
    points (transformed by the robot's true base pose) and injects the clipped
    height scan into the solution as `solution._live_scan`,
  * writes the recorded mp4 straight into the requested output dir.

Example:
  CUDA_VISIBLE_DEVICES=1 python scripts/play_taska_parkour.py \
      --task ATEC-TaskA-B2Piper --video --enable_cameras --video_length 500 \
      --video_dir /hdd0/sohyun/atec_2026_tesk_e/ATEC2026_Simulation_Challenge/datasets/atec_task_a
"""

import argparse
import os
import time
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record the parkour teacher on Task A.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video.")
parser.add_argument("--video_length", type=int, default=500, help="Video length (steps).")
parser.add_argument("--video_dir", type=str, default=None, help="Output dir for the mp4.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--task", type=str, default="ATEC-TaskA-B2Piper", help="Task name.")
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument("--no_terminate", action="store_true", default=False,
                    help="Disable fall/contact terminations so recording runs the full length.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --------------------------------------------------------------------------- imports
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab.utils.math import quat_apply, yaw_quat  # noqa: E402
from isaaclab.utils.warp import raycast_mesh  # noqa: E402

import atec_rl_lab.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from atec_2026_tesk_e.ATEC2026_Simulation_Challenge.scripts.rl_utils import camera_follow  # noqa: E402
from atec_rl_lab.tasks.task_base.action_base import apply_safe_action_spec  # noqa: E402

from atec_2026_tesk_e.ATEC2026_Simulation_Challenge.demo.solution_taska_parkour12 import AlgSolution  # noqa: E402

solution = AlgSolution()

SCAN_Z_OFFSET = 0.3


def _get_ground_warp_mesh(env):
    """Return the warp mesh of /World/ground, reusing the lidar RayCaster's cache.

    The RayCaster sensor (env's `lidar_sensor`, cast against /World/ground) builds and
    caches a warp mesh in the class-level dict `RayCaster.meshes[prim_path]` at init.
    We reuse that exact mesh so our scan raycasts hit the real Task A terrain.
    """
    from isaaclab.sensors.ray_caster.ray_caster import RayCaster
    meshes = getattr(RayCaster, "meshes", {}) or {}
    if "/World/ground" in meshes:
        return meshes["/World/ground"]
    # fall back to any single cached mesh
    if len(meshes) == 1:
        return next(iter(meshes.values()))
    if meshes:
        print(f"[WARN] multiple cached warp meshes {list(meshes)}; using first.", flush=True)
        return next(iter(meshes.values()))
    return None


def _compute_live_scan(env, ground_mesh):
    """Raycast the real terrain at the 132 teacher grid points; return clipped scan."""
    robot = env.unwrapped.scene["robot"]
    base_pos = robot.data.root_pos_w[0]          # (3,)  world
    base_quat = robot.data.root_quat_w[0]        # (4,)  world (w,x,y,z)
    yaw_q = yaw_quat(base_quat.unsqueeze(0))     # (1,4) yaw-only

    local = solution.scan_local                  # (132, 2) in base frame (x fwd, y left)
    n = local.shape[0]
    local3 = torch.zeros((n, 3), device=local.device, dtype=torch.float32)
    local3[:, 0] = local[:, 0]
    local3[:, 1] = local[:, 1]
    # rotate grid by base yaw, translate to world; cast rays straight down from high up.
    world_xy = quat_apply(yaw_q.repeat(n, 1), local3) + base_pos.unsqueeze(0)
    ray_starts = world_xy.clone()
    ray_starts[:, 2] = base_pos[2] + 5.0
    ray_dirs = torch.zeros_like(ray_starts)
    ray_dirs[:, 2] = -1.0

    # raycast_mesh views output as the input shape and needs >=2D input; pass (132,3).
    hits = raycast_mesh(ray_starts, ray_dirs, ground_mesh)[0]     # (132, 3)
    heights = hits[:, 2]
    # missed rays come back as +inf -> treat as flat ground (height 0).
    heights = torch.where(torch.isfinite(heights), heights, torch.zeros_like(heights))
    scan = torch.clip(base_pos[2] - heights - SCAN_Z_OFFSET, -1.0, 1.0)
    return scan


def play() -> tuple[float, float]:
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device,
        num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric,
    )

    action_spec = solution.get_action_spec() if hasattr(solution, "get_action_spec") else None
    env_cfg = apply_safe_action_spec(env_cfg, json.dumps(action_spec))

    # For a continuous recording, disable early termination (fall / illegal contact)
    # so the episode always runs the full video_length instead of ending on a stumble.
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
            "name_prefix": "taska_parkour_b2",
        }
        print("[INFO] Recording video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    obs, _ = env.reset()
    if hasattr(solution, "reset"):
        solution.reset()

    ground_mesh = _get_ground_warp_mesh(env)
    if ground_mesh is None:
        print("[WARN] no ground mesh -> solution will fall back to a FLAT scan.", flush=True)

    dt = env.unwrapped.step_dt if hasattr(env.unwrapped, "step_dt") else None
    timestep = 0
    total_episode_reward = 0.0
    total_elapsed_time = 0.0

    while simulation_app.is_running():
        with torch.inference_mode():
            start_time = time.time()

            # Inject the real Task A terrain scan for this step.
            if ground_mesh is not None:
                try:
                    solution._live_scan = _compute_live_scan(env, ground_mesh)
                except Exception as exc:
                    if timestep == 0:
                        print(f"[WARN] live scan failed ({exc}); using flat.", flush=True)
                    solution._live_scan = None

            resp = solution.predicts(obs, total_episode_reward)
            if resp["giveup"]:
                break
            actions = torch.tensor(resp["action"], dtype=torch.float32, device='cuda').view(1, -1)
            obs, reward, terminated, truncated, info = env.step(actions)
            camera_follow(env)

            if timestep % 50 == 0:
                _rob = env.unwrapped.scene["robot"]
                _p = _rob.data.root_pos_w[0]
                _q = _rob.data.root_quat_w[0]
                import math as _m
                _yaw = _m.atan2(2*(float(_q[0])*float(_q[3]) + float(_q[1])*float(_q[2])),
                                1 - 2*(float(_q[2])**2 + float(_q[3])**2))
                print(f"[TRAJ t={timestep:4d}] x={float(_p[0]):+.2f} y={float(_p[1]):+.2f} "
                      f"z={float(_p[2]):+.2f} yaw={_yaw:+.2f}rad", flush=True)

            sim_dt = info["Step_dt"]
            r = reward.mean().item() if isinstance(reward, torch.Tensor) else float(reward)
            total_episode_reward += r / sim_dt

            if isinstance(info, dict) and "Elapsed_Time" in info:
                elapsed = info["Elapsed_Time"]
                total_elapsed_time = elapsed.item() if hasattr(elapsed, "item") else float(elapsed)
            elif dt is not None:
                total_elapsed_time += dt

            done = (terminated.item() or truncated.item())
            if done and not args_cli.no_terminate:
                print(f"[INFO] episode done at step {timestep}.", flush=True)
                break

            timestep += 1
            if args_cli.video and timestep >= args_cli.video_length:
                break
            if args_cli.real_time and dt is not None:
                sleep_time = dt - (time.time() - start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    env.close()
    return total_episode_reward, total_elapsed_time


if __name__ == "__main__":
    score, elapsed_time = play()
    print(f"score: {score:.2f}, elapsed_time: {elapsed_time:.2f} seconds")
    print("Closing simulation app...")
    simulation_app.close()
