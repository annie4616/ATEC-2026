"""Record the A2 Pro + Piper X LiDAR parkour student on the Task A (Hiking) course.

Unlike play_taska_parkour.py (which raycasts a synthetic 132-scandot grid for the old
12-DoF teacher), the lidar student consumes the env's real 5760-ray extero.lidar_scan
directly, so NO live-scan injection is needed. The arm is held stowed behind the back by
the solution (arm action ignored).

Run (GPU 2, MPS bypass required on this box):
  CUDA_VISIBLE_DEVICES=2 CUDA_MPS_PIPE_DIRECTORY=/tmp/nonexistent_x \
  TMPDIR=/hdd0/sohyun/tmp \
  python scripts/steering_ytrack/play_taska_lidar_student.py \
    --task ATEC-TaskA-A2Piper --video --enable_cameras --video_length 1500 \
    --video_dir /hdd0/sohyun/atec_2026_tesk_e/task_a_real_exp/taska_lidar_student
"""

import argparse
import os
import time
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record the lidar parkour student on Task A.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video.")
parser.add_argument("--video_length", type=int, default=1500, help="Video length (steps).")
parser.add_argument("--video_dir", type=str, default=None, help="Output dir for the mp4.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--task", type=str, default="ATEC-TaskA-A2Piper", help="Task name.")
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument("--no_terminate", action="store_true", default=False,
                    help="Disable fall/contact terminations so recording runs the full length.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# We disable the head/ee cameras in TaskAEnvA2Cfg (the lidar student never reads them), which
# leaves the sim with no RTX sensors — so AppLauncher's --enable_cameras does NOT flip the sim
# into a rendering mode, and RecordVideo's env.render() would raise "NO_GUI_OR_RENDERING".
# For video runs, force offscreen rendering on so the viewport can be captured.
if args_cli.video:
    import carb  # noqa: E402
    carb.settings.get_settings().set("/isaaclab/render/offscreen", True)

# --------------------------------------------------------------------------- imports
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402

import sys  # noqa: E402
# Resolve sibling packages by absolute filesystem path (robust to how PYTHONPATH is set):
# scripts/common/rl_utils.py holds camera_follow; demo/ holds the solution.
_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../scripts
_REPO_DIR = os.path.dirname(_SCRIPTS_DIR)                                    # .../ATEC2026_Simulation_Challenge
for _p in (os.path.join(_SCRIPTS_DIR, "common"), os.path.join(_REPO_DIR, "demo")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import atec_rl_lab.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rl_utils import camera_follow  # noqa: E402  (scripts/common/rl_utils.py)
from atec_rl_lab.tasks.task_base.action_base import apply_safe_action_spec  # noqa: E402

from solution_taska_lidar_student import AlgSolution  # noqa: E402  (demo/)

solution = AlgSolution()


def _dump_obs_shapes(obs):
    """One-time diagnostic: the adapter assumes obs['proprio'] (72) + obs['extero']['lidar_scan']
    (5760). Print the actual obs structure so a mismatch is obvious in the log."""
    try:
        if isinstance(obs, dict):
            for k, v in obs.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        print(f"[OBS] {k}.{kk}: shape={tuple(vv.shape)}", flush=True)
                elif hasattr(v, "shape"):
                    print(f"[OBS] {k}: shape={tuple(v.shape)}", flush=True)
                else:
                    print(f"[OBS] {k}: {type(v)}", flush=True)
    except Exception as exc:
        print(f"[OBS] dump failed: {exc}", flush=True)


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
            "name_prefix": "taska_lidar_student",
        }
        print("[INFO] Recording video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    obs, _ = env.reset()
    if hasattr(solution, "reset"):
        solution.reset()
    _dump_obs_shapes(obs)

    dt = env.unwrapped.step_dt if hasattr(env.unwrapped, "step_dt") else None
    timestep = 0
    total_episode_reward = 0.0
    total_elapsed_time = 0.0

    while simulation_app.is_running():
        with torch.inference_mode():
            start_time = time.time()

            # Inject the robot's TRUE world pose (roll/pitch/yaw + xy) so the adapter builds a
            # stable look-ahead delta_yaw (steer toward a point on the spawn line) instead of the
            # drifting -ryaw integral, and reads euler roll/pitch instead of the gravity approx.
            try:
                _rb = env.unwrapped.scene["robot"]
                _rq = _rb.data.root_quat_w[0]
                from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi
                _r, _p2, _y2 = euler_xyz_from_quat(_rq.unsqueeze(0))
                solution._live_roll = float(wrap_to_pi(_r)[0])
                solution._live_pitch = float(wrap_to_pi(_p2)[0])
                solution._live_yaw = float(wrap_to_pi(_y2)[0])
                _pw = _rb.data.root_pos_w[0]
                solution._live_pos = (float(_pw[0]), float(_pw[1]))
            except Exception:
                solution._live_roll = solution._live_pitch = None
                solution._live_yaw = solution._live_pos = None

            resp = solution.predicts(obs, total_episode_reward)
            if resp["giveup"]:
                break
            actions = torch.tensor(resp["action"], dtype=torch.float32, device='cuda').view(1, -1)
            obs, reward, terminated, truncated, info = env.step(actions)
            # camera_follow needs the viewport camera controller, which only exists when
            # rendering (video). In a headless verify run it's None, so guard the call.
            if args_cli.video:
                try:
                    camera_follow(env)
                except Exception:
                    pass

            if timestep % 50 == 0:
                _rob = env.unwrapped.scene["robot"]
                _p = _rob.data.root_pos_w[0]
                _q = _rob.data.root_quat_w[0]
                import math as _m
                _yaw = _m.atan2(2*(float(_q[0])*float(_q[3]) + float(_q[1])*float(_q[2])),
                                1 - 2*(float(_q[2])**2 + float(_q[3])**2))
                print(f"[TRAJ t={timestep:4d}] x={float(_p[0]):+.2f} y={float(_p[1]):+.2f} "
                      f"z={float(_p[2]):+.2f} yaw={_yaw:+.2f}rad", flush=True)

            sim_dt = info.get("Step_dt", dt or 0.02)
            r = reward.mean().item() if isinstance(reward, torch.Tensor) else float(reward)
            total_episode_reward += r / sim_dt if sim_dt else r

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

    print(f"[DONE] steps={timestep} final_x reached; total_reward={total_episode_reward:.1f} "
          f"elapsed={total_elapsed_time:.1f}s", flush=True)
    env.close()
    return total_episode_reward, total_elapsed_time


if __name__ == "__main__":
    play()
    simulation_app.close()
