# Server-like validation for active-alignment replay.
# Unlike play_score_mem_final.py, this does NOT reproduce the demo start
# (no _reset_asset_root + oracle warmup). It does a PLAIN env.reset() so the robot
# starts in the default/landed pose -- mimicking the eval server's air-drop start --
# to verify that the solution's ACTIVE alignment (walk at zero vel) converges to the
# demo-start joint pose and that the subsequent open-loop replay scores.

import argparse
import os
import time
import json

import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Server-like play (plain reset, active align).")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=2000)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument("--debug", action="store_true", default=False)
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
from isaaclab_tasks.utils import parse_env_cfg
from atec_2026_tesk_e.ATEC2026_Simulation_Challenge.scripts.rl_utils import camera_follow
from atec_rl_lab.tasks.task_base.action_base import apply_safe_action_spec

import sys as _sys, os as _os
_SUB = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'submission_mem')
_sys.path.insert(0, _SUB)
import importlib as _il
_SOLMOD = _os.environ.get("MEM_SOLUTION_MODULE", "solution")  # default = active +16 track solution
AlgSolution = getattr(_il.import_module(_SOLMOD), "AlgSolution")
print(f"[play_score_mem] using solution module: {_SOLMOD}", flush=True)
solution = AlgSolution()


def play():
    if args_cli.task is None:
        raise ValueError("Please provide --task")

    is_task_e = isinstance(args_cli.task, str) and args_cli.task.startswith("ATEC-TaskE")
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric)
    action_spec = solution.get_action_spec() if hasattr(solution, "get_action_spec") else None
    env_cfg = apply_safe_action_spec(env_cfg, json.dumps(action_spec))
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.abspath(os.path.join("logs", "videos", args_cli.task, "server")),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video during server-like play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # PLAIN reset only -- server-like start. NO _reset_asset_root, NO oracle warmup.
    obs, _ = env.reset()

    _robot = env.unwrapped.scene["robot"]
    try:
        _box = env.unwrapped.scene["box"]
    except Exception:
        _box = None

    # Diagnostic toggles (env vars) to isolate which part of the demo start the
    # replay actually needs:  MEM_RESET_ROOT=1 -> _reset_asset_root(robot)+(box);
    # MEM_WARMUP=N -> run N oracle (zero-vel) warmup steps before handing to solution.
    _do_reset_root = os.environ.get("MEM_RESET_ROOT", "0") == "1"
    _warmup_n = int(os.environ.get("MEM_WARMUP", "0"))
    if _do_reset_root or _warmup_n > 0:
        import sys as _s, os as _o
        _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))), "scripts", "act"))
        from task_d.collector import _reset_asset_root
        if _do_reset_root:
            _reset_asset_root(_robot)
            if _box is not None:
                _reset_asset_root(_box)
            print("[server] applied _reset_asset_root(robot+box)", flush=True)
        if _warmup_n > 0:
            from task_d.walking_oracle import WalkingOracle
            _POL = "/hdd0/sohyun/atec_2026_tesk_e/ATEC2026_Simulation_Challenge/logs/rsl_rl/unitree_b2_rough/2026-06-18_04-37-10/exported/policy.pt"
            _oracle = WalkingOracle(_POL, _robot, "cuda")
            _oracle.reset()
            for _ in range(_warmup_n):
                obs, *_ = env.step(_oracle.compute_action())
            print(f"[server] applied {_warmup_n} oracle warmup steps", flush=True)

    if hasattr(solution, "reset"):
        solution.reset()

    jr_ref = getattr(solution, "jprel_start", None)
    n = jr_ref.shape[0] if jr_ref is not None else 0

    def _proprio_jprel(o):
        pr = o["proprio"]
        p = pr.detach().cpu().numpy().reshape(-1) if hasattr(pr, "detach") else np.asarray(pr).reshape(-1)
        return p[12:12 + n]

    rp0 = _robot.data.root_pos_w[0]
    msg = f"[server] PLAIN reset: robot=({rp0[0].item():+.2f},{rp0[1].item():+.2f},{rp0[2].item():+.2f})"
    if _box is not None:
        bp0 = _box.data.root_pos_w[0]
        msg += f" box=({bp0[0].item():+.2f},{bp0[1].item():+.2f},{bp0[2].item():+.2f})"
    if n:
        d0 = float(np.max(np.abs(_proprio_jprel(obs) - jr_ref)))
        msg += f" jprel_err={d0:.3f}"
    print(msg, flush=True)

    dt = env.unwrapped.step_dt if hasattr(env.unwrapped, "step_dt") else None
    timestep = 0
    total_episode_reward = 0.0
    total_elapsed_time = 0.0
    aligned_reported = False

    while simulation_app.is_running():
        with torch.inference_mode():
            start_time = time.time()
            resp = solution.predicts(obs, total_episode_reward)
            if resp["giveup"]:
                break
            actions = torch.tensor(resp["action"], dtype=torch.float32, device='cuda').view(1, -1)

            # Report the moment alignment completes (replay begins).
            if (not aligned_reported) and getattr(solution, "_aligned", False):
                aligned_reported = True
                err = float(np.max(np.abs(_proprio_jprel(obs) - jr_ref))) if n else -1
                rp = _robot.data.root_pos_w[0]
                print(f"[server] ALIGNED after {solution._align_steps} steps "
                      f"(jprel_err={err:.3f}); replay begins. "
                      f"robot=({rp[0].item():+.2f},{rp[1].item():+.2f},{rp[2].item():+.2f})", flush=True)

            obs, reward, terminated, truncated, info = env.step(actions)
            if not is_task_e:
                camera_follow(env)

            sim_dt = info["Step_dt"]
            r = reward.mean().item() if isinstance(reward, torch.Tensor) else float(reward)
            total_episode_reward += r / sim_dt

            if isinstance(info, dict) and "Elapsed_Time" in info:
                elapsed = info["Elapsed_Time"]
                total_elapsed_time = elapsed.item() if hasattr(elapsed, "item") else float(elapsed)
            elif dt is not None:
                total_elapsed_time += dt

            # fine-grained print in the cross region, coarse elsewhere
            _every = 10 if timestep >= 850 else 50
            if args_cli.debug and timestep % _every == 0:
                rp = _robot.data.root_pos_w[0]
                _bx = _box.data.root_pos_w[0] if _box is not None else None
                _bxs = f" box=({_bx[0].item():+.2f},{_bx[2].item():+.2f})" if _bx is not None else ""
                _ci = getattr(solution, "_cross_i", -1)
                print(f"  t={timestep} R={total_episode_reward:.1f}{_bxs} "
                      f"robot=({rp[0].item():+.2f},{rp[1].item():+.2f},{rp[2].item():+.2f}) "
                      f"cross_i={_ci}", flush=True)

            done = (terminated.item() or truncated.item())
            if done:
                print(f"[server] done at t={timestep}: terminated={terminated.item()} "
                      f"truncated={truncated.item()}", flush=True)
                break
            timestep += 1
            if args_cli.video and timestep >= args_cli.video_length:
                break
            if args_cli.real_time and dt is not None:
                sleep_time = dt - (time.time() - start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    rp = _robot.data.root_pos_w[0]
    print(f"[server] FINAL robot=({rp[0].item():+.2f},{rp[1].item():+.2f},{rp[2].item():+.2f})", flush=True)
    env.close()
    return total_episode_reward, total_elapsed_time


if __name__ == "__main__":
    score, elapsed_time = play()
    print(f"score: {score:.2f}, elapsed_time: {elapsed_time:.2f} seconds")
    print("Closing simulation app...")
    simulation_app.close()
