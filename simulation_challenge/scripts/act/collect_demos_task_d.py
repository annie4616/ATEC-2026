"""Scripted-oracle data collection for Task D (obstacle traversal, B2 + Piper).

The "expert" is the pretrained B2 flat-locomotion policy (demo/policy.pt) driven
with a fixed forward velocity command. We roll out episodes until the robot
crosses x = 3.5 (the env's success termination) and save those trajectories.

Usage
-----
python scripts/act/collect_demos_task_d.py --num_demos 50 --headless
python scripts/act/collect_demos_task_d.py --num_demos 10 --save_video --enable_cameras

Only the B2 robot (ATEC-TaskD-B2Piper / TaskDEnvB2Cfg) is supported, because the
shipped walking policy is trained for the B2's 12-DoF legs.
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect Task D demonstrations for ACT.")
parser.add_argument("--num_demos", type=int, default=50,
                    help="Number of successful demos to collect.")
parser.add_argument("--output_dir", type=str, default="./datasets/atec_task_d",
                    help="Directory to save trajectory.hdf5 and trajectory.json.")
parser.add_argument("--policy_path", type=str, default=None,
                    help="Path to the walking policy .pt. Defaults to demo/policy.pt.")
parser.add_argument("--save_video", action="store_true", default=False,
                    help="Save an MP4 per demo (requires imageio + a camera).")
parser.add_argument("--video_dir", type=str, default=None,
                    help="Output dir for MP4 files. Defaults to <output_dir>/videos/.")
parser.add_argument("--save_images", action="store_true", default=False,
                    help="Save raw RGB frames into HDF5 under traj_N/images/rgb.")
parser.add_argument("--max_attempts", type=int, default=0,
                    help="Abort after this many total attempts (0 = unlimited).")
parser.add_argument("--diag", action="store_true", default=False,
                    help="Print robot/box pose every 25 steps for diagnosis.")
parser.add_argument("--no_push", action="store_true", default=False,
                    help="Disable the box-into-pit push controller (walk straight only).")
parser.add_argument("--cross_runup", action="store_true", default=False,
                    help="Use PushIntoPitCrossController (v230 run-up cross) instead of the "
                         "plain PushIntoPitController.")
parser.add_argument("--record_attempt", action="store_true", default=False,
                    help="Record ONE attempt to video regardless of success (for inspection).")
parser.add_argument("--record_steps", type=int, default=1200,
                    help="Step budget when recording a single attempt.")
parser.add_argument("--third_person", action="store_true", default=False,
                    help="Record a third-person follow camera (whole scene) instead of "
                         "the robot's first-person head camera.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.record_attempt:
    args_cli.save_video = True
# The shared BaseSceneCfg always declares head_camera/ee_camera sensors, so the
# camera pipeline must be enabled for the env to initialise — even when we are
# not saving any frames. Force it on unconditionally.
args_cli.enable_cameras = True

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

import h5py
import json
import time
import numpy as np

from isaaclab.envs import ManagerBasedRLEnv

from atec_rl_lab.tasks.task_d.env_cfg import TaskDEnvB2Cfg

from task_d.walking_oracle import WalkingOracle
import task_d.collector as collector_mod
from task_d.collector import collect_one_demo, robot_reached_goal
from task_d.push_controller import PushIntoPitController
# Cross-variant controller (push box + run-up charge to cross the 0.4 m step). Same
# compute_velocity_command/reset/.state interface, selected with --cross_runup.
# Use the LOCKED 3-of-3 reproducible crossing controller (box050 / bail190 / v230):
# PUSH_X_BAIL_RX=-1.90, RUN_UP y-tol=0.18 — the config that crossed (x>3.5) 3/3 times.
from task_d.push_controller_cross import PushIntoPitCrossController

# The cross-runup controller tries to CROSS the pit, so a demo only counts when the
# robot actually reaches the finish line (x > 3.5), not merely when the box settles.
if args_cli.cross_runup:
    collector_mod.REQUIRE_CROSSING = True

# camera_follow lives in scripts/rl_utils.py (one level up from scripts/act/).
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_utils import camera_follow  # noqa: E402


def build_env(need_camera: bool, third_person: bool = False) -> ManagerBasedRLEnv:
    cfg = TaskDEnvB2Cfg()
    cfg.seed             = int(time.time_ns() % (2**31))
    cfg.scene.num_envs   = 1
    # episode_length_s default is large; keep env terminations but cap via collector.
    # rgb_array render mode is needed for a third-person viewport video.
    render_mode = "rgb_array" if third_person else None
    return ManagerBasedRLEnv(cfg, render_mode=render_mode)


def init_output(output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    traj_path = os.path.join(output_dir, "trajectory.hdf5")
    json_path = os.path.join(output_dir, "trajectory.json")
    with h5py.File(traj_path, "w"):
        pass
    with open(json_path, "w") as fh:
        json.dump({"env_info": {"env_kwargs": {"control_mode": "pd_joint_pos"}}}, fh)
    return traj_path


def save_traj(traj_path: str, traj_idx: int, data: dict, save_images: bool) -> None:
    with h5py.File(traj_path, "a") as f:
        grp = f.create_group(f"traj_{traj_idx}")
        grp.create_dataset("obs",       data=data["qpos"],      compression="gzip")
        # HIERARCHICAL ACT: the learned target is the 3-D velocity command (vx,vy,wz),
        # NOT the full 20-D joint action — a separate walking RL policy turns the
        # command into joint actions at deploy time. So `actions` = vel_cmd (3-D); the
        # full joint action is kept under `joint_action` for reference/debugging.
        grp.create_dataset("actions",      data=data["vel_cmd"],   compression="gzip")
        grp.create_dataset("joint_action", data=data["action"],    compression="gzip")
        grp.create_dataset("qvel",      data=data["qvel"],      compression="gzip")
        grp.create_dataset("root_pos",  data=data["root_pos"],  compression="gzip")
        grp.create_dataset("root_quat", data=data["root_quat"], compression="gzip")
        if save_images and "frames" in data:
            img_grp = grp.create_group("images")
            img_grp.create_dataset("rgb", data=data["frames"], compression="gzip")
            # Metric depth (T, H, W) float32 from the head camera, alongside RGB. The
            # ACT backbone consumes RGB+D (4-channel) so it can gauge distance to the
            # box/pit — the crossing run-up/charge timing depends on it.
            if "depth" in data:
                img_grp.create_dataset("depth", data=data["depth"], compression="gzip")


def main() -> None:
    need_camera = args_cli.save_video or args_cli.save_images

    env = build_env(need_camera, third_person=args_cli.third_person)
    dev = env.unwrapped.device

    # First-person head camera (default) vs third-person follow viewport.
    if args_cli.third_person:
        camera = None

        def frame_fn():
            camera_follow(env)            # aim the viewport behind the robot
            return env.render()           # rgb_array frame of the whole scene
    else:
        camera = env.unwrapped.scene["head_camera"] if need_camera else None
        frame_fn = None

    policy_path = args_cli.policy_path or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "demo", "policy.pt",
    )
    if not os.path.isfile(policy_path):
        raise FileNotFoundError(f"Walking policy not found: {policy_path}")
    robot  = env.unwrapped.scene["robot"]
    oracle = WalkingOracle(policy_path, robot, dev)
    if args_cli.no_push:
        push = None
    elif args_cli.cross_runup:
        # 3-of-3 LOCKED crossing config: v230 (cross + post-box sprint both at 2.30).
        push = PushIntoPitCrossController(v_cross=2.30, v_forward_cross=2.30)
    else:
        push = PushIntoPitController()

    video_dir = None
    imageio   = None
    if args_cli.save_video:
        video_dir = args_cli.video_dir or os.path.join(args_cli.output_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
        import imageio as _io
        imageio = _io

    # Single-attempt recording mode: roll out once, save the video regardless of
    # whether the box made it into the pit, then exit.
    if args_cli.record_attempt:
        print(f"\n[INFO] Recording one attempt ({args_cli.record_steps} steps)...")
        data = collect_one_demo(env, oracle, dev, camera=camera,
                                verbose=args_cli.diag, push=push,
                                save_failed=True, max_steps=args_cli.record_steps,
                                frame_fn=frame_fn)
        if data is None or "frames" not in data:
            print("[WARN] No frames captured.")
        else:
            vp = os.path.join(video_dir, "attempt.mp4")
            imageio.mimwrite(vp, data["frames"], fps=50, quality=7)
            print(f"[INFO] reached_goal={data.get('reached')}, "
                  f"final_box/robot logged; video -> {vp}")
        env.close()
        return

    traj_path = init_output(args_cli.output_dir)

    n_ok = 0
    attempt = 0
    while n_ok < args_cli.num_demos:
        attempt += 1
        if args_cli.max_attempts and attempt > args_cli.max_attempts:
            print(f"[WARN] Reached max_attempts={args_cli.max_attempts}; stopping.")
            break
        print(f"\n[INFO] Demo {n_ok + 1}/{args_cli.num_demos}  (attempt {attempt})")

        data = collect_one_demo(env, oracle, dev, camera=camera,
                                verbose=args_cli.diag, push=push, frame_fn=frame_fn)
        if data is None:
            print("[WARN] Did not reach goal (fell / early term) — skipping.")
            continue

        save_traj(traj_path, n_ok, data, args_cli.save_images)

        T = len(data["qpos"])
        notes = [f"{T} steps", f"final_x={data['root_pos'][-1, 0]:.2f}"]
        if args_cli.save_video and "frames" in data:
            vp = os.path.join(video_dir, f"demo_{n_ok:04d}.mp4")
            imageio.mimwrite(vp, data["frames"], fps=50, quality=7)
            notes.append(f"video -> {vp}")
        if args_cli.save_images and "frames" in data:
            notes.append("images saved")
        print(f"[INFO] traj_{n_ok}: {', '.join(notes)}")
        n_ok += 1

    print(f"\n[INFO] Collected {n_ok} demos -> {traj_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
