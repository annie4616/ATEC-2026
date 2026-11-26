"""Collect Task A straight-walking demonstrations for imitation learning.

Teacher = pretrained B2 walking policy (demo/policy.pt) + WORLD-YAW heading feedback
(privileged). We record student-visible obs (head RGB + depth + proprio) and the
teacher's 3-D velocity command (vx, vy, wz) as the ACT target. A student trained on
this learns to hold heading from obs alone (no world yaw at eval).

Usage
-----
python scripts/act/collect_demos_task_a.py --num_demos 5  --headless          # smoke
python scripts/act/collect_demos_task_a.py --num_demos 100 --headless --save_images

Saves datasets/atec_task_a_il/trajectory.hdf5 with the SAME schema as Task D
(obs / actions=vel_cmd / joint_action / images/rgb / images/depth), so train_task_a.py
(a thin copy of train_task_d.py) can consume it.
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect Task A straight-walk demos.")
parser.add_argument("--num_demos", type=int, default=50)
parser.add_argument("--output_dir", type=str, default="./datasets/atec_task_a_il")
parser.add_argument("--policy_path", type=str, default=None)
parser.add_argument("--save_images", action="store_true", default=False)
parser.add_argument("--n_steps", type=int, default=500)
parser.add_argument("--vx", type=float, default=2.0)
parser.add_argument("--kyaw", type=float, default=2.0)
parser.add_argument("--wz_max", type=float, default=1.0)
parser.add_argument("--yaw_jitter", type=float, default=0.3,
                    help="Random initial-yaw perturbation per episode (rad).")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_attempts", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True                       # head_camera sensor must init

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import h5py
import json
import numpy as np

from isaaclab.envs import ManagerBasedRLEnv
from atec_rl_lab.tasks.task_a.env_cfg import TaskAEnvB2Cfg

# task_a collector + reused task_d WalkingOracle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))       # scripts/act
from task_d.walking_oracle import WalkingOracle
from task_a.collector import collect_one_demo


def build_env() -> ManagerBasedRLEnv:
    cfg = TaskAEnvB2Cfg()
    cfg.scene.num_envs = 1
    return ManagerBasedRLEnv(cfg)


def init_output(output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    traj_path = os.path.join(output_dir, "trajectory.hdf5")
    with h5py.File(traj_path, "w"):
        pass
    return traj_path


def save_traj(traj_path: str, idx: int, data: dict, save_images: bool) -> None:
    with h5py.File(traj_path, "a") as f:
        grp = f.create_group(f"traj_{idx}")
        grp.create_dataset("obs",          data=data["qpos"],      compression="gzip")
        grp.create_dataset("actions",      data=data["vel_cmd"],   compression="gzip")
        grp.create_dataset("joint_action", data=data["action"],    compression="gzip")
        grp.create_dataset("qvel",         data=data["qvel"],      compression="gzip")
        grp.create_dataset("root_pos",     data=data["root_pos"],  compression="gzip")
        grp.create_dataset("root_quat",    data=data["root_quat"], compression="gzip")
        if save_images and "frames" in data:
            ig = grp.create_group("images")
            ig.create_dataset("rgb", data=data["frames"], compression="gzip")
            if "depth" in data:
                ig.create_dataset("depth", data=data["depth"], compression="gzip")


def main() -> None:
    dev = args_cli.device
    env = build_env()
    camera = env.unwrapped.scene["head_camera"] if args_cli.save_images else None
    depth_camera = camera                                    # same sensor has depth

    policy_path = args_cli.policy_path or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "demo", "policy.pt")
    if not os.path.isfile(policy_path):
        raise FileNotFoundError(policy_path)
    robot = env.unwrapped.scene["robot"]
    oracle = WalkingOracle(policy_path, robot, dev)

    traj_path = init_output(args_cli.output_dir)
    rng = np.random.default_rng(args_cli.seed)

    saved, attempts = 0, 0
    while saved < args_cli.num_demos:
        attempts += 1
        if args_cli.max_attempts and attempts > args_cli.max_attempts:
            print(f"[WARN] hit max_attempts={args_cli.max_attempts}, stopping.", flush=True)
            break
        data = collect_one_demo(
            env, oracle, dev,
            camera=camera, depth_camera=depth_camera,
            vx=args_cli.vx, kyaw=args_cli.kyaw, wz_max=args_cli.wz_max,
            init_yaw_jitter=args_cli.yaw_jitter, n_steps=args_cli.n_steps,
            save_images=args_cli.save_images, rng=rng, verbose=True,
        )
        if data is None:
            print(f"[attempt {attempts}] dropped (fallen).", flush=True)
            continue
        save_traj(traj_path, saved, data, args_cli.save_images)
        yfin = data["root_quat"][-1]
        import math
        yaw = math.atan2(2*(yfin[0]*yfin[3]+yfin[1]*yfin[2]),
                         1-2*(yfin[2]**2+yfin[3]**2))
        print(f"[demo {saved+1}/{args_cli.num_demos}] T={len(data['qpos'])} "
              f"end x={data['root_pos'][-1][0]:+.1f} y={data['root_pos'][-1][1]:+.2f} "
              f"yaw={yaw:+.2f}", flush=True)
        saved += 1

    with open(os.path.join(args_cli.output_dir, "trajectory.json"), "w") as f:
        json.dump({"num_demos": saved, "attempts": attempts,
                   "vx": args_cli.vx, "kyaw": args_cli.kyaw,
                   "n_steps": args_cli.n_steps, "yaw_jitter": args_cli.yaw_jitter}, f, indent=2)
    print(f"[DONE] saved {saved} demos ({attempts} attempts) -> {traj_path}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
