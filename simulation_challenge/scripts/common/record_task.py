# Record a short video of an ATEC task running with zero actions (env-only sanity check).
import argparse
import os
import itertools
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record ATEC Task B/D video (zero action).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--task", type=str, required=True, choices=["B", "D"], help="Which task to record.")
parser.add_argument("--video_length", type=int, default=300, help="Number of steps to record.")
parser.add_argument("--video_folder", type=str, default=None, help="Output folder for the video.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Video recording requires cameras enabled.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import gymnasium as gym

if args_cli.task == "B":
    from atec_rl_lab.tasks.task_b import TaskBEnvB2Cfg as EnvCfg
else:
    from atec_rl_lab.tasks.task_d import TaskDEnvB2Cfg as EnvCfg


def main():
    env_cfg = EnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    from isaaclab.envs import ManagerBasedRLEnv

    env = ManagerBasedRLEnv(env_cfg, render_mode="rgb_array")

    for name, articulation in env.scene.articulations.items():
        print("-" * 100)
        print("Robot name:", name)
        print("Bodies:", articulation.num_bodies, "->", articulation.body_names)
        print("Joints:", articulation.num_joints, "->", articulation.joint_names)
        articulation.set_joint_position_target(articulation.data.default_joint_pos)

    video_folder = args_cli.video_folder or os.path.abspath(
        os.path.join("logs", "videos", f"Task{args_cli.task}", "record")
    )
    os.makedirs(video_folder, exist_ok=True)
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_folder,
        step_trigger=lambda step: step == 0,
        video_length=args_cli.video_length,
        disable_logger=True,
    )
    print(f"[INFO] Recording to: {video_folder}")

    action_space = env.action_space
    obs, info = env.reset()

    for i in itertools.count():
        if not simulation_app.is_running():
            break
        action = torch.zeros(action_space.shape, device=env.unwrapped.device)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated
        if done.any():
            env_ids = done.nonzero(as_tuple=False).squeeze(-1)
            env.reset(env_ids=env_ids)
        if i >= args_cli.video_length:
            break

    env.close()
    print(f"[INFO] Done. Video saved under: {video_folder}")


if __name__ == "__main__":
    main()
    simulation_app.close()
