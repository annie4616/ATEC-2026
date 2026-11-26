"""Evaluate a trained Task-D box-push ACT policy in the eval env.

Loads a checkpoint produced by train_task_d.py (norm_stats + agent/ema_agent), runs
the ACT policy in TaskDEnvB2Cfg, and reports whether the box gets pushed into the pit
(box x in [-1.4, 0.7] AND z < -0.5 — same success test as collect_demos_task_d.py's
robot_reached_goal). The policy sees exactly what training recorded: state = absolute
joint_pos (20), rgb = the third-person env.render() frame resized to 224x224.

Run:
  CUDA_VISIBLE_DEVICES=N TMPDIR=/hdd0/sohyun/tmp PYTHONPATH=/hdd0/sohyun \
    python scripts/act/eval_act_task_d.py \
      --ckpt runs/taskd_box_act/checkpoints/best_loss.pt \
      --episodes 10 --max_steps 1000 --save_video --headless --enable_cameras
"""

import argparse
import os
import sys
import types

# Same isaaclab-free shim as train_task_d.py so the ACT module imports don't drag in
# the locomotion/pxr chain (see train_task_d.py for the rationale).
_SRC_PARENT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "source", "atec_rl_lab")
_PKG = os.path.join(_SRC_PARENT, "atec_rl_lab")
if _SRC_PARENT not in sys.path:
    sys.path.insert(0, _SRC_PARENT)
for _name, _dir in (("atec_rl_lab", _PKG),
                    ("atec_rl_lab.train", os.path.join(_PKG, "train"))):
    if _name not in sys.modules:
        _m = types.ModuleType(_name)
        _m.__path__ = [_dir]
        sys.modules[_name] = _m

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default="runs/taskd_box_act/checkpoints/best_loss.pt")
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--max_steps", type=int, default=1000)
parser.add_argument("--use_ema", action="store_true", default=True)
parser.add_argument("--save_video", action="store_true", default=False)
parser.add_argument("--output_dir", type=str, default="./datasets/atec_task_d_act_eval")
# HIERARCHICAL: ACT outputs a 3-D velocity command (vx,vy,wz); this walking RL policy
# turns it into joint actions (same WalkingOracle used during demo collection).
parser.add_argument("--policy_path", type=str,
                    default="logs/rsl_rl/unitree_b2_rough/2026-06-18_04-37-10/exported/policy.pt")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import torchvision.transforms.functional as TF
import imageio

from isaaclab.envs import ManagerBasedRLEnv
from atec_rl_lab.tasks.task_d.env_cfg import TaskDEnvB2Cfg

# ACT model (resolved via the shim above)
from train_task_d import Agent, Args  # noqa: E402  same-dir import

# Walking RL policy wrapper (image->vel_cmd hierarchy). scripts/act is on sys.path so
# `task_d.walking_oracle` resolves the same way the collector imports it.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from task_d.walking_oracle import WalkingOracle  # noqa: E402
from task_d.collector import _reset_asset_root  # noqa: E402  restore box/robot root each ep


# Box-into-pit success test (matches collector.robot_reached_goal).
def box_in_pit(box) -> bool:
    bx = box.data.root_pos_w[0, 0].item()
    bz = box.data.root_pos_w[0, 2].item()
    # eval env origin x = -4.2, so world target band [-1.4,0.7] is checked directly
    # on world x here (the collector uses the same absolute-x band).
    return (-1.4 <= bx <= 0.7) and (bz < -0.5)


# Crossing success test: the robot actually traverses the pit and reaches the finish
# line (root x > 3.5) — the real Task D success, matching the 3-of-3 demo criterion.
CROSS_X = 3.5
def robot_crossed(robot) -> bool:
    return robot.data.root_pos_w[0, 0].item() > CROSS_X


def main():
    os.makedirs(args_cli.output_dir, exist_ok=True)
    device = "cuda"

    # ---- load checkpoint ----
    ckpt = torch.load(args_cli.ckpt, map_location=device)
    norm = ckpt["norm_stats"]
    state_dim = norm["state_mean"].shape[-1]
    act_dim = norm["action_mean"].shape[-1]
    weight_key = "ema_agent" if (args_cli.use_ema and "ema_agent" in ckpt) else "agent"

    targs = Args()
    targs.num_queries = 30
    targs.include_rgb = any("backbone" in k for k in ckpt[weight_key].keys())
    # Infer RGB vs RGB+D from the backbone's first conv: 3 in-channels = RGB,
    # 4 = RGB+D. Must be set BEFORE building Agent so conv1 is sized to match.
    targs.include_depth = False
    for k, v in ckpt[weight_key].items():
        if k.endswith("conv1.weight") and hasattr(v, "shape") and len(v.shape) == 4:
            targs.include_depth = (v.shape[1] == 4)
            break
    agent = Agent(state_dim, act_dim, targs).to(device)
    agent.load_state_dict(ckpt[weight_key])
    agent.eval()

    s_mean = norm["state_mean"].to(device)
    s_std = norm["state_std"].to(device)
    a_mean = norm["action_mean"].to(device)
    a_std = norm["action_std"].to(device)
    num_queries = targs.num_queries
    print(f">>> loaded {args_cli.ckpt} | state_dim={state_dim} act_dim={act_dim} "
          f"include_rgb={targs.include_rgb} include_depth={targs.include_depth} "
          f"weights={weight_key}", flush=True)

    # ---- env ----
    cfg = TaskDEnvB2Cfg()
    cfg.scene.num_envs = 1
    env = ManagerBasedRLEnv(cfg, render_mode="rgb_array")
    robot = env.unwrapped.scene["robot"]
    box = env.unwrapped.scene["box"]
    # Feed the ACT policy the SAME camera training recorded: the robot head_camera
    # (collect_demos_task_d.py default). The earlier eval wrongly used env.render()
    # (third-person) -> train/eval mismatch. Keep env.render() only for the human video.
    head_camera = env.unwrapped.scene["head_camera"]
    # Walking RL policy: turns the ACT-predicted velocity command into joint actions.
    oracle = WalkingOracle(args_cli.policy_path, robot, device)

    successes = 0
    for ep in range(args_cli.episodes):
        env.reset()
        # env.reset() only resets robot JOINTS, not the box or robot ROOT pose — without
        # this the box stays wherever the previous episode left it (in the pit -> instant
        # false successes). Restore both to their start poses, like the collector does.
        _reset_asset_root(robot)
        _reset_asset_root(box)
        oracle.reset()
        # let the robot settle on the ground after the teleport before acting
        for _ in range(10):
            env.step(oracle.compute_action())
        # temporal-ensemble buffer: all-step x all-query predictions
        all_time_actions = torch.zeros(
            (args_cli.max_steps, args_cli.max_steps + num_queries, act_dim), device=device)
        frames = []
        ok = False
        for t in range(args_cli.max_steps):
            # state = absolute joint_pos (20), normalized
            qpos = robot.data.joint_pos[0:1].to(device).float()       # (1, 20)
            state = (qpos - s_mean) / s_std
            obs = {"state": state}

            if targs.include_rgb:
                # Policy input: robot head_camera RGB (same as training).
                hc = head_camera.data.output["rgb"][0, :, :, :3]      # (H, W, 3)
                rgb = hc.to(device).permute(2, 0, 1).unsqueeze(0)     # (1,3,H,W)
                if rgb.dtype != torch.uint8:
                    rgb = (rgb.float()).clamp(0, 255).to(torch.uint8) \
                        if rgb.max() > 1.5 else (rgb.float() * 255).clamp(0, 255).to(torch.uint8)
                rgb = TF.resize(rgb, [224, 224], antialias=True)
                obs["rgb"] = rgb.unsqueeze(1)                         # (1,1,3,224,224)
                # Depth channel (must match the training normalization in
                # DemoDataset_ACT: nan/inf -> far clip, clamp [0,50] m, /50 -> [0,1]).
                if getattr(targs, "include_depth", False):
                    hd = head_camera.data.output["depth"][0]         # (H, W) or (H,W,1)
                    hd = hd.to(device).float().squeeze(-1) if hd.dim() == 3 else hd.to(device).float()
                    hd = torch.nan_to_num(hd, nan=50.0, posinf=50.0, neginf=0.0)
                    hd = hd.clamp(0.0, 50.0) / 50.0
                    depth = TF.resize(hd.unsqueeze(0), [224, 224], antialias=True)  # (1,224,224)
                    obs["depth"] = depth.unsqueeze(0).unsqueeze(0)    # (1,1,1,224,224)
                # Human video: third-person render (not fed to the policy).
                if args_cli.save_video:
                    fr = env.render()
                    if fr is not None:
                        frames.append(fr[:, :, :3])

            with torch.no_grad():
                a_seq = agent.get_action(obs)                         # (1, num_queries, act_dim)
            # temporal ensembling (exponential weighting over overlapping predictions)
            all_time_actions[t, t:t + num_queries] = a_seq[0]
            pop = all_time_actions[:t + 1, t]
            pop = pop[(pop != 0).any(dim=1)]
            if pop.numel() == 0:
                a_norm = a_seq[0, 0]
            else:
                w = torch.exp(-0.01 * torch.arange(len(pop), device=device))
                w = (w / w.sum()).unsqueeze(1)
                a_norm = (pop * w).sum(dim=0)
            vel_cmd = (a_norm * a_std[0] + a_mean[0])                 # denormalize -> (3,) (vx,vy,wz)

            # HIERARCHICAL: ACT-predicted velocity command -> walking RL -> joint action.
            oracle.set_velocity_command(float(vel_cmd[0]), float(vel_cmd[1]), float(vel_cmd[2]))
            action = oracle.compute_action()                          # (1, 20)

            env.step(action)
            # Success = the robot CROSSES the pit (root x > 3.5). Do NOT stop when the
            # box merely settles — keep stepping so the actual crossing is captured on
            # video, ending only once the robot reaches the finish line.
            if robot_crossed(robot):
                ok = True
                break

        successes += int(ok)
        bx = box.data.root_pos_w[0, 0].item(); bz = box.data.root_pos_w[0, 2].item()
        rx = robot.data.root_pos_w[0, 0].item()
        pit = box_in_pit(box)
        print(f"[ep {ep:2d}] crossed={ok}  robot_x={rx:+.2f}  box=({bx:+.2f},{bz:+.2f})  "
              f"box_in_pit={pit}  steps={t+1}", flush=True)
        if args_cli.save_video and frames:
            vp = os.path.join(args_cli.output_dir, f"act_eval_ep{ep}.mp4")
            imageio.mimwrite(vp, frames, fps=50, quality=7)
            print(f"        video -> {vp}", flush=True)

    print(f"\n[RESULT] {successes}/{args_cli.episodes} pit CROSSINGS (robot x > 3.5)", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
