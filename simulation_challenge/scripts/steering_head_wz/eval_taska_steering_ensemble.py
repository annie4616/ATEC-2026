"""TASK A END-TO-END: drive Task A with the VISION steering head (no privileged yaw).

The low-level fast policy follows a (vx, vy, wz) command. Instead of the kyaw teacher
(privileged world yaw, eval-illegal), we compute wz from the trained steering head that
reads the head camera RGB (frozen DINOv2) + lidar — exactly the eval-legal sensors. This
tests whether "look at the red boundary -> steer" actually keeps the robot on course all
the way to the goal, end-to-end in sim, with NO world-yaw cheat.

  CUDA_VISIBLE_DEVICES=1 TMPDIR=/hdd0/sohyun/tmp PYTHONPATH=/hdd0/sohyun \
    python scripts/eval_taska_steering_head.py --headless \
      --head /hdd0/sohyun/atec_2026_tesk_e/steering_head/steering_head.pt \
      --vx 2.0 --steps 8000 --out /hdd0/sohyun/atec_2026_tesk_e/taska_head_eval
"""
import argparse
import os
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Task A end-to-end with vision steering head.")
parser.add_argument("--task", type=str, default="ATEC-TaskA-B2Piper")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--head", type=str, nargs="+", required=True,
                    help="one or more steering_head.pt paths. >1 -> ensemble (see --ensemble).")
parser.add_argument("--ensemble", type=str, default="avg", choices=["avg", "gate"],
                    help="avg = mean of head wz; gate = use head[0] for x<gate_x, head[1] for x>=gate_x "
                         "(head[0]=front/slope-strong e.g. PD7, head[1]=stairs-strong e.g. PD5).")
parser.add_argument("--gate_x", type=float, default=55.0, help="x boundary for --ensemble gate")
parser.add_argument("--vx", type=float, default=2.0)
parser.add_argument("--wz_max", type=float, default=0.8)
parser.add_argument("--steps", type=int, default=8000)
parser.add_argument("--every", type=int, default=1, help="run the head every N steps (hold wz between)")
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--video", action="store_true", default=False, help="record an mp4 of the run")
parser.add_argument("--video_length", type=int, default=8000, help="frames to record")
parser.add_argument("--seed", type=int, default=0, help="seed for reproducible closed-loop eval")
parser.add_argument("--enhanced_determinism", action="store_true", default=False,
                    help="enable PhysX enhanced determinism for reproducible GPU physics")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import gymnasium as gym  # noqa: E402
import json  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
import atec_rl_lab.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from atec_rl_lab.tasks.task_base.action_base import apply_safe_action_spec  # noqa: E402
from atec_2026_tesk_e.ATEC2026_Simulation_Challenge.demo.solution_rl import AlgSolution  # noqa: E402

solution = AlgSolution()


# --- steering head (must match scripts/train_steering_head.py) ---
class SteeringHead(nn.Module):
    def __init__(self, rgb_dim, lidar_dim, hidden=256):
        super().__init__()
        self.rgb_mlp = nn.Sequential(
            nn.Linear(rgb_dim, hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.lidar_mlp = nn.Sequential(
            nn.Linear(lidar_dim, hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, rgb_feat, lidar):
        r = self.rgb_mlp(rgb_feat)
        l = self.lidar_mlp(lidar)
        return self.fuse(torch.cat([r, l], -1)).squeeze(-1)


def main():
    os.makedirs(args_cli.out, exist_ok=True)
    dev = "cuda"

    # load one or more steering heads, each with its OWN normalization stats.
    # An ensemble mixes their wz outputs (avg or x-gated) -- a probe to see if combining
    # a slope-strong head (PD7) with a stairs-strong head (PD5) beats either alone.
    heads = []
    for hp in args_cli.head:
        ckpt = torch.load(hp, map_location=dev, weights_only=False)
        h = SteeringHead(ckpt["rgb_dim"], ckpt["lidar_dim"]).to(dev).eval()
        h.load_state_dict(ckpt["model"])
        heads.append({
            "model": h,
            "rf_mu": torch.tensor(ckpt["rf_mu"], dtype=torch.float32, device=dev),
            "rf_sd": torch.tensor(ckpt["rf_sd"], dtype=torch.float32, device=dev),
            "li_mu": torch.tensor(ckpt["li_mu"], dtype=torch.float32, device=dev),
            "li_sd": torch.tensor(ckpt["li_sd"], dtype=torch.float32, device=dev),
        })
    print(f"[ensemble] loaded {len(heads)} head(s), mode={args_cli.ensemble}"
          + (f" gate_x={args_cli.gate_x}" if args_cli.ensemble == "gate" else ""), flush=True)

    def ensemble_wz(rgb_feat_raw, rng_raw, cur_x):
        """rgb_feat_raw: (1,384) un-normalized DINOv2 CLS; rng_raw: (1,L) un-normalized lidar."""
        wzs = []
        for hd in heads:
            rf = (rgb_feat_raw - hd["rf_mu"]) / hd["rf_sd"]
            lf = (rng_raw - hd["li_mu"]) / hd["li_sd"]
            wzs.append(float(hd["model"](rf, lf)[0]))
        if len(wzs) == 1:
            return wzs[0]
        if args_cli.ensemble == "gate":
            # head[0] for the front (slope) zone, head[1] once we reach the stairs zone.
            # NOTE: uses world x -> NOT eval-legal; this is a DIAGNOSTIC probe only. If it
            # wins, replace the gate with a lidar/depth terrain classifier (real-legal).
            return wzs[0] if cur_x < args_cli.gate_x else wzs[1]
        return sum(wzs) / len(wzs)  # avg

    # frozen DINOv2 backbone
    from transformers import AutoModel, AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    dino = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval()

    def rgb_to_feat(rgb_np):
        inp = proc(images=[rgb_np], return_tensors="pt").to(dev)
        with torch.inference_mode():
            return dino(**inp).last_hidden_state[:, 0]  # (1,384)

    # env
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device,
                            num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    action_spec = solution.get_action_spec() if hasattr(solution, "get_action_spec") else None
    env_cfg = apply_safe_action_spec(env_cfg, json.dumps(action_spec))
    if args_cli.enhanced_determinism:
        # PhysX enhanced determinism: reproducible GPU physics at some perf cost. Reduces
        # the run-to-run max_x spread so head comparisons are trustworthy (GPU float
        # reduction is otherwise non-deterministic and seed alone can't fix it).
        env_cfg.sim.physx.enable_enhanced_determinism = True
    # let it run long; keep fall termination so a real fall ends the episode honestly
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    if args_cli.video:
        video_dir = os.path.join(args_cli.out, "video")
        env = gym.wrappers.RecordVideo(
            env, video_folder=video_dir, name_prefix="taska_head",
            step_trigger=lambda s: s == 0, video_length=args_cli.video_length,
            disable_logger=True,
        )
        print(f"[video] recording to {video_dir}", flush=True)

    # seed everything so the closed loop is reproducible across runs (otherwise max_x
    # varies wildly and we can't tell if a head change helped or it was just RNG).
    import random as _random
    _random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    torch.cuda.manual_seed_all(args_cli.seed)
    obs, _ = env.reset(seed=args_cli.seed)
    robot = env.unwrapped.scene["robot"]
    head_cam = env.unwrapped.scene["head_camera"]
    try:
        lidar = env.unwrapped.scene["lidar_sensor"]
    except (KeyError, Exception):
        lidar = None
        print("[WARN] no lidar_sensor; head needs lidar — aborting.", flush=True)

    wz = 0.0
    traj = []
    max_x = -1e9
    fell = False

    for step in range(args_cli.steps):
        with torch.inference_mode():
            if step % args_cli.every == 0 and lidar is not None:
                # --- read eval-legal sensors ---
                rgb = head_cam.data.output["rgb"][0].cpu().numpy()[:, :, :3].astype(np.uint8)
                hits = lidar.data.ray_hits_w[0]
                base = robot.data.root_pos_w[0]
                rng = torch.norm(hits - base.unsqueeze(0), dim=1)
                rng = torch.where(torch.isfinite(rng), rng, torch.full_like(rng, 10.0))
                rng = torch.clip(rng, 0.0, 10.0).view(1, -1)      # (1,L)
                # --- steering head(s) -> wz (each head normalizes with its own stats) ---
                rgb_feat_raw = rgb_to_feat(rgb)                    # (1,384) un-normalized
                cur_x = float(robot.data.root_pos_w[0, 0])
                wz = ensemble_wz(rgb_feat_raw, rng, cur_x)
                wz = max(-args_cli.wz_max, min(args_cli.wz_max, wz))

            solution.fixed_velocity_commands = torch.tensor(
                [args_cli.vx, 0.0, wz], device=solution.device, dtype=torch.float32).view(1, 3)

            resp = solution.predicts(obs, 0.0)
            act = (torch.zeros((1, 20), device=dev) if (resp.get("giveup") or not resp["action"])
                   else torch.tensor(resp["action"], dtype=torch.float32, device=dev).view(1, -1))
            obs, _, term, trunc, _ = env.step(act)

            p = robot.data.root_pos_w[0]
            x, y = float(p[0]), float(p[1])
            max_x = max(max_x, x)
            if step % args_cli.every == 0:
                traj.append((step, x, y, wz))
            if step % 200 == 0:
                print(f"[step {step:4d}] x={x:+.1f} y={y:+.2f} wz={wz:+.2f} max_x={max_x:+.1f}", flush=True)

            if bool(term[0]):
                # terminated: either reached goal (x>145) or fell
                if x < 140:
                    fell = True
                    print(f"[TERM] episode ended at step {step}, x={x:+.1f} y={y:+.2f} (fall/contact)", flush=True)
                else:
                    print(f"[GOAL] reached x={x:+.1f} at step {step}!", flush=True)
                break

    # save trajectory
    np.savez_compressed(os.path.join(args_cli.out, "head_eval_traj.npz"),
                        traj=np.array(traj, dtype=np.float32))
    goal = max_x >= 145
    print(f"\n[DONE] max_x={max_x:+.1f}  goal_reached={goal}  fell={fell}", flush=True)
    print(f"  Task A course: spawn x=-141 -> goal x>145 (~286 m).", flush=True)
    print(f"  traj -> {os.path.join(args_cli.out, 'head_eval_traj.npz')}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
