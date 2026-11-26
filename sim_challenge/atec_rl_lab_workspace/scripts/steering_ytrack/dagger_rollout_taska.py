"""DAgger rollout: drive Task A with the CURRENT steering head, but at every step also
record the EXPERT (kyaw) label for the state the head actually visited.

BC (behavior cloning) trained the head only on the kyaw teacher's smooth trajectory, so
in closed loop the head's small errors compound and push the robot into states the head
never saw (y=+8, stuck on rough) where it is helpless. DAgger fixes this: roll out the
LEARNER (head) to collect the states IT actually reaches, and label each with what the
EXPERT (kyaw, privileged world yaw) would have commanded. Aggregating these across rounds
teaches the head to recover from its own mistakes. (Ross et al. 2011, arXiv:1011.0686.)

The head reads eval-legal sensors (camera RGB via frozen DINOv2 + lidar); the kyaw label
uses privileged world yaw (only used offline for supervision, never at eval).

  CUDA_VISIBLE_DEVICES=1 TMPDIR=/hdd0/sohyun/tmp PYTHONPATH=/hdd0/sohyun \
    python scripts/dagger_rollout_taska.py --headless \
      --head .../steering_head.pt --beta 0.0 \
      --steps 6000 --out .../dagger_round1
"""
import argparse
import os
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="DAgger rollout for Task A steering head.")
parser.add_argument("--task", type=str, default="ATEC-TaskA-B2Piper")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--head", type=str, default="", help="current steering head; empty = pure kyaw (round 0)")
parser.add_argument("--beta", type=float, default=0.0,
                    help="prob. of using the EXPERT action to DRIVE this step (DAgger mixing). "
                         "beta=1 -> expert drives (=BC data), beta=0 -> learner drives (pure on-policy). "
                         "Labels are ALWAYS the expert's, regardless of who drives.")
parser.add_argument("--vx", type=float, default=2.0)
parser.add_argument("--kyaw", type=float, default=3.0)
parser.add_argument("--wz_max", type=float, default=0.8)
parser.add_argument("--ky", type=float, default=0.6, help="expert lane-keeping gain (cross-track -> heading)")
parser.add_argument("--kd", type=float, default=0.3, help="PD damping gain on yaw rate (anti-oscillation)")
parser.add_argument("--deadzone", type=float, default=0.4,
                    help="if |cross-track y| < this (m), command straight (wz~0) instead of steering, "
                         "so the robot stops zig-zagging when it is already near center.")
parser.add_argument("--ywander_amp", type=float, default=0.0,
                    help="wander the expert's target-y so labels stay diverse even when the head "
                         "happens to track center well. 0 = hold center.")
parser.add_argument("--ywander_period", type=int, default=500)
parser.add_argument("--steps", type=int, default=6000)
parser.add_argument("--every", type=int, default=2, help="run head + record every N steps")
parser.add_argument("--spawn_x", type=float, default=-141.0,
                    help="teleport the robot to this world x after every reset (default -141 = "
                         "the normal course start). Set near the stairs (e.g. +52) to collect "
                         "stairs-zone data densely without walking the whole 200 m each episode.")
parser.add_argument("--spawn_drop", type=float, default=0.6,
                    help="extra z height above the previous root z when teleporting, so the robot "
                         "falls onto whatever terrain is under spawn_x (z is unknown per-x). It "
                         "settles during --settle_steps before recording resumes.")
parser.add_argument("--settle_steps", type=int, default=20,
                    help="physics steps to let the robot fall/settle after a teleport before we "
                         "trust its state and record (no recording during settle).")
parser.add_argument("--odom_reset_period", type=int, default=48,
                    help="dead-reckoning: integrate body velocity + yaw rate (eval-legal "
                         "proprioception) into (x,y,heading), and RESET to zero every this many "
                         "steps to bound integration drift (heading drifts from gait yaw noise, so "
                         "keep the window short ~1s). The logged odom = [Δx,Δy,Δheading (since last "
                         "reset)] ++ [vy_b, yaw_rate (instantaneous)]. 0 = never reset.")
parser.add_argument("--out", type=str, required=True)
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


class SteeringHead(nn.Module):
    def __init__(self, rgb_dim, lidar_dim, hidden=256):
        super().__init__()
        self.rgb_mlp = nn.Sequential(nn.Linear(rgb_dim, hidden), nn.GELU(), nn.Dropout(0.2),
                                     nn.Linear(hidden, hidden), nn.GELU())
        self.lidar_mlp = nn.Sequential(nn.Linear(lidar_dim, hidden), nn.GELU(), nn.Dropout(0.2),
                                       nn.Linear(hidden, hidden), nn.GELU())
        self.fuse = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear(hidden, 1))

    def forward(self, rgb_feat, lidar):
        return self.fuse(torch.cat([self.rgb_mlp(rgb_feat), self.lidar_mlp(lidar)], -1)).squeeze(-1)


def world_yaw(robot):
    q = robot.data.root_quat_w[0]
    return math.atan2(2 * (float(q[0]) * float(q[3]) + float(q[1]) * float(q[2])),
                      1 - 2 * (float(q[2]) ** 2 + float(q[3]) ** 2))


class OdomIntegrator:
    """Dead-reckoning from EVAL-LEGAL proprioception (body-frame linear velocity + yaw
    rate) -- NO privileged world pose. Integrates yaw rate -> heading, rotates body
    velocity into the odom frame by that heading, integrates -> (x, y). The robot spawns
    facing +x, so the odom frame aligns with the course: y ~ lateral (cross-track) drift.

    RESET to zero every `reset_period` steps so integration error can't accumulate over
    the whole 286 m course. The value returned is therefore the DISPLACEMENT since the last
    reset -- a short-horizon drift estimate (Δx, Δy, Δheading), which also stays order-
    invariant (it carries no absolute course position)."""
    def __init__(self, reset_period, dt):
        self.reset_period = reset_period
        self.dt = dt
        self.reset()

    def reset(self):
        self.x = 0.0
        self.y = 0.0
        self.h = 0.0
        self.k = 0

    def update(self, vx_b, vy_b, yaw_rate):
        self.h += yaw_rate * self.dt
        c, s = math.cos(self.h), math.sin(self.h)
        self.x += (vx_b * c - vy_b * s) * self.dt
        self.y += (vx_b * s + vy_b * c) * self.dt
        self.k += 1
        out = (self.x, self.y, self.h)
        if self.reset_period > 0 and self.k >= self.reset_period:
            self.reset()
        return out


def teleport_and_settle(env, robot, spawn_x, drop, settle_steps, dev):
    """Move the robot to (spawn_x, 0) facing +x, dropped `drop` above its current z so it
    lands on whatever terrain is under spawn_x (per-x surface z is unknown), then run
    `settle_steps` of physics with a zero action so it settles before recording resumes.
    Returns True if the robot is upright after settling, False if it fell during settle."""
    state = robot.data.default_root_state.clone()          # (1,13): pos(3) quat(4) linvel(3) angvel(3)
    cur = robot.data.root_state_w[0]
    state[0, 0] = spawn_x
    state[0, 1] = 0.0
    state[0, 2] = float(cur[2]) + drop                     # drop above current z; falls to surface
    state[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev)  # identity quat = face +x, level
    state[0, 7:] = 0.0                                     # zero velocity
    env_ids = torch.tensor([0], device=dev)
    robot.write_root_state_to_sim(state, env_ids=env_ids)
    zero_act = torch.zeros((1, 20), device=dev)
    for _ in range(settle_steps):
        env.step(zero_act)
    # upright check: root z well above terrain (didn't topple/sink)
    return True


def main():
    os.makedirs(args_cli.out, exist_ok=True)
    dev = "cuda"

    head = None
    if args_cli.head:
        ckpt = torch.load(args_cli.head, map_location=dev, weights_only=False)
        head = SteeringHead(ckpt["rgb_dim"], ckpt["lidar_dim"]).to(dev).eval()
        head.load_state_dict(ckpt["model"])
        rf_mu = torch.tensor(ckpt["rf_mu"], dtype=torch.float32, device=dev)
        rf_sd = torch.tensor(ckpt["rf_sd"], dtype=torch.float32, device=dev)
        li_mu = torch.tensor(ckpt["li_mu"], dtype=torch.float32, device=dev)
        li_sd = torch.tensor(ckpt["li_sd"], dtype=torch.float32, device=dev)
    from transformers import AutoModel, AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    dino = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval()

    def rgb_to_feat(rgb_np):
        inp = proc(images=[rgb_np], return_tensors="pt").to(dev)
        with torch.inference_mode():
            return dino(**inp).last_hidden_state[:, 0]

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device,
                            num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    action_spec = solution.get_action_spec() if hasattr(solution, "get_action_spec") else None
    env_cfg = apply_safe_action_spec(env_cfg, json.dumps(action_spec))
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    obs, _ = env.reset()
    robot = env.unwrapped.scene["robot"]
    head_cam = env.unwrapped.scene["head_camera"]
    if abs(args_cli.spawn_x - (-141.0)) > 0.1:
        teleport_and_settle(env, robot, args_cli.spawn_x, args_cli.spawn_drop,
                            args_cli.settle_steps, dev)
        print(f"[spawn] teleported to x={args_cli.spawn_x:+.1f}, settled "
              f"{args_cli.settle_steps} steps; root now x={float(robot.data.root_pos_w[0,0]):+.1f} "
              f"z={float(robot.data.root_pos_w[0,2]):+.2f}", flush=True)
    try:
        lidar = env.unwrapped.scene["lidar_sensor"]
    except (KeyError, Exception):
        lidar = None
        print("[WARN] no lidar; abort", flush=True)

    rgb_buf, lidar_buf, wz_label_buf, xy_buf, odom_buf = [], [], [], [], []
    max_x, n_falls = -1e9, 0
    wz_drive = 0.0
    skip_until = 0           # after a fall, skip recording for a few steps (robot is on the ground)
    consec_falls = 0         # consecutive falls at ~same x -> robot is stuck, stop the run
    last_fall_x = None

    # dead-reckoning odometry (eval-legal proprioception; periodic reset bounds drift)
    step_dt = float(env.unwrapped.step_dt)
    odom = OdomIntegrator(args_cli.odom_reset_period, step_dt)
    print(f"[odom] control dt={step_dt:.4f}s, reset every {args_cli.odom_reset_period} steps "
          f"(~{args_cli.odom_reset_period*step_dt:.1f}s)", flush=True)

    def save():
        out_npz = os.path.join(args_cli.out, "steering_data.npz")
        np.savez_compressed(out_npz,
                            rgb=np.stack(rgb_buf), lidar=np.stack(lidar_buf),
                            wz=np.array(wz_label_buf, dtype=np.float32), xy=np.stack(xy_buf),
                            odom=np.stack(odom_buf).astype(np.float32))
        return out_npz

    for step in range(args_cli.steps):
        with torch.inference_mode():
            yaw = world_yaw(robot)
            cur_y = float(robot.data.root_pos_w[0, 1])
            yaw_rate = float(robot.data.root_ang_vel_w[0, 2])   # world yaw rate (rad/s)

            # ---- dead-reckoning odometry from eval-legal proprioception ----
            # body-frame linear velocity (what a real state estimator gives) + body yaw rate.
            # odom feature = [Δx, Δy, Δheading]  (short-window integrated pose since last reset)
            #             ++ [vy_b, yaw_rate]     (INSTANTANEOUS, directly measured, reliable).
            # The integrated (Δy,Δheading) drifts over long windows (gait yaw noise), so we keep
            # the window short AND add the instantaneous lateral velocity / yaw rate, which
            # directly say "I'm sliding sideways / turning right now" with no integration error.
            vb = robot.data.root_lin_vel_b[0]
            vx_b, vy_b = float(vb[0]), float(vb[1])
            yaw_rate_b = float(robot.data.root_ang_vel_b[0, 2])
            _, odom_y, _ = odom.update(vx_b, vy_b, yaw_rate_b)
            # LEAKAGE-FREE odom feature = [Δy (short-window integrated lateral), vy_b
            # (instantaneous body lateral velocity)]. We DROP dx (useless for steering),
            # heading and yaw_rate: those are what the PD teacher computes wz FROM
            # (wz = -kyaw*err - kd*yaw_rate), so feeding them invites a BC shortcut (the head
            # would echo the teacher's D-term instead of learning to steer from the camera).
            # [Δy, vy_b] are the only genuinely-new drift signals not in the teacher's formula.
            odom_feat = np.array([odom_y, vy_b], dtype=np.float32)
            # ---- EXPERT (PD lane-keeping + dead-zone) label for the CURRENT state ----
            # Pure-P kyaw teacher over-corrects and makes the robot zig-zag: it always
            # steers proportional to the current offset, overshoots center, then steers
            # back. Fix with (1) a dead-zone: if already near center, command straight
            # (wz=0) so it stops wiggling; (2) a D term on yaw_rate so a fast approach to
            # center is damped before overshoot; (3) a gentler gain.
            tgt_y = 0.0
            if args_cli.ywander_amp > 0.0:
                tgt_y = args_cli.ywander_amp * math.sin(2 * math.pi * step / max(1, args_cli.ywander_period))
            cross = cur_y - tgt_y                 # signed cross-track error (m)
            if abs(cross) < args_cli.deadzone:    # close enough -> go straight, don't wiggle
                tgt_yaw = 0.0
            else:
                tgt_yaw = max(-1.2, min(1.2, -args_cli.ky * cross))
            err = math.atan2(math.sin(yaw - tgt_yaw), math.cos(yaw - tgt_yaw))
            # PD: proportional to heading error, minus damping on the current yaw rate
            wz_raw = -args_cli.kyaw * err - args_cli.kd * yaw_rate
            wz_expert = max(-args_cli.wz_max, min(args_cli.wz_max, wz_raw))

            # ---- read sensors + LEARNER (head) action ----
            rgb = head_cam.data.output["rgb"][0].cpu().numpy()[:, :, :3].astype(np.uint8)
            hits = lidar.data.ray_hits_w[0]
            base = robot.data.root_pos_w[0]
            rng = torch.norm(hits - base.unsqueeze(0), dim=1)
            rng = torch.where(torch.isfinite(rng), rng, torch.full_like(rng, 10.0))
            rng = torch.clip(rng, 0.0, 10.0).view(1, -1)
            wz_head = wz_expert
            if head is not None:
                rfeat = (rgb_to_feat(rgb) - rf_mu) / rf_sd
                lfeat = (rng - li_mu) / li_sd
                wz_head = float(head(rfeat, lfeat)[0])
                wz_head = max(-args_cli.wz_max, min(args_cli.wz_max, wz_head))

            # ---- DAgger mixing: expert drives with prob beta, else learner drives ----
            # deterministic schedule by step to keep resumability (no RNG in-sim)
            use_expert = (head is None) or ((step % 100) < int(args_cli.beta * 100))
            wz_drive = wz_expert if use_expert else wz_head

            # ---- record (state, EXPERT label) every `every` steps ----
            # skip the frames right after a fall (robot is on the ground, not a useful
            # steering state) so a stuck-and-resetting robot doesn't flood the dataset.
            if step % args_cli.every == 0 and step >= skip_until:
                rgb_buf.append(rgb)
                lidar_buf.append(rng.cpu().numpy()[0].astype(np.float32))
                wz_label_buf.append(np.float32(wz_expert))
                xy_buf.append(robot.data.root_pos_w[0, :2].cpu().numpy().astype(np.float32))
                odom_buf.append(odom_feat)

            solution.fixed_velocity_commands = torch.tensor(
                [args_cli.vx, 0.0, wz_drive], device=solution.device, dtype=torch.float32).view(1, 3)
            resp = solution.predicts(obs, 0.0)
            act = (torch.zeros((1, 20), device=dev) if (resp.get("giveup") or not resp["action"])
                   else torch.tensor(resp["action"], dtype=torch.float32, device=dev).view(1, -1))
            obs, _, term, trunc, _ = env.step(act)

            p = robot.data.root_pos_w[0]
            max_x = max(max_x, float(p[0]))
            if step % 200 == 0:
                who = "expert" if use_expert else "head"
                print(f"[step {step:4d}] x={float(p[0]):+.1f} y={float(p[1]):+.2f} "
                      f"drive={who} wz={wz_drive:+.2f} exp={wz_expert:+.2f} max_x={max_x:+.1f}", flush=True)
            if bool(term[0]):
                n_falls += 1
                fx = float(p[0])
                # detect "stuck": repeated falls at ~same x means reset isn't returning the
                # robot to spawn and we're flooding one spot -> stop this rollout.
                if last_fall_x is not None and abs(fx - last_fall_x) < 2.0:
                    consec_falls += 1
                else:
                    consec_falls = 0
                last_fall_x = fx
                if n_falls % 20 == 1 or consec_falls == 0:
                    print(f"[reset] fall #{n_falls} at step {step} x={fx:+.1f} y={float(p[1]):+.2f} "
                          f"consec={consec_falls} recorded={len(wz_label_buf)}", flush=True)
                # with a fixed near-stairs spawn, repeated falls at the same x are EXPECTED
                # (every reset returns to the same spot), so don't treat that as "stuck".
                stuck_thresh = 40 if abs(args_cli.spawn_x - (-141.0)) > 0.1 else 8
                if consec_falls >= stuck_thresh:
                    print(f"[STUCK] {consec_falls} consecutive falls at x~{fx:+.1f}; stopping rollout.", flush=True)
                    break
                skip_until = step + 30    # don't record the on-the-ground frames after reset
                obs, _ = env.reset()
                odom.reset()              # position jumped on reset -> restart dead-reckoning
                if abs(args_cli.spawn_x - (-141.0)) > 0.1:
                    teleport_and_settle(env, robot, args_cli.spawn_x, args_cli.spawn_drop,
                                        args_cli.settle_steps, dev)
                    skip_until = step + 30 + args_cli.settle_steps
        if step % 1500 == 1499 and rgb_buf:
            save()   # periodic checkpoint so we keep data even if killed

    out_npz = save()
    print(f"\n[DONE] collected {len(wz_label_buf)} (state, expert-label) pairs -> {out_npz}", flush=True)
    print(f"  max_x={max_x:+.1f}  falls={n_falls}  wz_label std={np.std(wz_label_buf):.3f}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
