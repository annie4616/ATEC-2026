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
parser.add_argument("--iplanner_model", type=str,
                    default="/hdd0/sohyun/atec_2026_tesk_e/iplanner/models/plannernet.pt",
                    help="pretrained iPlanner plannernet.pt")
parser.add_argument("--crop_size", type=int, nargs=2, default=[360, 640],
                    help="depth image size fed to iPlanner (H W)")
parser.add_argument("--max_depth", type=float, default=10.0,
                    help="clip/zero depths beyond this (m), matching iPlanner training")
parser.add_argument("--goal_dist", type=float, default=8.0,
                    help="local goal placed this far ahead along +x (robot frame) each step")
parser.add_argument("--goal_world", type=float, nargs=2, default=None,
                    help="FINAL destination in world (x,y). If set, the local goal fed to "
                         "iPlanner is this point transformed into the robot frame (clamped to "
                         "goal_dist range), so iPlanner turns corners toward the destination "
                         "instead of always pointing straight ahead. Knowing your destination "
                         "bearing is standard navigation (not a world-yaw steering cheat).")
parser.add_argument("--wp_idx", type=int, default=2,
                    help="which iPlanner keypoint to steer toward (pure-pursuit lookahead)")
parser.add_argument("--wp_gain", type=float, default=1.0,
                    help="wz = wp_gain * bearing-to-lookahead-waypoint. <1 damps steering.")
parser.add_argument("--vx", type=float, default=2.0)
parser.add_argument("--wz_max", type=float, default=0.8)
parser.add_argument("--wz_ema", type=float, default=1.0,
                    help="EMA smoothing on the head's wz: wz=a*wz_new+(1-a)*wz_prev. a=1 -> no "
                         "smoothing (raw head). a<1 damps the step-to-step jerk that makes the "
                         "robot topple on rough terrain (failing seeds have 24-42%% |wz|>0.5 vs "
                         "kyaw's smooth 15%%). Try 0.2-0.5.")
parser.add_argument("--steps", type=int, default=8000)
parser.add_argument("--every", type=int, default=1, help="run the head every N steps (hold wz between)")
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--video", action="store_true", default=False, help="record an mp4 of the run")
parser.add_argument("--video_length", type=int, default=8000, help="frames to record")
parser.add_argument("--seed", type=int, default=0, help="seed for reproducible closed-loop eval")
parser.add_argument("--enhanced_determinism", action="store_true", default=False,
                    help="enable PhysX enhanced determinism for reproducible GPU physics")
parser.add_argument("--no_illegal_contact", action="store_true", default=False,
                    help="disable the illegal_contact termination (base/hip/thigh touching "
                         "terrain ends the episode). Real-world has no such rule; failures were "
                         "almost all this. Keeps fall (root height) termination.")
parser.add_argument("--vx_slow_k", type=float, default=0.0,
                    help="TOP-Nav anti-drift: vx = vx0*exp(-k*|wz|). Slows forward speed when "
                         "steering hard so the robot recovers before drifting out of course. "
                         "0 = constant speed. Try 0.5-1.5.")
parser.add_argument("--course_half_w", type=float, default=-1.0,
                    help="override the red-strip half-width (strips at y=+/-this). Default -1 "
                         "keeps the env's built-in 4.0. Use to test whether the head generalizes "
                         "to a DIFFERENT lane spacing than it trained on (robustness, no retrain).")
parser.add_argument("--kd", type=float, default=0.0,
                    help="RUNTIME derivative damping on the head's wz: wz -= kd*yaw_rate (body-frame "
                         "yaw rate, eval-legal). The head outputs wz directly and can oscillate on "
                         "slope/stairs; subtracting a term proportional to how fast the robot is "
                         "ALREADY turning damps that oscillation (classic D control) without "
                         "retraining. 0 = off. Try 0.3-0.8; stronger = less wiggle but slower to "
                         "commit to a turn. Applied BEFORE wz_max clip and EMA.")
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


# iPlanner net + traj optimizer are loaded from the cloned repo (pickled full model).
import sys as _sys  # noqa: E402
_IPLANNER_ROOT = "/hdd0/sohyun/atec_2026_tesk_e/iplanner"
_sys.path.insert(0, _IPLANNER_ROOT + "/iplanner")
_sys.path.insert(0, _IPLANNER_ROOT)
import torchvision.transforms as _T  # noqa: E402
from iplanner import traj_opt as _traj_opt  # noqa: E402


def main():
    os.makedirs(args_cli.out, exist_ok=True)
    dev = "cuda"

    # --- load pretrained iPlanner (front-depth -> waypoints) ---
    ipnet, _ = torch.load(args_cli.iplanner_model, map_location=dev, weights_only=False)
    ipnet = ipnet.to(dev).eval()
    traj_gen = _traj_opt.TrajOpt()
    depth_resize = _T.Resize(tuple(args_cli.crop_size), antialias=None)
    print(f"[iplanner] loaded {args_cli.iplanner_model}  crop={args_cli.crop_size} "
          f"max_depth={args_cli.max_depth}", flush=True)

    # env
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device,
                            num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    action_spec = solution.get_action_spec() if hasattr(solution, "get_action_spec") else None
    env_cfg = apply_safe_action_spec(env_cfg, json.dumps(action_spec))
    if args_cli.course_half_w > 0.0:
        # Move the red boundary strips to a DIFFERENT half-width than training (4.0) to test
        # lane-spacing robustness. The strips are AssetBaseCfg with init_state.pos=(x,y,z).
        hw = args_cli.course_half_w
        for name, sign in (("red_strip_left", +1.0), ("red_strip_right", -1.0)):
            strip = getattr(env_cfg.scene, name, None)
            if strip is not None:
                x, _, z = strip.init_state.pos
                strip.init_state.pos = (x, sign * hw, z)
        print(f"[cfg] red-strip half-width overridden to {hw:.2f} m (train was 4.0)", flush=True)
    if args_cli.enhanced_determinism:
        # PhysX enhanced determinism: reproducible GPU physics at some perf cost. Reduces
        # the run-to-run max_x spread so head comparisons are trustworthy (GPU float
        # reduction is otherwise non-deterministic and seed alone can't fix it).
        env_cfg.sim.physx.enable_enhanced_determinism = True
    if args_cli.no_illegal_contact:
        # illegal_contact ends the episode when base/hip/thigh grazes the terrain. Term-cause
        # logging showed nearly ALL failures are this (robot upright, z normal, just a hip/
        # thigh touch on rough/slope/stairs) -- a sim-only rule; real-world has no such kill.
        # Drop it so only a genuine fall (root height < min) or reaching the goal ends the run.
        env_cfg.terminations.illegal_contact = None
        print("[cfg] illegal_contact termination DISABLED", flush=True)
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
    goal_reached_flag = False

    for step in range(args_cli.steps):
        with torch.inference_mode():
            if step % args_cli.every == 0:
                # --- read front DEPTH image (camera depth for now; will be swapped for
                # LiDAR-projected depth once this integration is verified) ---
                depth = head_cam.data.output["depth"][0]            # (H,W) or (H,W,1) meters
                depth = depth.squeeze(-1) if depth.dim() == 3 else depth
                d = depth.clone().float()
                d[~torch.isfinite(d)] = 0.0
                d[d > args_cli.max_depth] = 0.0
                d = depth_resize(d.unsqueeze(0).unsqueeze(0))        # (1,1,H,W)
                d = d.expand(1, 3, -1, -1)                           # iPlanner wants 3ch

                # --- local goal in the ROBOT frame ---
                if args_cli.goal_world is not None:
                    # transform the FINAL world destination into the robot frame, then clamp to
                    # the local goal_dist range. iPlanner then plans toward the destination
                    # bearing and turns corners toward it (its own depth avoids the walls).
                    rp = robot.data.root_pos_w[0]
                    rq = robot.data.root_quat_w[0]     # (w,x,y,z)
                    dxw = args_cli.goal_world[0] - float(rp[0])
                    dyw = args_cli.goal_world[1] - float(rp[1])
                    # yaw from quaternion
                    w, xq, yq, zq = float(rq[0]), float(rq[1]), float(rq[2]), float(rq[3])
                    yaw = math.atan2(2.0 * (w * zq + xq * yq), 1.0 - 2.0 * (yq * yq + zq * zq))
                    # world delta -> robot frame (rotate by -yaw)
                    cx = math.cos(-yaw) * dxw - math.sin(-yaw) * dyw
                    cy = math.sin(-yaw) * dxw + math.cos(-yaw) * dyw
                    dist = math.hypot(cx, cy)
                    if dist > args_cli.goal_dist:      # clamp far goal into local range
                        cx, cy = cx / dist * args_cli.goal_dist, cy / dist * args_cli.goal_dist
                    goal_rf = torch.tensor([[cx, cy, 0.0]], device=dev, dtype=torch.float32)
                else:
                    goal_rf = torch.tensor([[args_cli.goal_dist, 0.0, 0.0]],
                                           device=dev, dtype=torch.float32)

                keypoints, fear = ipnet(d, goal_rf)                 # (1,5,3),(1,1)
                kp = keypoints[0]                                   # (5,3) robot-frame waypoints
                # pure-pursuit: steer toward the wp_idx-th waypoint's bearing, scaled by a gain.
                i = min(args_cli.wp_idx, kp.shape[0] - 1)
                wp_x, wp_y = float(kp[i, 0]), float(kp[i, 1])
                bearing = math.atan2(wp_y, max(wp_x, 1e-3))         # bearing to lookahead wp
                wz_raw = args_cli.wp_gain * bearing
                if args_cli.kd > 0.0:
                    yaw_rate_b = float(robot.data.root_ang_vel_b[0, 2])
                    wz_raw = wz_raw - args_cli.kd * yaw_rate_b
                wz_raw = max(-args_cli.wz_max, min(args_cli.wz_max, wz_raw))
                wz = args_cli.wz_ema * wz_raw + (1.0 - args_cli.wz_ema) * wz

            # TOP-Nav-style anti-drift: slow forward speed when steering hard (= being pushed
            # / correcting hard), so the robot has time to pull back before it drifts out of
            # the course. vx = vx0 * exp(-k*|wz|). k=0 keeps constant speed.
            vx_cmd = args_cli.vx * math.exp(-args_cli.vx_slow_k * abs(wz))

            solution.fixed_velocity_commands = torch.tensor(
                [vx_cmd, 0.0, wz], device=solution.device, dtype=torch.float32).view(1, 3)

            resp = solution.predicts(obs, 0.0)
            act = (torch.zeros((1, 20), device=dev) if (resp.get("giveup") or not resp["action"])
                   else torch.tensor(resp["action"], dtype=torch.float32, device=dev).view(1, -1))
            obs, _, term, trunc, _ = env.step(act)

            p = robot.data.root_pos_w[0]
            x, y, z = float(p[0]), float(p[1]), float(p[2])
            max_x = max(max_x, x)
            if step % args_cli.every == 0:
                traj.append((step, x, y, wz, z))
            if step % 200 == 0:
                print(f"[step {step:4d}] x={x:+.1f} y={y:+.2f} z={z:+.2f} wz={wz:+.2f} max_x={max_x:+.1f}", flush=True)

            if bool(term[0]):
                # terminated: either reached goal (x>145) or fell.
                # Report WHICH termination term fired (illegal_contact vs fall vs time_out)
                # so we can tell a real topple from an eval-only hip/thigh-contact kill.
                which = []
                try:
                    tm = env.unwrapped.termination_manager
                    for tname in tm.active_terms:
                        val = tm.get_term(tname)
                        if bool(val[0]):
                            which.append(tname)
                except Exception as e:
                    which = [f"(term-read failed: {e})"]
                # goal = the reach_goal term actually fired (works for both the x>=145 goal
                # and the bent-goal (x,y)-proximity goal). Fall back to x for older logs.
                reached_goal = ("reach_goal_x" in which) or (x >= 140 and not which)
                if not reached_goal:
                    fell = True
                    tail = traj[-8:]
                    ztail = " ".join(f"{t[4]:+.2f}" for t in tail)
                    print(f"[TERM] step {step} x={x:+.1f} y={y:+.2f} z={z:+.2f} by={which}", flush=True)
                    print(f"[TERM] last z tail: {ztail}", flush=True)
                else:
                    goal_reached_flag = True
                    print(f"[GOAL] reached x={x:+.1f} y={y:+.2f} at step {step}! by={which}", flush=True)
                break

    # save trajectory
    np.savez_compressed(os.path.join(args_cli.out, "head_eval_traj.npz"),
                        traj=np.array(traj, dtype=np.float32))
    goal = goal_reached_flag or (max_x >= 145)
    print(f"\n[DONE] max_x={max_x:+.1f}  goal_reached={goal}  fell={fell}", flush=True)
    print(f"  Task A course: spawn x=-141 -> goal x>145 (~286 m).", flush=True)
    print(f"  traj -> {os.path.join(args_cli.out, 'head_eval_traj.npz')}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
