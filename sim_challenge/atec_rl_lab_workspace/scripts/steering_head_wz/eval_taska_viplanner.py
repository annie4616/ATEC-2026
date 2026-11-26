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
parser.add_argument("--task", type=str, default="ATEC-TaskA-B2Piper-MazeViPlanner")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--viplanner_model", type=str,
                    default="/hdd0/sohyun/atec_2026_tesk_e/viplanner_model/model.pt",
                    help="pretrained ViPlanner model.pt")
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
parser.add_argument("--explore_angles", type=float, nargs="+",
                    default=[0, 20, -20, 40, -40, 60, -60, 80, -80],
                    help="candidate goal bearings (deg, robot frame) probed each step; ViPlanner "
                         "fear per candidate drives map-free corridor following.")
parser.add_argument("--explore_mode", choices=["argmin", "softmin"], default="softmin",
                    help="argmin: steer fully toward the single lowest-fear opening. softmin: "
                         "fear-weighted average of candidate bearings -> smoother, stays centered.")
parser.add_argument("--explore_temp", type=float, default=0.15,
                    help="softmin temperature. Lower -> sharper (closer to argmin); higher -> "
                         "smoother blend of openings.")
parser.add_argument("--explore_fwd_bias", type=float, default=0.15,
                    help="added to fear proportional to |angle|/90, so forward is preferred when "
                         "several openings are similarly safe (prevents spinning / U-turns).")
parser.add_argument("--corner_slow_k", type=float, default=1.5,
                    help="slow vx at corners: vx *= exp(-k*max(0,front_fear-thresh)). High "
                         "front_fear = wall ahead = corner -> slow for the sharp turn.")
parser.add_argument("--corner_fear_thresh", type=float, default=0.5,
                    help="only slow when front_fear exceeds this (a real wall ahead), so open "
                         "straights keep full speed.")
parser.add_argument("--center_gain", type=float, default=0.0,
                    help="LiDAR corridor-centering gain: wz += k*(d_left - d_right) nearest-wall "
                         "distances, so the robot recenters instead of hugging a wall. Try 0.1-0.4.")
parser.add_argument("--no_backtrack", type=float, default=0.0,
                    help="penalty on candidate directions that point BACKWARD vs the recent "
                         "travel heading -> stops the map-free explorer U-turning back the way "
                         "it came. 0 = off. Try 0.5-1.5.")
parser.add_argument("--backtrack_release", type=int, default=120,
                    help="TIME WINDOW for no-backtrack: if stuck (no new ground) for this many "
                         "steps, DROP the ban so the robot can turn around when the corridor "
                         "genuinely reverses (U-turn / spiral course). Keeps the ban temporary.")
parser.add_argument("--travel_ema", type=float, default=0.05,
                    help="EMA weight for the travel-heading estimate. Small -> remembers a "
                         "longer recent history (smoother 'came-from' direction).")
parser.add_argument("--lidar_turn_gain", type=float, default=0.0,
                    help="LiDAR corner-turn: when the camera (narrow ~47deg FOV) sees a wall "
                         "dead ahead (front_fear>=corner_turn_thresh), turn hard (wz=sign*gain) "
                         "toward the LiDAR-detected open side (the 90deg opening the camera "
                         "can't see). 0 = off. Try 0.6-0.8 (near wz_max).")
parser.add_argument("--corner_turn_thresh", type=float, default=0.85,
                    help="front_fear above which the LiDAR corner-turn engages (a real wall "
                         "ahead = corridor end). Lower = engages earlier.")
parser.add_argument("--corner_open_min", type=float, default=2.5,
                    help="a side sector must have at least this much free distance (m) to count "
                         "as an 'open' turn direction for the LiDAR corner-turn.")
parser.add_argument("--corner_turn_stuck", type=int, default=15,
                    help="(legacy) unused by the LiDAR-front corner-turn.")
parser.add_argument("--corner_front_near", type=float, default=2.5,
                    help="LiDAR corner-turn engages when a wall is within this distance (m) dead "
                         "ahead (front cone median). Set to ~half the corridor length remaining "
                         "so it turns BEFORE ramming the end wall (which pushes the side opening "
                         "behind the robot). Try 2.0-3.0.")
parser.add_argument("--corner_asym_min", type=float, default=3.0,
                    help="the two side openings must differ by at least this much (m) for the "
                         "corner-turn to fire -> on a straight both sides read ~equal so we keep "
                         "going straight; only a real 1-sided corner opening triggers a turn.")
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
parser.add_argument("--dbg_explore", action="store_true",
                    help="log per-angle ViPlanner fear + chosen best angle every 200 steps, "
                         "to diagnose why a corner opening is/isn't picked.")
parser.add_argument("--fix_vx", action="store_true",
                    help="Hold vx constant at --vx (bypass corner_slow_k and vx_slow_k "
                         "multipliers). The low-level policy is being upgraded separately, so "
                         "let it handle turns; we keep forward speed steady and only steer via wz.")
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


# ViPlanner (depth + semantic -> waypoints), loaded directly from the cloned repo.
import sys as _sys  # noqa: E402
_VIPLANNER_ROOT = "/hdd0/sohyun/atec_2026_tesk_e/viplanner"
_sys.path.insert(0, _VIPLANNER_ROOT)
import torchvision.transforms as _T  # noqa: E402
from viplanner.plannernet.autoencoder import DualAutoEncoder  # noqa: E402


class _VIPCfg:
    # matches the published sem checkpoint (encoder_depth/encoder_sem = PlannerNet)
    rgb = False
    pre_train_sem = False
    pre_train_freeze = True
    decoder_small = False
    in_channel = 16
    knodes = 5


def main():
    os.makedirs(args_cli.out, exist_ok=True)
    dev = "cuda"

    # --- load pretrained ViPlanner (depth + semantic -> waypoints) ---
    vipnet = DualAutoEncoder(_VIPCfg()).to(dev).eval()
    sd, _ = torch.load(args_cli.viplanner_model, map_location=dev, weights_only=True)
    missing, unexpected = vipnet.load_state_dict(sd, strict=False)
    assert len(missing) == 0 and len(unexpected) == 0, \
        f"ViPlanner weight mismatch: missing={missing[:3]} unexpected={unexpected[:3]}"
    depth_resize = _T.Resize(tuple(args_cli.crop_size), antialias=None)
    print(f"[viplanner] loaded {args_cli.viplanner_model}  crop={args_cli.crop_size} "
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
    if args_cli.video:
        # follow the robot from behind-above so the maze corridor + corners are visible
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.eye = (-7.0, -7.0, 7.0)
        env_cfg.viewer.lookat = (2.0, 0.0, 0.0)
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

    # Label the terrain ground as "floor" so the semantic camera renders it as ViPlanner's
    # traversable color (walls/red strips are already tagged obstacle in the cfg). Without
    # this the ground is unlabeled (background) and ViPlanner sees "nowhere to go".
    try:
        from isaaclab.sim.utils import add_labels
        import isaacsim.core.utils.stage as _stage_utils
        stage = _stage_utils.get_current_stage()
        n_labeled = 0
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if path.startswith("/World/ground") and prim.GetTypeName() in ("Mesh", "Xform"):
                add_labels(prim, labels=["floor"], instance_name="class")
                n_labeled += 1
        print(f"[viplanner] labeled {n_labeled} ground prims as 'floor'", flush=True)
    except Exception as e:
        print(f"[WARN] terrain floor labeling failed: {e}", flush=True)

    try:
        lidar = env.unwrapped.scene["lidar_sensor"]
        print(f"[lidar] sensor acquired: {type(lidar).__name__} "
              f"n_rays={lidar.data.ray_hits_w.shape}", flush=True)
    except Exception as e:
        lidar = None
        print(f"[lidar] NOT available: {type(e).__name__}: {e}", flush=True)

    wz = 0.0
    front_fear = 0.0
    travel_heading = None          # smoothed world heading the robot is actually moving along
    prev_xy = None
    stuck_steps = 0                # consecutive steps with no net displacement (stuck at a wall)
    best_progress = -1e9           # farthest travel distance so far (for stuck detection)
    progress_accum = 0.0
    dbg_best_ang = 0.0
    dbg_fvals = None
    dbg_cost = None
    traj = []
    max_x = -1e9
    fell = False
    goal_reached_flag = False

    for step in range(args_cli.steps):
        with torch.inference_mode():
            if step % args_cli.every == 0:
                # --- read front DEPTH (meters) + SEMANTIC (ViPlanner-convention colors) ---
                depth = head_cam.data.output["depth"][0]            # (H,W) or (H,W,1) meters
                depth = depth.squeeze(-1) if depth.dim() == 3 else depth
                d = depth.clone().float()
                d[~torch.isfinite(d)] = 0.0
                d[d > args_cli.max_depth] = 0.0
                d = depth_resize(d.unsqueeze(0).unsqueeze(0))        # (1,1,H,W)  ViPlanner expands to 3ch

                sem = head_cam.data.output["semantic_segmentation"][0]   # (H,W,4) uint8 RGBA
                sem = sem[..., :3].permute(2, 0, 1).float()             # (3,H,W)
                sem = depth_resize(sem.unsqueeze(0))                    # (1,3,H,W)
                sem = sem / 255.0                                       # ViPlanner normalizes /255

                # --- EXPLORATION goal selection (no map, no destination) ---
                # ViPlanner's `fear` scores how unsafe a goal direction is. We probe several
                # candidate bearings (straight, +/- angles) as local goals and pick the one
                # with the LOWEST fear -> "go toward the open corridor". This turns corners
                # (straight=wall=high fear -> a side opening=low fear wins) with NO global
                # path/waypoints. A slight forward bias keeps it from spinning in place.
                angles = args_cli.explore_angles                    # e.g. [0, ±30, ±60, ±90] deg
                gx = [args_cli.goal_dist * math.cos(math.radians(t)) for t in angles]
                gy = [args_cli.goal_dist * math.sin(math.radians(t)) for t in angles]
                goals = torch.tensor([[gx[j], gy[j], 0.0] for j in range(len(angles))],
                                     device=dev, dtype=torch.float32)
                B = goals.shape[0]
                kps, fears = vipnet(d.expand(B, -1, -1, -1), sem.expand(B, -1, -1, -1), goals)
                fvals = fears.view(-1).cpu().numpy()
                ang = np.array(angles, dtype=np.float32)
                # fear of going STRAIGHT (angle closest to 0) = "is there a wall ahead" -> used
                # to slow down at corners so the low-level policy can execute the sharp turn.
                front_fear = float(fvals[int(np.argmin(np.abs(ang)))])
                # forward-bias: penalize turning away from straight so it prefers going forward
                # when several openings are similarly safe (avoids oscillation / U-turns).
                cost = fvals + args_cli.explore_fwd_bias * (np.abs(ang) / 90.0)

                # NO-BACKTRACK (time-windowed): a map-free explorer with no goal will happily
                # U-turn back the way it came (an "open" direction). We penalize candidate
                # directions pointing BACKWARD vs the recent travel heading -> keeps forward
                # progress. BUT the real course may legitimately double back (U-turn / spiral),
                # so the ban is NOT permanent: if the robot has been stuck (no forward progress)
                # for `backtrack_release` steps, we DROP the penalty this step so it can turn
                # around when the corridor really does reverse.
                banned = (args_cli.no_backtrack > 0.0 and travel_heading is not None
                          and stuck_steps < args_cli.backtrack_release)
                if banned:
                    rq = robot.data.root_quat_w[0]
                    ww, xq, yq, zq = float(rq[0]), float(rq[1]), float(rq[2]), float(rq[3])
                    yaw_w = math.atan2(2 * (ww * zq + xq * yq), 1 - 2 * (yq * yq + zq * zq))
                    cand_world = yaw_w + np.radians(ang)               # each candidate's world dir
                    diff = np.arctan2(np.sin(cand_world - travel_heading),
                                      np.cos(cand_world - travel_heading))
                    cost = cost + args_cli.no_backtrack * (np.abs(diff) / math.pi)
                if args_cli.explore_mode == "softmin":
                    # SOFT direction: weight each candidate's own waypoint bearing by exp(-cost/T)
                    # and average. Blends several safe openings -> smooth cornering + stays more
                    # centered than a hard argmin that snaps fully to one side.
                    i = min(args_cli.wp_idx, kps.shape[1] - 1)
                    bearings = torch.atan2(kps[:, i, 1],
                                           torch.clamp(kps[:, i, 0], min=1e-3)).cpu().numpy()
                    wgt = np.exp(-(cost - cost.min()) / max(args_cli.explore_temp, 1e-3))
                    wgt = wgt / wgt.sum()
                    bearing = float((wgt * bearings).sum())
                else:  # argmin: hard pick the single lowest-cost opening
                    best = int(np.argmin(cost))
                    kp = kps[best]
                    i = min(args_cli.wp_idx, kp.shape[0] - 1)
                    bearing = math.atan2(float(kp[i, 1]), max(float(kp[i, 0]), 1e-3))
                    dbg_best_ang = float(ang[best])
                    dbg_fvals = fvals.copy()
                    dbg_cost = cost.copy()
                wz_raw = args_cli.wp_gain * bearing

                # LiDAR corridor-CENTERING: measure nearest wall distance on the robot's left
                # (+90deg) vs right (-90deg); if one side is closer, nudge wz toward the open
                # side so the robot stays centered instead of hugging a wall (which is what
                # made it lose the 2nd corner). Eval-legal (raw lidar geometry). k=0 = off.
                if args_cli.center_gain != 0.0 and lidar is not None:
                    hits = lidar.data.ray_hits_w[0]
                    rp = robot.data.root_pos_w[0]
                    rq = robot.data.root_quat_w[0]
                    ww, xq, yq, zq = float(rq[0]), float(rq[1]), float(rq[2]), float(rq[3])
                    yaw_r = math.atan2(2 * (ww * zq + xq * yq), 1 - 2 * (yq * yq + zq * zq))
                    rel = hits - rp.unsqueeze(0)
                    cc, ss = math.cos(-yaw_r), math.sin(-yaw_r)
                    rxb = cc * rel[:, 0] - ss * rel[:, 1]
                    ryb = ss * rel[:, 0] + cc * rel[:, 1]
                    ab = torch.atan2(ryb, rxb)
                    rr = torch.norm(rel[:, :2], dim=1)
                    fin = torch.isfinite(rr) & (rr < 9.5)
                    lm = fin & (ab > math.radians(60)) & (ab < math.radians(120))
                    rm = fin & (ab < -math.radians(60)) & (ab > -math.radians(120))
                    d_left = float(rr[lm].min()) if lm.any() else 9.5
                    d_right = float(rr[rm].min()) if rm.any() else 9.5
                    # left farther (right wall closer) -> steer +wz (toward left/open) to recenter
                    wz_raw = wz_raw + args_cli.center_gain * (d_left - d_right)

                # LiDAR CORNER-TURN: the head camera's FOV is only ~47deg, so at the end of a
                # narrow corridor the 90-deg side opening is OUT OF the camera's view and
                # ViPlanner cannot see it (it reports the side as blocked). The 360-deg LiDAR
                # always sees it. When the camera says a wall is dead ahead (front_fear high) and
                # the robot is nearly stopped, measure the side sectors' OPENNESS and turn hard
                # toward whichever side is open -- unless that side points back the way we came
                # (no-backtrack). This is the geometric fix for the FOV limit.
                if args_cli.lidar_turn_gain != 0.0 and lidar is not None:
                    hits = lidar.data.ray_hits_w[0]
                    rp = robot.data.root_pos_w[0]
                    rq = robot.data.root_quat_w[0]
                    ww, xq, yq, zq = float(rq[0]), float(rq[1]), float(rq[2]), float(rq[3])
                    yaw_r = math.atan2(2 * (ww * zq + xq * yq), 1 - 2 * (yq * yq + zq * zq))
                    rel = hits - rp.unsqueeze(0)
                    cc, ss = math.cos(-yaw_r), math.sin(-yaw_r)
                    rxb = cc * rel[:, 0] - ss * rel[:, 1]
                    ryb = ss * rel[:, 0] + cc * rel[:, 1]
                    ab = torch.atan2(ryb, rxb)
                    rr = torch.norm(rel[:, :2], dim=1)
                    fin = torch.isfinite(rr) & (rr > 0.05)
                    # FRONT wall proximity from LiDAR (not the narrow camera): a wall dead ahead
                    # within corner_front_near means we're AT a corridor end and must turn -- and
                    # LiDAR sees it while the opening is still to the side, before the robot rams
                    # the wall (which is what pushed the side opening behind the robot last time).
                    fs = fin & (ab > -math.radians(20)) & (ab < math.radians(20))
                    front_d = float(rr[fs].median()) if fs.any() else args_cli.max_depth
                    # side OPENNESS = FARTHEST reach in a WIDE side cone (45..135). At a corridor
                    # end the side opening lets rays fly far (big); a wall caps them at ~half_w.
                    ls = fin & (ab > math.radians(45)) & (ab < math.radians(135))    # left
                    rs = fin & (ab < -math.radians(45)) & (ab > -math.radians(135))  # right
                    open_l = float(rr[ls].max()) if ls.any() else 0.0
                    open_r = float(rr[rs].max()) if rs.any() else 0.0

                    def _faces_back(turn_sign):
                        if travel_heading is None or stuck_steps >= args_cli.backtrack_release:
                            return False
                        cand = yaw_r + turn_sign * math.radians(90.0)
                        diff = math.atan2(math.sin(cand - travel_heading),
                                          math.cos(cand - travel_heading))
                        return abs(diff) > math.radians(120.0)   # >120deg off travel = backward

                    # only a CORNER turn: a wall ahead (front_d small) AND the two sides are
                    # ASYMMETRIC (one clearly more open). On a straight both sides read ~equal
                    # (open_l ~= open_r) so the asymmetry gate keeps us going straight; at a real
                    # corner only one side opens, so |open_l-open_r| is large and we turn to it.
                    fired = False
                    if front_d <= args_cli.corner_front_near \
                            and abs(open_l - open_r) >= args_cli.corner_asym_min:
                        cand, vetoed = [], []
                        if open_l >= args_cli.corner_open_min:
                            (cand if not _faces_back(+1) else vetoed).append((open_l, +1))
                        if open_r >= args_cli.corner_open_min:
                            (cand if not _faces_back(-1) else vetoed).append((open_r, -1))
                        if cand:
                            cand.sort(reverse=True)                # most-open side wins
                            wz_raw = cand[0][1] * args_cli.lidar_turn_gain
                            fired = True
                    if step % 200 == 0:
                        print(f"    [lidar-turn] step {step} front_d={front_d:.1f} "
                              f"openL={open_l:.1f} openR={open_r:.1f} "
                              f"fired={'Y' if fired else 'N'} stuck={stuck_steps}", flush=True)

                if args_cli.kd > 0.0:
                    yaw_rate_b = float(robot.data.root_ang_vel_b[0, 2])
                    wz_raw = wz_raw - args_cli.kd * yaw_rate_b
                wz_raw = max(-args_cli.wz_max, min(args_cli.wz_max, wz_raw))
                wz = args_cli.wz_ema * wz_raw + (1.0 - args_cli.wz_ema) * wz

            # CORNER slowdown: when a wall is ahead (high front_fear) the robot is at a corner
            # and the low-level velocity policy can't take a sharp 90-deg turn at full speed --
            # it rams the wall. Cut vx by exp(-k*front_fear) so it nearly turns in place, giving
            # the sharp turn time to complete. Plus the TOP-Nav anti-drift on |wz|.
            # only slow once front_fear crosses a threshold (a real wall ahead), so open
            # straights keep full speed and only actual corners slow down.
            ff_excess = max(0.0, front_fear - args_cli.corner_fear_thresh)
            if args_cli.fix_vx:
                vx_cmd = args_cli.vx
            else:
                vx_cmd = args_cli.vx * math.exp(-args_cli.vx_slow_k * abs(wz)) \
                    * math.exp(-args_cli.corner_slow_k * ff_excess)

            solution.fixed_velocity_commands = torch.tensor(
                [vx_cmd, 0.0, wz], device=solution.device, dtype=torch.float32).view(1, 3)

            resp = solution.predicts(obs, 0.0)
            act = (torch.zeros((1, 20), device=dev) if (resp.get("giveup") or not resp["action"])
                   else torch.tensor(resp["action"], dtype=torch.float32, device=dev).view(1, -1))
            obs, _, term, trunc, _ = env.step(act)

            p = robot.data.root_pos_w[0]
            x, y, z = float(p[0]), float(p[1]), float(p[2])
            max_x = max(max_x, x)

            # update the smoothed travel heading from actual world displacement (only when the
            # robot is really moving, so slow turns-in-place don't corrupt it).
            if prev_xy is not None:
                dxs, dys = x - prev_xy[0], y - prev_xy[1]
                disp = math.hypot(dxs, dys)
                if disp > 1e-3:
                    h = math.atan2(dys, dxs)
                    if travel_heading is None:
                        travel_heading = h
                    else:
                        a = args_cli.travel_ema
                        travel_heading = math.atan2(
                            a * math.sin(h) + (1 - a) * math.sin(travel_heading),
                            a * math.cos(h) + (1 - a) * math.cos(travel_heading))
                # stuck detection: accumulate displacement; if it doesn't grow, we're stuck at
                # a wall/corner and the no-backtrack ban should time out so the robot can turn
                # around (in case the corridor genuinely reverses).
                progress_accum += disp
                if progress_accum > best_progress + 0.5:   # moved 0.5 m of new ground
                    best_progress = progress_accum
                    stuck_steps = 0
                else:
                    stuck_steps += 1
            prev_xy = (x, y)
            if step % args_cli.every == 0:
                traj.append((step, x, y, wz, z))
            if step % 200 == 0:
                extra = ""
                if args_cli.dbg_explore and dbg_fvals is not None:
                    fv = " ".join(f"{int(a):+d}:{f:.2f}" for a, f in
                                  zip(args_cli.explore_angles, dbg_fvals))
                    extra = (f" | best_ang={dbg_best_ang:+.0f} front_fear={front_fear:.2f}"
                             f" stuck={stuck_steps} fear[{fv}]")
                print(f"[step {step:4d}] x={x:+.1f} y={y:+.2f} z={z:+.2f} wz={wz:+.2f} "
                      f"max_x={max_x:+.1f}{extra}", flush=True)

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
