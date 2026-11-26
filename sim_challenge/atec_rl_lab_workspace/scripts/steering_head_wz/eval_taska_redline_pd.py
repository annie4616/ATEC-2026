"""TASK A RED-LINE FOLLOWING with a COLOR-DETECT + GEOMETRIC PD controller (RGB only).

Real courses have no walls -- just a red boundary strip. This drives the robot by looking
at the head-camera RGB, segmenting the red pixels (HSV), and steering to keep the red-line
corridor centered and aligned. No ViPlanner, no LiDAR, no learning: a transparent baseline
for "can RGB alone follow the red line through 90-deg bends?".

Steering (image-space geometric PD):
  * Red pixels -> mask. For a set of image ROWS (near..far), take the mean column of red
    pixels in that row = the corridor center at that lookahead.
  * cross-track error  = (near-row center - image center) -> robot is off to one side.
  * heading error      = (far-row center - near-row center) -> the corridor bends ahead.
  * wz = -(k_ct * cross_track + k_hd * heading), clipped; forward vx constant.

  CUDA_VISIBLE_DEVICES=6 TMPDIR=/hdd0/sohyun/tmp PYTHONPATH=/hdd0/sohyun \
    python scripts/steering_head_wz/eval_taska_redline_pd.py --headless \
      --task ATEC-TaskA-B2Piper-RedLineCornerViPlanner --steps 6000 \
      --out /hdd0/sohyun/atec_2026_tesk_e/task_a_real_exp/redline_pd
"""
import argparse
import os
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Task A red-line following via color-detect PD.")
parser.add_argument("--task", type=str, default="ATEC-TaskA-B2Piper-RedLineCornerViPlanner")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--vx", type=float, default=1.0)
parser.add_argument("--fix_vx", action="store_true", default=True)
parser.add_argument("--wz_max", type=float, default=0.8)
parser.add_argument("--k_ct", type=float, default=1.6,
                    help="cross-track gain: how hard to recenter when the corridor center is "
                         "off to one side of the image.")
parser.add_argument("--k_hd", type=float, default=2.2,
                    help="heading gain: how hard to pre-turn when the corridor bends ahead "
                         "(far row center vs near row center).")
parser.add_argument("--wz_ema", type=float, default=0.6,
                    help="EMA smoothing on wz (a=1 -> raw).")
parser.add_argument("--near_row", type=float, default=0.12,
                    help="image row (fraction of height, 0=top) used for cross-track. The red "
                         "strips appear near the TOP of the image (they recede into the "
                         "distance), so both sample rows are high up.")
parser.add_argument("--far_row", type=float, default=0.03,
                    help="image row (fraction) used for heading (farther ahead = higher up).")
parser.add_argument("--red_s_min", type=float, default=0.35, help="HSV min saturation for red")
parser.add_argument("--red_v_min", type=float, default=0.25, help="HSV min value for red")
parser.add_argument("--band", type=int, default=8, help="+/- rows averaged around each sample row")
parser.add_argument("--min_red_px", type=int, default=15,
                    help="if fewer red pixels than this in a row band, treat as 'no line seen'")
# --- line-follow (parallel + bend) controller gains ---
parser.add_argument("--ctrl", type=str, default="linefollow", choices=["linefollow", "center"],
                    help="'linefollow' (fit the red line -> keep parallel + turn at bends) or "
                         "'center' (old L/R strip-midpoint PD).")
parser.add_argument("--k_lat", type=float, default=1.2,
                    help="[linefollow] lateral gain: recenter on the line's image position.")
parser.add_argument("--k_ang", type=float, default=1.6,
                    help="[linefollow] alignment gain: null the line's image tilt (go parallel).")
parser.add_argument("--k_bend", type=float, default=2.0,
                    help="[linefollow] bend gain: turn toward the direction the line bends ahead.")
parser.add_argument("--split_row", type=float, default=0.5,
                    help="[linefollow] split the red pixels above/below this row fraction into a "
                         "FAR half (top) and NEAR half (bottom) to measure the bend between them.")
parser.add_argument("--bend_clip", type=float, default=1.0,
                    help="[linefollow] clamp |bend| to this before applying k_bend, so a single "
                         "extreme frame at a corner can't slam wz to saturation and overshoot.")
parser.add_argument("--ang_clip", type=float, default=1.0,
                    help="[linefollow] clamp |ang| likewise (sparse corner frames blow up tilt).")
parser.add_argument("--lost_decay", type=float, default=0.90,
                    help="[linefollow] when the line is momentarily lost, decay wz by this factor "
                         "each step toward the SEEK value instead of holding it (prevents the "
                         "robot from spinning in place on the last corner command).")
parser.add_argument("--lost_seek", type=float, default=0.25,
                    help="[linefollow] while the line is lost, steer this hard TOWARD the side "
                         "where it was last seen (sign from last lat) to re-acquire it.")
parser.add_argument("--lost_max", type=int, default=40,
                    help="[linefollow] after this many consecutive lost frames, stop seeking and "
                         "just go straight (avoid runaway when the line truly isn't there).")
parser.add_argument("--corner_vx", type=float, default=0.45,
                    help="[linefollow] forward speed WHILE turning through a corner. Slowing down "
                         "in the turn shortens the arc so the robot doesn't overshoot the narrow "
                         "corridor and fall off the map. Full vx on straights.")
parser.add_argument("--corner_turn_thresh", type=float, default=0.35,
                    help="[linefollow] |wz| above which we consider the robot to be cornering and "
                         "apply corner_vx. Below it, ramp back up to full vx.")
parser.add_argument("--vx_ramp", type=float, default=0.08,
                    help="[linefollow] per-step slew on vx so speed changes are smooth, not a "
                         "step (a sudden vx drop can itself destabilize the gait).")
parser.add_argument("--steps", type=int, default=6000)
parser.add_argument("--seed", type=int, default=3)
parser.add_argument("--no_illegal_contact", action="store_true", default=False)
parser.add_argument("--dbg", action="store_true", help="log per-step detection every 200 steps")
parser.add_argument("--dbg_every", type=int, default=0,
                    help="if >0, log cn/cf/kind/wz EVERY N steps (fine-grained corner diagnosis)")
parser.add_argument("--dump_rgb", type=int, default=0,
                    help="if >0, save the head RGB + red mask as PNG at this step and also "
                         "print the per-row red-pixel count profile, then exit. Diagnoses "
                         "WHERE the red line sits in the image (which rows to sample).")
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=6000)
parser.add_argument("--headcam_video", action="store_true", default=False,
                    help="save an mp4 of the HEAD CAMERA view with detection overlay (red mask "
                         "tint, sample rows, detected centers, wz bar) -- shows what the robot "
                         "sees and how it decides to steer.")
parser.add_argument("--headcam_every", type=int, default=2,
                    help="record a head-cam frame every N steps (2 -> ~half real-time size).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
import torch
import numpy as np
import gymnasium as gym
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
import atec_rl_lab.tasks  # noqa: F401
from atec_rl_lab.tasks.task_base.action_base import apply_safe_action_spec

# low-level fast policy (same one the ViPlanner eval uses)
import sys
sys.path.insert(0, "/hdd0/sohyun")
from atec_2026_tesk_e.ATEC2026_Simulation_Challenge.demo.solution_rl import AlgSolution  # noqa

solution = AlgSolution()


def rgb_to_hsv(rgb):
    """rgb: (H,W,3) float in [0,1] -> h,s,v each (H,W). Vectorized torch."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, _ = rgb.max(dim=-1)
    mn, _ = rgb.min(dim=-1)
    diff = mx - mn + 1e-8
    # hue in [0,1)
    h = torch.zeros_like(mx)
    mask = mx == r
    h[mask] = ((g[mask] - b[mask]) / diff[mask]) % 6.0
    mask = mx == g
    h[mask] = (b[mask] - r[mask]) / diff[mask] + 2.0
    mask = mx == b
    h[mask] = (r[mask] - g[mask]) / diff[mask] + 4.0
    h = h / 6.0
    s = diff / (mx + 1e-8)
    v = mx
    return h, s, v


def red_mask(rgb):
    """rgb (H,W,3) in [0,1] -> bool mask of red pixels. Red hue wraps at 0/1."""
    h, s, v = rgb_to_hsv(rgb)
    is_red_hue = (h < 0.05) | (h > 0.95)
    return is_red_hue & (s > args_cli.red_s_min) & (v > args_cli.red_v_min)


def _fit_line_xofr(rows, cols):
    """Fit column = a*row + b (least squares) over the red pixels. Returns (a, b, ok).
    `a` = dcol/drow = how much the line shifts sideways per row DOWN the image. In head-cam
    coords the line recedes upward, so `a` is the line's image tilt: a≈0 means the strip runs
    straight up/down the image (robot parallel to it); a>0 means it leans so lower rows are to
    the right. Robust enough for the near-straight red strips we see."""
    if rows.numel() < 8:
        return 0.0, 0.0, False
    r = rows.float()
    c = cols.float()
    rm = r.mean()
    cm = c.mean()
    dr = r - rm
    denom = float((dr * dr).sum())
    if denom < 1e-6:
        return 0.0, float(cm), True          # vertical-ish: constant column
    a = float((dr * (c - cm)).sum()) / denom
    b = float(cm - a * rm)
    return a, b, True


def linefollow_signal(mask, W, H):
    """Line-following controller from the red pixels (works with ONE strip or two).

    Returns (lat, ang, bend, ok, dbg):
      * lat  : line's horizontal position at mid-image, normalized [-1,1] (0=centered).
               Keeps the robot laterally on/near the line.
      * ang  : line's image tilt (its column-slope over rows), signed. 0 => the strip runs
               straight up the image => robot is PARALLEL to the line. Nulling this = go parallel.
      * bend : FAR-half tilt minus NEAR-half tilt. Nonzero => the line curves ahead (a corner);
               its sign is the turn direction. This is what makes 90-deg bends symmetric L/R.
    All three are computed from the SAME red pixels, so a single visible strip still yields a
    consistent parallel-follow + bend command (no L/R-midpoint asymmetry)."""
    idx = mask.nonzero(as_tuple=False)               # (N,2) [row, col]
    if idx.shape[0] < args_cli.min_red_px:
        return 0.0, 0.0, 0.0, False, "none"
    rows = idx[:, 0]
    cols = idx[:, 1]
    mid = (W - 1) / 2.0

    # lateral: where the line sits horizontally, at the vertical center of the red pixels.
    a_all, b_all, ok = _fit_line_xofr(rows, cols)
    r_ref = float(rows.float().mean())
    col_at_ref = a_all * r_ref + b_all
    lat = (col_at_ref - mid) / mid
    # alignment (tilt) as an ANGLE, not a raw slope: atan of (dcol/drow) is inherently bounded
    # to [-pi/2, pi/2] no matter how few/noisy the pixels, then scaled to ~[-1,1] at 45 deg.
    # (A raw slope * H/2 exploded to +-5 on sparse corner frames and slammed wz to saturation.)
    ang = math.atan(a_all) / (math.pi / 4.0)

    # bend: split into FAR (top) and NEAR (bottom) halves; compare only their HORIZONTAL center
    # positions (both bounded to [-1,1]). If the FAR segment sits to one side of the NEAR one,
    # the line curves that way ahead. We DROP the tilt-difference term -- it was the blow-up
    # source (two noisy slopes subtracted, then amplified).
    split = args_cli.split_row * (H - 1)
    far_m = rows.float() < split
    near_m = ~far_m
    bend = 0.0
    if int(far_m.sum()) >= 8 and int(near_m.sum()) >= 8:
        cf = float(cols[far_m].float().mean())
        cn = float(cols[near_m].float().mean())
        bend = (cf - cn) / mid                       # bounded to ~[-2,2]
    dbg = f"lat={lat:+.2f} ang={ang:+.2f} bend={bend:+.2f} n={idx.shape[0]}"
    return lat, ang, bend, True, dbg


def corridor_center_at_row(mask, row_frac, band, W, H):
    """Corridor center at a horizontal band around row_frac, from the LEFT and RIGHT red
    boundary lines separately. The two red strips appear on either side of the image; their
    midpoint is the corridor center. Returns center normalized to [-1,1] (0=image center,
    +1=right edge), or None if neither line is seen.

    In a narrow corridor only ONE strip may be in view when the robot drifts; if so we infer
    the center from that single strip assuming it should sit near the image edge."""
    r = int(round(row_frac * (H - 1)))
    r0 = max(0, r - band)
    r1 = min(H, r + band + 1)
    sub = mask[r0:r1]                          # (rows, W)
    cols = sub.nonzero(as_tuple=False)         # (N,2) [row, col]
    if cols.shape[0] < args_cli.min_red_px:
        return None
    col = cols[:, 1].float()
    mid = (W - 1) / 2.0
    left = col[col < mid]
    right = col[col >= mid]
    have_l = left.numel() >= args_cli.min_red_px // 2
    have_r = right.numel() >= args_cli.min_red_px // 2
    if have_l and have_r:
        center = (float(left.mean()) + float(right.mean())) / 2.0
        kind = "LR"
    elif have_l:
        # only the LEFT strip visible -> robot drifted right; center is to the LEFT of it.
        center = float(left.mean()) - 0.30 * W
        kind = "L"
    elif have_r:
        center = float(right.mean()) + 0.30 * W
        kind = "R"
    else:
        center = float(col.mean())
        kind = "?"
    corridor_center_at_row.last_kind = kind
    return (center - mid) / mid


def main():
    os.makedirs(args_cli.out, exist_ok=True)
    from isaaclab_tasks.utils import parse_env_cfg
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    action_spec = solution.get_action_spec() if hasattr(solution, "get_action_spec") else None
    env_cfg = apply_safe_action_spec(env_cfg, json.dumps(action_spec))
    if args_cli.no_illegal_contact:
        try:
            env_cfg.terminations.illegal_contact = None
            print("[cfg] illegal_contact termination DISABLED", flush=True)
        except Exception:
            pass

    render_mode = "rgb_array" if args_cli.video else None
    if args_cli.video:
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
            env, video_folder=video_dir, name_prefix="taska_redline",
            step_trigger=lambda s: s == 0, video_length=args_cli.video_length,
            disable_logger=True)
        print(f"[video] recording to {video_dir}", flush=True)

    import random as _random
    _random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    torch.cuda.manual_seed_all(args_cli.seed)
    obs, _ = env.reset(seed=args_cli.seed)
    robot = env.unwrapped.scene["robot"]
    head_cam = env.unwrapped.scene["head_camera"]
    dev = solution.device

    wz = 0.0
    traj = []
    max_x = -1e9
    fell = False
    goal_reached_flag = False
    last_center = 0.0
    last_lat = 0.0          # last lateral position of the line (for lost-line seek direction)
    lost_count = 0          # consecutive frames with no line
    vx_cur = args_cli.vx    # slewed forward speed (drops in corners)
    headcam_frames = [] if args_cli.headcam_video else None

    for step in range(args_cli.steps):
        with torch.inference_mode():
            rgb = head_cam.data.output["rgb"][0]                # (H,W,3 or 4) uint8
            rgb = rgb[..., :3].float() / 255.0                  # (H,W,3) in [0,1]
            H, W = rgb.shape[0], rgb.shape[1]
            mask = red_mask(rgb)

            if args_cli.dump_rgb > 0 and step == args_cli.dump_rgb:
                import imageio.v2 as imageio
                rgb_np = (rgb.cpu().numpy() * 255).astype(np.uint8)
                m_np = (mask.cpu().numpy() * 255).astype(np.uint8)
                imageio.imwrite(os.path.join(args_cli.out, "dbg_rgb.png"), rgb_np)
                imageio.imwrite(os.path.join(args_cli.out, "dbg_redmask.png"), m_np)
                row_counts = mask.sum(dim=1).cpu().numpy()   # red px per row
                print(f"[dump] step {step} img {H}x{W} total_red={int(mask.sum())}", flush=True)
                print("[dump] per-row red counts (row_frac: count) where count>5:", flush=True)
                for r in range(0, H, max(1, H // 24)):
                    print(f"    row {r:3d} ({r/(H-1):.2f}): {int(row_counts[r])}", flush=True)
                env.close(); simulation_app.close(); return

            if args_cli.ctrl == "linefollow":
                lat, ang, bend, ok_lf, lf_dbg = linefollow_signal(mask, W, H)
                c_near = None; c_far = None; kind_n = kind_f = "-"
                if not ok_lf:
                    lost_count += 1
                    if lost_count <= args_cli.lost_max:
                        # line lost (usually a corner where it swept out of the FOV). Steer TOWARD
                        # the side it was last on to re-acquire, and decay the previous wz so we
                        # don't keep spinning on a stale corner command.
                        seek = -args_cli.lost_seek * (1.0 if last_lat > 0 else -1.0)
                        wz_raw = args_cli.lost_decay * wz + (1.0 - args_cli.lost_decay) * seek
                    else:
                        wz_raw = args_cli.lost_decay * wz    # truly gone -> straighten out
                    seen = f"none(lost{lost_count})"
                else:
                    lost_count = 0
                    # keep parallel (null tilt) + recenter (lat) + turn into the bend.
                    # a red strip to the RIGHT of image center (lat>0) means the line/corridor
                    # is to our right -> steer right (wz<0). Same sign convention for ang/bend.
                    bend_c = max(-args_cli.bend_clip, min(args_cli.bend_clip, bend))
                    ang_c = max(-args_cli.ang_clip, min(args_cli.ang_clip, ang))
                    wz_raw = -(args_cli.k_lat * lat + args_cli.k_ang * ang_c
                               + args_cli.k_bend * bend_c)
                    last_center = lat
                    last_lat = lat
                    seen = lf_dbg
                # expose the fit for the overlay
                linefollow_signal.last = (lat, ang, bend) if ok_lf else None
            else:
                c_near = corridor_center_at_row(mask, args_cli.near_row, args_cli.band, W, H)
                kind_n = getattr(corridor_center_at_row, "last_kind", "?")
                c_far = corridor_center_at_row(mask, args_cli.far_row, args_cli.band, W, H)
                kind_f = getattr(corridor_center_at_row, "last_kind", "?")
                if c_near is None and c_far is None:
                    wz_raw = wz
                    seen = "none"
                else:
                    cn = c_near if c_near is not None else c_far
                    cf = c_far if c_far is not None else c_near
                    cross_track = cn
                    heading = cf - cn
                    wz_raw = -(args_cli.k_ct * cross_track + args_cli.k_hd * heading)
                    last_center = cn
                    seen = f"n={cn:+.2f} f={cf:+.2f}"

            wz_raw = max(-args_cli.wz_max, min(args_cli.wz_max, wz_raw))
            wz = args_cli.wz_ema * wz_raw + (1.0 - args_cli.wz_ema) * wz

            # head-cam frame with detection overlay: red mask tint + sample-row lines +
            # detected corridor centers (near=cyan dot, far=yellow dot) + wz bar.
            if headcam_frames is not None and step % args_cli.headcam_every == 0:
                fr = (rgb.cpu().numpy() * 255).astype(np.uint8).copy()   # (H,W,3)
                m = mask.cpu().numpy()
                fr[m] = (fr[m] * 0.4 + np.array([255, 60, 60]) * 0.6).astype(np.uint8)  # tint red px
                nr = int(round(args_cli.near_row * (H - 1)))
                far = int(round(args_cli.far_row * (H - 1)))
                fr[max(0, nr - 1):nr + 2, :, :] = [0, 200, 255]         # near row = cyan line
                fr[max(0, far - 1):far + 2, :, :] = [255, 230, 0]        # far row  = yellow line
                mid = (W - 1) / 2.0
                fr[:, int(mid) - 1:int(mid) + 2, :] = [180, 180, 180]    # image center = gray
                if args_cli.ctrl == "linefollow":
                    lf = getattr(linefollow_signal, "last", None)
                    if lf is not None:
                        lat_o, ang_o, bend_o = lf
                        # cyan vertical marker = line lateral position at mid image
                        cx = int(mid + lat_o * mid)
                        fr[:, max(0, cx - 1):cx + 2, :] = [0, 255, 255]
                        # yellow tick at top = bend direction (right if >0)
                        bx = int(mid + max(-1, min(1, bend_o)) * mid)
                        fr[2:12, max(0, bx - 3):bx + 4, :] = [255, 255, 0]
                elif c_near is not None:
                    cx = int(mid + c_near * mid)
                    fr[max(0, nr - 6):nr + 6, max(0, cx - 3):cx + 4, :] = [0, 255, 255]
                    if c_far is not None:
                        cx = int(mid + c_far * mid)
                        fr[max(0, far - 6):far + 6, max(0, cx - 3):cx + 4, :] = [255, 255, 0]
                # wz bar along the bottom: green right / magenta left, length ~ |wz|
                bar_len = int(abs(wz) / max(args_cli.wz_max, 1e-3) * (W // 2))
                col = [80, 255, 80] if wz < 0 else [255, 80, 255]       # wz<0 = steer right
                x0 = int(mid) if wz < 0 else int(mid) - bar_len
                fr[H - 10:H - 2, max(0, x0):min(W, x0 + bar_len), :] = col
                headcam_frames.append(fr)

            # corner slow-down: if we're turning hard (|wz| large) OR a bend is detected ahead,
            # target corner_vx; else full vx. Slew toward the target so the gait stays stable.
            cornering = (abs(wz) >= args_cli.corner_turn_thresh)
            if args_cli.ctrl == "linefollow":
                lf_now = getattr(linefollow_signal, "last", None)
                if lf_now is not None and abs(lf_now[2]) >= 0.5:   # bend ahead
                    cornering = True
            vx_target = args_cli.corner_vx if cornering else args_cli.vx
            if vx_cur < vx_target:
                vx_cur = min(vx_target, vx_cur + args_cli.vx_ramp)
            else:
                vx_cur = max(vx_target, vx_cur - args_cli.vx_ramp)
            vx_cmd = vx_cur
            solution.fixed_velocity_commands = torch.tensor(
                [vx_cmd, 0.0, wz], device=dev, dtype=torch.float32).view(1, 3)

            resp = solution.predicts(obs, 0.0)
            act = (torch.zeros((1, 20), device=dev) if (resp.get("giveup") or not resp["action"])
                   else torch.tensor(resp["action"], dtype=torch.float32, device=dev).view(1, -1))
            obs, _, term, trunc, _ = env.step(act)

            p = robot.data.root_pos_w[0]
            x, y, z = float(p[0]), float(p[1]), float(p[2])
            max_x = max(max_x, x)
            traj.append((step, x, y, wz, z))
            if step % 200 == 0:
                red_n = int(mask.sum())
                extra = f" | {seen} red_px={red_n}" if args_cli.dbg else ""
                print(f"[step {step:4d}] x={x:+.1f} y={y:+.2f} z={z:+.2f} wz={wz:+.2f} "
                      f"max_x={max_x:+.1f}{extra}", flush=True)
            if args_cli.dbg_every > 0 and step % args_cli.dbg_every == 0:
                if args_cli.ctrl == "linefollow":
                    print(f"  <dbg {step:4d}> x={x:+.1f} y={y:+.2f} | {seen} -> wz={wz:+.2f}",
                          flush=True)
                else:
                    cn_s = f"{c_near:+.2f}" if c_near is not None else "None"
                    cf_s = f"{c_far:+.2f}" if c_far is not None else "None"
                    print(f"  <dbg {step:4d}> x={x:+.1f} y={y:+.2f} | "
                          f"near={cn_s}[{kind_n}] far={cf_s}[{kind_f}] -> wz={wz:+.2f}", flush=True)

            if bool(term[0]):
                if z < 0.2:
                    fell = True
                which = "fell" if fell else "goal/term"
                print(f"[TERM] step {step} x={x:+.1f} y={y:+.2f} z={z:+.2f} by={which}", flush=True)
                if not fell:
                    goal_reached_flag = True
                break

    np.savez(os.path.join(args_cli.out, "redline_pd_traj.npz"),
             traj=np.array(traj, dtype=np.float32))
    if headcam_frames:
        import imageio.v2 as imageio
        vp = os.path.join(args_cli.out, "headcam_view.mp4")
        imageio.mimwrite(vp, headcam_frames, fps=20, macro_block_size=None)
        print(f"[headcam] wrote {vp} ({len(headcam_frames)} frames)", flush=True)
    print(f"[DONE] max_x={max_x:+.1f}  goal_reached={goal_reached_flag}  fell={fell}", flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
