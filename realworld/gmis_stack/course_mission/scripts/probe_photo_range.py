"""Range-resolved detection probe: WHERE does the sweep-range detector lose an object?

Motivated 0826: the photo world converts close-range detection 6/6 (photodet1) yet the
mission sweep NEVER entered a lunchbox into `seen` across 6 sweeps of seed 0 (basket7,
40 vs the plain world's 60). This teleports the robot to face each target class at a
ladder of ranges, runs the real detector, and — when the class is missed — replays the
blob loop with every gate instrumented, so the failing gate is named instead of guessed.
Each pose's raw arrays are saved as npz for offline tuning (tune_detector.py workflow).

Run (photo world): MISSION_VENUE_PHOTO=1 CUDA_VISIBLE_DEVICES=7 \
  python course_mission/scripts/probe_photo_range.py --headless --device cuda:0
"""
import argparse
import sys
from pathlib import Path

_S = Path(__file__).resolve()
RW = _S.parents[2]
for _p in (str(RW / "source" / "atec_rl_lab"), str(RW / "eunha_rl"), str(RW / "a2pro_rl"), str(RW)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import os as _os
_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--speed", type=str, default="s075")
parser.add_argument("--depth", type=str, default="0.24")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--per_type", type=int, default=4)
parser.add_argument("--walk_ckpt", type=str, default="")
parser.add_argument("--grasp_ckpt", type=str, default="")
parser.add_argument("--trials", type=str, default="lunchbox:mud,lunchbox:pebble,rind:grass,can:grass")
parser.add_argument("--dists", type=str, default="0.9,1.3,1.7,2.1")
parser.add_argument("--out_dir", type=str, default=str(RW / "course_mission" / "videos_deploy" / "photo_range"))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
import isaaclab.utils.math as mu  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import course_mission  # noqa: F401, E402
import course_sim.terrain as ct  # noqa: E402
from course_mission.launch import head_cam_cfg, TASK, resolve_ckpts  # noqa: E402
from course_mission.runner import MissionRunner, Profile, ground_z_arc  # noqa: E402
from course_mission.mission import arc_of  # noqa: E402
from course_mission.perception import Perception, _blobs, _as_rot  # noqa: E402
from course_mission.catalogue import OBJ_DIMS  # noqa: E402


def gate_report(P, rgb, depth, K, cam_p, cam_R, gz, want_cls, truth_w):
    """Replay detect_arrays' blob loop for ONE class with every rejection named."""
    depth = np.asarray(depth).reshape(depth.shape[0], depth.shape[1])
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    R = _as_rot(cam_R)
    valid = np.isfinite(depth) & (depth > P.NEAR_CUT) & (depth < P.FAR_CUT)
    h, s, v = P._rgb_to_hsv(rgb)
    lab, keys = P.label_map(h, s, v)
    reports = []
    for i, cname in enumerate(keys):
        if P.PROTO_ALIAS.get(cname, cname) != want_cls:
            continue
        m = (lab == i + 1) & valid
        npx = int(m.sum())
        if npx < P.MIN_PIX:
            reports.append(f"proto '{cname}': only {npx} px labelled (MIN_PIX {P.MIN_PIX})")
            continue
        for vs, us in _blobs(m, P.MIN_PIX, min_cell=P.CELL_FILL.get(want_cls, 0)):
            vb = vs >= np.percentile(vs, 80)
            ub, vbot = us[vb].mean(), vs[vb].mean()
            ray_c = np.array([(ub - cx) / fx, (vbot - cy) / fy, 1.0]); ray_w = R @ ray_c
            pos = (cam_p + (cam_p[2] - gz) / (-ray_w[2]) * ray_w) if ray_w[2] < -1e-3 else np.full(3, np.nan)
            near = np.linalg.norm(pos[:2] - truth_w[:2]) if np.isfinite(pos).all() else 9.9
            rng = float(np.median(depth[vs, us]))
            dims = sorted(OBJ_DIMS[want_cls], reverse=True)[:2]
            exp_px = dims[0] * dims[1] * fx * fy / max(rng, 0.3) ** 2
            wp = np.full((len(vs), 3), np.nan, np.float32)
            d_ = depth[vs, us]
            wp[:, 0] = (us - cx) / fx * d_; wp[:, 1] = (vs - cy) / fy * d_; wp[:, 2] = d_
            pw = cam_p[None, :] + wp @ R.T
            plo, phi = np.nanpercentile(pw[:, :2], P.SPAN_PCT, axis=0)
            span = float(np.linalg.norm(phi - plo))
            sat_med = float(np.median(s[vs, us]))
            verdicts = []
            if span > P.CLS_MAX_SPAN.get(want_cls, P.MAX_SPAN):
                verdicts.append(f"SPAN {span:.2f}>{P.CLS_MAX_SPAN.get(want_cls, P.MAX_SPAN)}")
            if sat_med < P.MIN_BLOB_SAT.get(want_cls, 0.0):
                verdicts.append(f"SAT {sat_med:.2f}<{P.MIN_BLOB_SAT.get(want_cls, 0)}")
            lo = P.AREA_LO.get(want_cls, 0.08) * exp_px
            if not (lo <= len(vs) <= 5.0 * exp_px):
                verdicts.append(f"AREA {len(vs)}px vs [{lo:.0f},{5*exp_px:.0f}]")
            reports.append(f"blob {len(vs)}px @{near:.2f}m-from-truth rng {rng:.2f} "
                           f"sat {sat_med:.2f} span {span:.2f} -> "
                           f"{'PASS' if not verdicts else ' & '.join(verdicts)}")
    if not reports:
        # nothing even labelled: measure the hue/sat the object's TRUE pixels carry
        reports.append("0 px labelled as this class anywhere in frame")
    return reports


def main():
    device = args_cli.device or "cuda:0"
    out = Path(args_cli.out_dir); out.mkdir(parents=True, exist_ok=True)
    ck = resolve_ckpts(args_cli)
    _os.environ["MISSION_LAYOUT_SEED"] = str(args_cli.seed)
    env_cfg = parse_env_cfg(TASK, device=device, num_envs=1)
    env_cfg.per_type = args_cli.per_type
    env_cfg.scene.head_cam = head_cam_cfg()
    env = gym.make(TASK, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    env.reset()
    cams = {"head": uenv.scene["head_cam"]}
    prof = Profile(name="photorange", max_terrain_rank=3, max_picks=99)
    r = MissionRunner(env, uenv, env_cfg, prof, ck, cams, {}, out, device)
    robot = uenv.scene["robot"]

    wanted = [t.split(":") for t in args_cli.trials.split(",")]
    picked = []
    for cls, sec in wanted:
        for name, c, s_, ax, w, _z in env_cfg.mission_layout:
            if c == cls and s_ == sec and (name, cls, sec) not in picked:
                picked.append((name, cls, sec)); break
    dists = [float(x) for x in args_cli.dists.split(",")]
    print(f"[RANGE] trials: {picked} dists {dists}", flush=True)

    zero = torch.zeros(1, 3, device=device)
    with torch.inference_mode():
        for _ in range(150):
            r._write_arm(r.travel_q, open_=False)
            r._step(r.legs.step(zero))
        for name, cls, sec in picked:
            obj = uenv.scene[name]
            for dist in dists:
                op = obj.data.root_pos_w[0].cpu().numpy()
                oax, ow = arc_of(float(op[0]), float(op[1]))
                # stand back along the course tangent (the sweep's viewing geometry)
                th = float(ct.curve_theta(oax - dist))
                ex, ey = ct.curve_map(oax - dist, float(np.clip(ow, -1.0, 1.0)))
                root = torch.tensor([[float(ex), float(ey),
                                      ground_z_arc(oax - dist) + 0.42]], device=device)
                # face the object
                yaw = float(np.arctan2(op[1] - ey, op[0] - ex))
                quat = mu.quat_from_euler_xyz(torch.zeros(1, device=device),
                                              torch.zeros(1, device=device),
                                              torch.tensor([yaw], device=device))
                robot.write_root_pose_to_sim(torch.cat([root, quat], dim=1))
                robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))
                r.legs.fresh = True
                for _ in range(110):
                    r._write_arm(r.travel_q, open_=False)
                    r._step(r.legs.step(zero))
                gz = ground_z_arc(oax)
                hits = 0
                for k in range(4):
                    dets = r.percep.detect(gz)
                    if any(d["cls"] == cls and
                           np.linalg.norm(np.asarray(d["pos_w"])[:2] - op[:2]) < 0.6
                           for d in dets):
                        hits += 1
                d = r.percep.cam.data
                cam_p = (r.percep.mount.pos_w if r.percep.mount is not None
                         else d.pos_w[0].cpu().numpy())
                cam_R = (r.percep.mount.R_ros if r.percep.mount is not None
                         else d.quat_w_ros[0].cpu().numpy())
                rgb = d.output["rgb"][0, ..., :3].cpu().numpy()
                dep = d.output["distance_to_image_plane"][0].cpu().numpy()
                K = d.intrinsic_matrices[0].cpu().numpy()
                np.savez_compressed(out / f"{cls}_{sec}_{dist:.1f}.npz",
                                    rgb=rgb.astype(np.uint8), depth=dep, K=K,
                                    cam_p=cam_p, cam_R=cam_R, gz=gz, truth=op, cls=cls)
                print(f"[RANGE] {cls:9s} {sec:7s} d={dist:.1f}  hits {hits}/4", flush=True)
                if hits < 4:
                    for line in gate_report(Perception, rgb, dep, K, cam_p, cam_R,
                                            gz, cls, op):
                        print(f"[RANGE]     {line}", flush=True)
    print("[RANGE] DONE", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
