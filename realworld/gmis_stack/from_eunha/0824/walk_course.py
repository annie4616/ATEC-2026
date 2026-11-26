"""Drive the Eunha RMA walk policy along the competition course and record a video.

Two camera modes: --follow (viewer tracks the robot root) or overview (static wide).
Legs run the 53-obs walk policy (manual obs build + frozen adaptation latent, action
rescaled from the policy's 0.125/0.25 convention to the task env's 0.5 joint action);
the arm holds its default pose. Heading is kept along +x via a yaw-proportional
ang-vel command (mimics the training-time heading controller).

Run (GPU5, briefly):
  EUNHA_ADAPT_CKPT=from_eunha/speed_1.0/adaptation.pt CUDA_VISIBLE_DEVICES=5 \
  ~/.conda/envs/isaaclab/bin/python course_sim/scripts/walk_course.py --headless \
      --device cuda:0 --walk_ckpt from_eunha/speed_1.0/model_14999.pt --follow
"""

import argparse
import os
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve()
RW = _SCRIPT.parents[2]
for _p in (str(RW / "source" / "atec_rl_lab"), str(RW), str(RW / "eunha_rl")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Walk the course with the Eunha policy.")
parser.add_argument("--task", type=str, default="ATEC-Course-A2Piper")
parser.add_argument("--walk_ckpt", type=str, required=True)
parser.add_argument("--vx", type=float, default=0.7)
parser.add_argument("--steps", type=int, default=3500)
parser.add_argument("--out_dir", type=str, default=str(RW / "course_sim" / "videos"))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
assert os.environ.get("EUNHA_ADAPT_CKPT"), "EUNHA_ADAPT_CKPT must point at adaptation.pt"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import course_sim  # noqa: F401, E402
from course_sim import terrain as ct  # noqa: E402
from epkg import rma  # noqa: E402
import numpy as np  # noqa: E402

LEG_ORDER = rma.LEG_ORDER


def main():
    device = args_cli.device if getattr(args_cli, "device", None) else "cuda:0"
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=1)
    # continuous footage: never reset on stumble
    for attr in list(vars(env_cfg.terminations)):
        if attr != "time_out":
            setattr(env_cfg.terminations, attr, None)
    env_cfg.episode_length_s = 10000.0

    # two scene cameras -> BOTH videos from ONE run
    from isaaclab.sensors import CameraCfg
    cam_common = dict(
        update_period=0.0, height=720, width=1280, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.1, 200.0)),
    )
    env_cfg.scene.overview_cam = CameraCfg(prim_path="/World/overview_cam", **cam_common)
    env_cfg.scene.follow_cam = CameraCfg(prim_path="/World/follow_cam", **cam_common)

    # GUIDE_NAV: LiDAR "walk parallel to the red belt" steering (ref: sohyun/course_student_a2).
    # A dedicated raycaster that hits ONLY the red boundary prims (belt webbing + posts), so we
    # get clean rail returns to fit a heading from -- robust to body pitch, unlike the head cam
    # which points at sky on climbs. We do NOT centre between rails; we hold parallel + a set
    # distance to the NEAR rail, so a lone belt is followed alongside, never walked into.
    _guide_nav = bool(os.environ.get("GUIDE_NAV"))
    if _guide_nav:
        from isaaclab.sensors import MultiMeshRayCasterCfg
        _base = getattr(env_cfg.scene, "lidar_sensor", None)
        if _base is not None:
            # REALISTIC_LIDAR=1: raycast the WHOLE scene (ground + belt + posts + trash), exactly
            # what a real LiDAR returns -- so _detect_rail must isolate the belt itself via the
            # height/geometry filter (no privileged belt-only segmentation). =0 keeps the
            # belt-only raycaster (sim-assisted segmentation, for A/B comparison).
            if int(os.environ.get("REALISTIC_LIDAR", "1")):
                _targets = [
                    MultiMeshRayCasterCfg.RaycastTargetCfg(
                        prim_expr="/World/ground/terrain_.*", track_mesh_transforms=False),
                    MultiMeshRayCasterCfg.RaycastTargetCfg(
                        prim_expr="/World/ground/strip_.*", track_mesh_transforms=False),
                    MultiMeshRayCasterCfg.RaycastTargetCfg(
                        prim_expr="/World/ground/belt_.*", track_mesh_transforms=False),
                    MultiMeshRayCasterCfg.RaycastTargetCfg(
                        prim_expr="/World/envs/env_0/Trash.*", track_mesh_transforms=True),
                ]
                print("[WALK] REALISTIC_LIDAR: rail_lidar sees FULL scene (belt via height filter)", flush=True)
            else:
                _targets = [
                    MultiMeshRayCasterCfg.RaycastTargetCfg(
                        prim_expr="/World/ground/strip_.*", track_mesh_transforms=False),
                    MultiMeshRayCasterCfg.RaycastTargetCfg(
                        prim_expr="/World/ground/belt_.*", track_mesh_transforms=False),
                ]
                print("[WALK] rail_lidar sees belt/strip prims only (privileged segmentation)", flush=True)
            env_cfg.scene.rail_lidar = MultiMeshRayCasterCfg(
                prim_path=_base.prim_path, update_period=_base.update_period,
                pattern_cfg=_base.pattern_cfg, max_distance=_base.max_distance,
                debug_vis=False, mesh_prim_paths=_targets,
            )
        else:
            print("[WALK] GUIDE_NAV: env has no lidar_sensor; cannot add rail_lidar", flush=True)

    os.makedirs(args_cli.out_dir, exist_ok=True)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env.reset()
    uenv = env.unwrapped

    import imageio.v2 as iio
    ov = uenv.scene["overview_cam"]
    fc = uenv.scene["follow_cam"]
    eye = torch.tensor([[0.0, -36.0, 24.0]], device=uenv.device)
    tgt = torch.tensor([[0.0, 2.0, 0.0]], device=uenv.device)
    ov.set_world_poses_from_view(eye, tgt)
    w_ov = iio.get_writer(os.path.join(args_cli.out_dir, "walk_overview.mp4"), fps=50)
    w_fc = iio.get_writer(os.path.join(args_cli.out_dir, "walk_follow.mp4"), fps=50)
    # onboard robot head camera (RGB) -- what the real robot actually sees, for belt line-following
    _cam_nav = bool(os.environ.get("CAMERA_NAV"))
    _hc = uenv.scene["head_camera"] if (os.environ.get("HEAD_CAM") or _cam_nav) else None
    w_hc = iio.get_writer(os.path.join(args_cli.out_dir, "head_cam.mp4"), fps=50) if os.environ.get("HEAD_CAM") else None
    _bnav = None
    if _cam_nav:
        import importlib.util as _u
        _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "belt_nav.py")
        _s = _u.spec_from_file_location("belt_nav", _p); _bnav = _u.module_from_spec(_s); _s.loader.exec_module(_bnav)
    _prev_wz = 0.0
    _lastgood_yaw = None  # last IMU heading while belt was visible (held during belt loss)
    robot = uenv.scene["robot"]
    _rail_lidar = None
    if _guide_nav:
        try:
            _rail_lidar = uenv.scene["rail_lidar"]
        except Exception as _e:
            print(f"[WALK] rail_lidar unavailable ({_e}); GUIDE_NAV will coast", flush=True)
    names = list(robot.joint_names)
    leg_ids = torch.tensor([names.index(n) for n in LEG_ORDER], device=uenv.device)

    actor = rma.build_walk_actor(args_cli.walk_ckpt, device=str(uenv.device))
    adapt = rma.AdaptationNet(os.environ["EUNHA_ADAPT_CKPT"], device=str(uenv.device))
    hist = torch.zeros(1, rma.HIST_LEN, rma.FEAT_DIM, device=uenv.device)
    last_a = torch.zeros(1, 12, device=uenv.device)

    # env joint_leg action: target = default + 0.5 * a_env  (order = LEG_ORDER)
    # policy action:        target = default + (0.125 hip / 0.25 other) * a_pol
    ratio = torch.tensor([0.125 / 0.5 if "_hip_" in n else 0.25 / 0.5 for n in LEG_ORDER],
                         device=uenv.device)
    n_act = uenv.action_manager.total_action_dim
    act = torch.zeros(1, n_act, device=uenv.device)

    # optional: teleport the spawn further into the course (room behind for a backward test)
    if os.environ.get("START_X"):
        _sx = float(os.environ["START_X"])
        _pose = robot.data.default_root_state[:, :7].clone()
        _pose[:, 0] = _sx
        _pose[:, 1] = float(ct.curve_y(_sx))
        robot.write_root_pose_to_sim(_pose)
        robot.write_root_velocity_to_sim(torch.zeros((_pose.shape[0], 6), device=uenv.device))
        print(f"[WALK] teleported start to x={_sx:.2f}", flush=True)

    # GUIDE_NAV: align spawn heading to the course tangent so we don't start a few degrees off
    # (robot spawns ON the curve but facing +x), which would trigger a big early recovery-steer.
    if _guide_nav and os.environ.get("SPAWN_ALIGN", "1") == "1":
        _rs = robot.data.root_state_w.clone()
        _sxa = float(_rs[0, 0]); _tha = float(ct.curve_theta(_sxa))
        _rs[0, 3] = np.cos(_tha / 2.0); _rs[0, 4] = 0.0; _rs[0, 5] = 0.0; _rs[0, 6] = np.sin(_tha / 2.0)
        robot.write_root_state_to_sim(_rs)
        uenv.sim.step(render=False)
        print(f"[WALK] spawn heading aligned to {_tha:+.3f} rad at x={_sxa:+.2f}", flush=True)

    # rail-height + forward-window filters for LiDAR guide-line detection
    _GZ_LO = float(os.environ.get("GUIDE_Z_LO", "0.20"))
    _GZ_HI = float(os.environ.get("GUIDE_Z_HI", "0.55"))
    _GFWD = float(os.environ.get("GUIDE_FWD", "3.0"))
    _GMAXRES = float(os.environ.get("GUIDE_MAX_RESID", "0.20"))  # reject non-linear (blob) fits

    def _detect_rail():
        """(heading_off, lat, nL, nR) in the SENSOR/body frame from the LiDAR point cloud.
        heading_off: yaw to ADD to become parallel to the near rail (0 = parallel).
        lat: signed lateral distance to the near rail [m] (>0 rail on left). None-ish if no rail.

        DEPLOYMENT-EQUIVALENT: uses ONLY the lidar's native output -- world hits are put into the
        sensor's OWN frame via the sensor pose the device inherently has (no robot world position,
        no ground-truth yaw, no course map), and the ground height is estimated from the cloud
        itself (no ground-truth base z). This is exactly what a real on-robot LiDAR provides."""
        if _rail_lidar is None:
            return None
        d = _rail_lidar.data
        hits = d.ray_hits_w[0]                                  # (N,3) world hit points
        spos = d.pos_w[0]; squat = d.quat_w[0]                  # the lidar's own pose = native frame
        rel = hits - spos.unsqueeze(0)                          # per-ray return vectors
        pts = math_utils.quat_apply_inverse(squat.unsqueeze(0).expand(rel.shape[0], -1), rel)
        rx, ry, rz = pts[:, 0], pts[:, 1], pts[:, 2]            # sensor frame: x fwd, y left, z up
        fin = torch.isfinite(rx) & torch.isfinite(ry) & torch.isfinite(rz)
        if int(fin.sum()) < 20:
            return (None, None, 0, 0)
        ground = torch.quantile(rz[fin], 0.10)                  # local ground height from the cloud
        dz = rz - ground                                       # height above ground (no gt base z)
        m = fin & (dz > _GZ_LO) & (dz < _GZ_HI) & (rx.abs() < _GFWD)
        if int(m.sum()) < 4:
            return (None, None, 0, 0)
        rxm, rym = rx[m], ry[m]
        left = rym > 0
        nL, nR = int(left.sum()), int((~left).sum())
        def _fit(side):
            if int(side.sum()) < 3:
                return None
            X = rxm[side]; Y = rym[side]
            Xm = X - X.mean(); Ym = Y - Y.mean()
            den = float((Xm * Xm).sum())
            if den < 1e-6:
                return None
            aa = float((Xm * Ym).sum() / den); bb = float(Y.mean() - aa * X.mean())
            # linearity check: a real belt is a thin LINE (small residual). Stair faces / raised
            # terrain caught by the height band form a broad blob (large residual) -> reject.
            if int(side.sum()) >= 5 and float((Y - (aa * X + bb)).std()) > _GMAXRES:
                return None
            return (aa, bb)
        fL, fR = _fit(left), _fit(~left)
        cand = [f for f in (fL, fR) if f is not None]
        if not cand:
            return (None, None, nL, nR)
        aa, bb = min(cand, key=lambda f: abs(f[1]))            # near rail = smaller |lateral|
        return (float(np.arctan(aa)), bb, nL, nR)

    _meas = bool(os.environ.get("MEASURE_POWER"))
    _tau_hist, _vel_hist = [], []
    _no_map = bool(os.environ.get("NO_MAP"))  # drop sim position/curve -> IMU heading-hold only
    _yaw0 = None

    first = True
    dir_sign = 1.0            # ping-pong along the course: never stop walking
    stall_x, stall_i = None, 0
    with torch.inference_mode():
        for i in range(args_cli.steps):
            data = robot.data
            q = data.joint_pos[:, leg_ids] - data.default_joint_pos[:, leg_ids]
            qd = data.joint_vel[:, leg_ids] * 0.05
            yaw = math_utils.euler_xyz_from_quat(data.root_quat_w)[2]
            px = data.root_pos_w[0, 0].item()
            # completion-based recording: stop once the robot first reaches the finish/bin zone,
            # so slow speeds are not cut off mid-course (set FINISH_X, e.g. 16.5).
            if os.environ.get("FINISH_X") and px >= float(os.environ["FINISH_X"]):
                print(f"[WALK] finished at x={px:.2f} (step {i})", flush=True)
                break
            py = data.root_pos_w[0, 1].item()
            th = float(ct.curve_theta(px))
            lat = float((py - ct.curve_y(px)) * np.cos(th) - 0.0)  # lateral offset from centerline
            # ping-pong: turn around before the bin zone / behind the start line
            if dir_sign > 0 and px > 17.0:
                dir_sign = -1.0
            elif dir_sign < 0 and px < -19.0:
                dir_sign = 1.0
            # anti-stall: if <0.3 m progress in 6 s, boost the speed command briefly
            if stall_x is None or i - stall_i >= 300:
                boost = 1.0 if (stall_x is not None and abs(px - stall_x) < 0.3) else 0.0
                stall_x, stall_i = px, i
            vx_cmd = min(1.0, args_cli.vx + (0.3 if 'boost' in dir() and boost else 0.0))
            # steer: follow the S tangent (reversed when walking back), heading biased
            # back toward the centerline — near the red strips the pull dominates
            base_th = float(th) if dir_sign > 0 else float(th) + np.pi
            head_target = base_th - max(-0.7, min(0.7, 0.8 * lat * dir_sign))
            err = head_target - yaw.item()
            err = (err + np.pi) % (2 * np.pi) - np.pi
            cmd = torch.zeros(1, 3, device=uenv.device)
            cmd[:, 0] = vx_cmd
            cmd[:, 1] = max(-0.3, min(0.3, -0.5 * lat * dir_sign))
            cmd[:, 2] = max(-0.7, min(0.7, 1.2 * err))
            if _no_map:
                # NO map/position: ignore course curve + root_pos_w. Hold the INITIAL heading
                # using only IMU-style orientation (yaw), like a real robot with no camera.
                if _yaw0 is None:
                    _yaw0 = yaw.item()
                e = ((_yaw0 - yaw.item()) + np.pi) % (2 * np.pi) - np.pi
                cmd[:, 0] = args_cli.vx
                cmd[:, 1] = 0.0
                cmd[:, 2] = max(-0.7, min(0.7, 1.2 * e))
            if _cam_nav and _hc is not None:
                # CAMERA nav: heading from the head-camera belt only (no sim position/curve)
                _rgb = _hc.data.output["rgb"][0, ..., :3].cpu().numpy()
                _vxc, _wzc, _dbg = _bnav.compute_command(_rgb, args_cli.vx, _prev_wz)
                # on stairs the body pitches up -> gravity's body-x grows. Lock steering to a
                # straight IMU heading hold (ignore belt turns that destabilize the climb).
                _gx = float(data.projected_gravity_b[0, 0])
                _on_stairs = abs(_gx) > 0.25
                _belt_seen = ("no_belt" not in _dbg) and ("no_img" not in _dbg)
                if _on_stairs:
                    # pitched up on stairs -> belt points at sky / is unreliable. Hold a straight
                    # IMU heading so the climb is not destabilized by bogus belt turns.
                    if _lastgood_yaw is None:
                        _lastgood_yaw = yaw.item()
                    _e = ((_lastgood_yaw - yaw.item()) + np.pi) % (2 * np.pi) - np.pi
                    _wzc = float(max(-0.7, min(0.7, 1.2 * _e)))
                    _vxc = args_cli.vx
                    _dbg += " [STAIR-LOCK]"
                elif _belt_seen:
                    # level + belt visible -> follow belt (belt_nav wz already set) and refresh heading
                    _lastgood_yaw = yaw.item()
                else:
                    # level but belt momentarily lost (segmented webbing / gap between posts):
                    # DON'T snap to straight -- keep belt_nav's coasted wz so we hold the curve.
                    _dbg += " [coast]"
                _prev_wz = _wzc
                cmd[:, 0] = _vxc
                cmd[:, 1] = 0.0
                cmd[:, 2] = _wzc
                if i % 100 == 0:
                    print(f"[CAMNAV] step {i}: {_dbg} yaw={yaw.item():+.2f} gx={_gx:+.2f}", flush=True)
            if _guide_nav:
                # LiDAR guide: steer PARALLEL to the near red rail + hold a set distance to it.
                _gx = float(data.projected_gravity_b[0, 0])
                _on_stairs = abs(_gx) > 0.25
                if _on_stairs:
                    # climbing/steep (pitched up): the belt-height filter also catches stair faces
                    # and raised terrain ahead -> bogus rail. Hold a straight IMU heading (stairs
                    # are straight) until level again. IMU-only, deployable.
                    if _lastgood_yaw is None:
                        _lastgood_yaw = yaw.item()
                    _e = ((_lastgood_yaw - yaw.item()) + np.pi) % (2 * np.pi) - np.pi
                    _wzc = float(max(-0.7, min(0.7, 1.2 * _e)))
                    _dbg = f"[STAIR-LOCK] hold={_lastgood_yaw:+.2f}"
                else:
                    _rail = _detect_rail()
                    if _rail is not None and _rail[0] is not None:
                        _hoff, _lat, _nL, _nR = _rail
                        _keep = float(os.environ.get("GUIDE_KEEP_D", "0.9"))  # target dist to rail [m]
                        _kd = float(os.environ.get("GUIDE_KD", "0.4"))         # lateral-correction gain
                        _sgn = 1.0 if _lat > 0 else -1.0                       # rail on left(+)/right(-)
                        _laterr = abs(_lat) - _keep                           # >0: too far from rail
                        _rel = _hoff + _sgn * _kd * max(-0.5, min(0.5, _laterr))
                        _rel = max(-0.40, min(0.40, _rel))                    # clamp heading correction
                        _wzc = float(max(-0.7, min(0.7, 1.2 * _rel)))
                        _lastgood_yaw = yaw.item()
                        _dbg = f"rail hoff={_hoff:+.2f} lat={_lat:+.2f} nL={_nL} nR={_nR} rel={_rel:+.2f}"
                    else:
                        _wzc = _prev_wz * 0.9                                  # no rail -> coast heading
                        _dbg = "no_rail [coast]"
                _prev_wz = _wzc
                cmd[:, 0] = args_cli.vx
                cmd[:, 1] = 0.0
                cmd[:, 2] = _wzc
                if i % 100 == 0:
                    print(f"[GUIDE] step {i}: {_dbg} yaw={yaw.item():+.2f} gx={_gx:+.2f}", flush=True)
            obs45 = torch.cat([data.root_ang_vel_b * 0.25, data.projected_gravity_b,
                               cmd, q, qd, last_a], dim=1)
            if first:
                hist[:] = obs45.unsqueeze(1).expand(-1, rma.HIST_LEN, -1)
                first = False
            hist = torch.roll(hist, shifts=-1, dims=1)
            hist[:, -1] = obs45
            latent = adapt(hist)
            a = actor(torch.cat([obs45, latent], dim=1))
            last_a = a
            act[:, :12] = a * ratio  # joint_leg block (leg order matches LEG_ORDER)
            if os.environ.get("RAISE_ARM"):
                # hold the Piper arm in a RAISED pose. arm block = act[:, 12:20] (arm_joint1..8),
                # target = default + 0.5*action  ->  action = 2*(target - default); arm_joint1-6 default 0.
                j2 = float(os.environ.get("ARM_J2", "1.5"))   # shoulder
                j3 = float(os.environ.get("ARM_J3", "-1.5"))  # elbow
                act[:, 12:20] = 0.0
                act[:, 13] = 2.0 * j2
                act[:, 14] = 2.0 * j3
            env.step(act)
            if _meas:
                _tau_hist.append(robot.data.applied_torque[0].clone())
                _vel_hist.append(robot.data.joint_vel[0].clone())
            # follow-cam: smooth position follow behind the robot, horizon-stable
            rp = data.root_pos_w[0]
            fc.set_world_poses_from_view(
                (rp + torch.tensor([-2.4, -2.4, 1.6], device=uenv.device)).unsqueeze(0),
                (rp + torch.tensor([0.0, 0.0, 0.3], device=uenv.device)).unsqueeze(0),
            )
            w_ov.append_data(ov.data.output["rgb"][0].cpu().numpy())
            w_fc.append_data(fc.data.output["rgb"][0].cpu().numpy())
            if w_hc is not None:
                w_hc.append_data(_hc.data.output["rgb"][0, ..., :3].cpu().numpy())
            if i % 200 == 0:
                x = data.root_pos_w[0, 0].item()
                z = data.root_pos_w[0, 2].item()
                print(f"[WALK] step {i}: x={x:.2f} z={z:.2f}", flush=True)
    w_ov.close()
    w_fc.close()
    if w_hc is not None:
        w_hc.close()
    print(f"[WALK] done: final x={robot.data.root_pos_w[0, 0].item():.2f}", flush=True)
    print("[WALK] videos: walk_overview.mp4 + walk_follow.mp4 (single run)", flush=True)
    if _meas and _tau_hist:
        T = torch.stack(_tau_hist); V = torch.stack(_vel_hist); P = (T * V).abs()
        names = list(robot.joint_names)
        print("[POWER] joint            max|tau|Nm  max|vel|rad/s  max|P|W   p95|P|W", flush=True)
        for j, n in enumerate(names):
            mt = T[:, j].abs().max().item(); mv = V[:, j].abs().max().item()
            mp = P[:, j].max().item(); pp = torch.quantile(P[:, j], 0.95).item()
            print(f"[POWER] {n:18s} {mt:8.1f} {mv:12.2f} {mp:10.1f} {pp:9.1f}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
