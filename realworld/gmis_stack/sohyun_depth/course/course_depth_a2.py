"""Drive OUR A2 parkour-teacher adapter along the competition course (course_sim) and record.

Reuses yewon's walk_course.py env-bring-up skeleton, but instead of the Eunha RMA velocity
policy it runs OUR solution_taska_parkour_a2.AlgSolution (the parkour teacher adapter with the
A' line-return steering). Purpose: reproduce OUR policy in the SAME course_sim arena the eunha
policy walked, to compare the crab-walk directly.

Per step we assemble the Task-A-style proprio (72) from robot.data in the exact layout our
adapter reads: [lin_vel(3) ang_vel(3) cmd(3) grav(3) jpos(20) jvel(20) act(20)], inject the
live yaw/roll/pitch/pos (+ a flat scan, since course_sim has no teacher scan grid), and call
solution.predicts. Two cameras (overview + follow) -> two videos from one run.

Run (GPU2):
  CUDA_VISIBLE_DEVICES=2 ... python course_parkour_a2.py --headless --steps 2000 --out_dir ...
"""
import argparse
import math
import os
import sys
from pathlib import Path

RW = Path("/hdd0/yewon/ATEC2026_Simulation_Challenge/real_world_preliminary/ATEC2026_Simulation_Challenge")
for _p in (str(RW / "source" / "atec_rl_lab"), str(RW), str(RW / "eunha_rl")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
# our adapter lives in our repo's demo/
_OUR_DEMO = "/hdd0/sohyun/atec_2026_tesk_e/ATEC2026_Simulation_Challenge/demo"
if _OUR_DEMO not in sys.path:
    sys.path.insert(0, _OUR_DEMO)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="ATEC-Course-A2Piper")
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--out_dir", type=str, default="/hdd0/sohyun/atec_real_world/course_parkour")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import numpy as np  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import course_sim  # noqa: F401, E402  (registers ATEC-Course-A2Piper)
from course_sim import terrain as ct  # noqa: E402  (SECTION_MESH_GROUPS: world-coord trimeshes)
from epkg import rma  # noqa: E402  (LEG_ORDER)

from solution_taska_depth_student import AlgSolution  # noqa: E402  (DEPTH STUDENT adapter)

LEG_ORDER = rma.LEG_ORDER   # 12 legs, per-leg order (FR,FL,RR,RL) -- same as Task A
SCAN_Z_OFFSET = 0.3


def _build_course_raymesh():
    """Concatenate the course section trimeshes into ONE trimesh ray intersector (world coords).
    The teacher's 132-pt scan is raycast straight down against this to read real terrain height,
    so stairs/slope are seen instead of a flat hard-code (which caused the earlier fall)."""
    import trimesh, numpy as np
    groups = getattr(ct, "SECTION_MESH_GROUPS", {}) or {}
    if not groups:
        return None
    merged = trimesh.util.concatenate([m for m in groups.values()])
    return merged


def _course_live_scan(robot, raymesh, scan_local, device):
    """Raycast the 132 teacher grid points against the course terrain; return the clipped scan.
    scan_local: (132,2) grid in base frame (x fwd, y left). Uses the robot's yaw-only pose."""
    import numpy as np
    import isaaclab.utils.math as _m
    base_pos = robot.data.root_pos_w[0]
    base_quat = robot.data.root_quat_w[0]
    yaw_q = _m.yaw_quat(base_quat.unsqueeze(0))
    n = scan_local.shape[0]
    local3 = torch.zeros((n, 3), device=device, dtype=torch.float32)
    local3[:, 0] = scan_local[:, 0]
    local3[:, 1] = scan_local[:, 1]
    world_xy = _m.quat_apply(yaw_q.repeat(n, 1), local3) + base_pos.unsqueeze(0)
    origins = world_xy.detach().cpu().numpy()
    origins[:, 2] = float(base_pos[2]) + 5.0                 # cast from high above, straight down
    dirs = np.zeros_like(origins); dirs[:, 2] = -1.0
    # trimesh: first hit per ray; fill misses with flat ground (height 0)
    locs, ridx, _ = raymesh.ray.intersects_location(ray_origins=origins, ray_directions=dirs,
                                                    multiple_hits=False)
    heights = np.zeros(n, dtype=np.float32)
    if len(ridx):
        heights[ridx] = locs[:, 2]
    ht = torch.from_numpy(heights).to(device)
    scan = torch.clip(base_pos[2] - ht - SCAN_Z_OFFSET, -1.0, 1.0)
    return scan


def main():
    device = args_cli.device if getattr(args_cli, "device", None) else "cuda:0"
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=1)
    for attr in list(vars(env_cfg.terminations)):
        if attr != "time_out":
            setattr(env_cfg.terminations, attr, None)
    env_cfg.episode_length_s = 10000.0

    from isaaclab.sensors import CameraCfg
    cam_common = dict(
        update_period=0.0, height=720, width=1280, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.1, 200.0)),
    )
    env_cfg.scene.overview_cam = CameraCfg(prim_path="/World/overview_cam", **cam_common)
    env_cfg.scene.follow_cam = CameraCfg(prim_path="/World/follow_cam", **cam_common)

    # The student reads the env's 5760-ray LiDAR. The course terrain sections live UNDER
    # /World/ground (COURSE_TERRAIN_CFG.prim_path="/World/ground", sections at
    # /World/ground/terrain_*), and the env's default lidar already targets /World/ground -- but
    # that's a container prim, so retarget the RayCaster at the section meshes so rays actually
    # hit the terrain surface.
    try:
        from isaaclab.sensors import MultiMeshRayCasterCfg
        _base = env_cfg.scene.lidar_sensor
        if _base is not None:
            env_cfg.scene.lidar_sensor = MultiMeshRayCasterCfg(
                prim_path=_base.prim_path, update_period=_base.update_period,
                pattern_cfg=_base.pattern_cfg, max_distance=_base.max_distance,
                debug_vis=False,
                mesh_prim_paths=(
                    [MultiMeshRayCasterCfg.RaycastTargetCfg(
                        prim_expr="/World/ground/terrain_.*", track_mesh_transforms=False)]
                    + ([MultiMeshRayCasterCfg.RaycastTargetCfg(
                        prim_expr="/World/envs/env_0/Trash.*", track_mesh_transforms=True)]
                       if int(os.environ.get("LIDAR_INCLUDE_OBJECTS", "0")) else [])
                ),
            )
            # SEPARATE rail raycaster: targets ONLY the red boundary prims (strip_* = rigid 0.40 m
            # standing marker, belt_* = webbing at BELT_Z), so the student's terrain height_scan is
            # untouched while _detect_rail() gets clean guide-line returns. Same pattern/pose.
            if int(os.environ.get("LIDAR_INCLUDE_RAILS", "1")):
                env_cfg.scene.rail_lidar = MultiMeshRayCasterCfg(
                    prim_path=_base.prim_path, update_period=_base.update_period,
                    pattern_cfg=_base.pattern_cfg, max_distance=_base.max_distance,
                    debug_vis=False,
                    mesh_prim_paths=[
                        MultiMeshRayCasterCfg.RaycastTargetCfg(
                            prim_expr="/World/ground/strip_.*", track_mesh_transforms=False),
                        MultiMeshRayCasterCfg.RaycastTargetCfg(
                            prim_expr="/World/ground/belt_.*", track_mesh_transforms=False),
                    ],
                )
            _inc = "terrain+OBJECTS" if int(os.environ.get("LIDAR_INCLUDE_OBJECTS", "0")) else "terrain only"
            print(f"[COURSE] retargeted LiDAR ({_inc})", flush=True)
    except Exception as _e:
        print(f"[COURSE] lidar retarget failed ({_e}); using default lidar.", flush=True)

    # DEPTH camera the student distills terrain from: a raycast-simulated depth camera matching the
    # TRAINING CAMERA_CFG (default_cfg.py) -- RayCasterCamera at the A2 head-camera pose on base_link,
    # 60x106 distance_to_camera, max_distance 2.0. MultiMesh so it hits the course terrain sections.
    try:
        import torch as _torch
        from isaaclab.sensors import MultiMeshRayCasterCameraCfg
        from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
        from isaaclab.utils.math import quat_from_euler_xyz as _qfe
        _rot = _qfe(*[_torch.deg2rad(_torch.tensor(float(a))) for a in (180.0, 70.0, -90.0)])
        _rot = tuple(float(x) for x in _rot)
        env_cfg.scene.depth_camera = MultiMeshRayCasterCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            update_period=0.0,
            data_types=["distance_to_camera"],
            depth_clipping_behavior="max",
            max_distance=2.0,
            offset=MultiMeshRayCasterCameraCfg.OffsetCfg(
                pos=(0.4216, float(os.environ.get("CAM_Y", "0.025")), 0.0619),
                rot=_rot, convention="ros"),
            pattern_cfg=PinholeCameraPatternCfg(
                focal_length=11.041, horizontal_aperture=20.955,
                vertical_aperture=12.240, height=60, width=106),
            debug_vis=False,
            mesh_prim_paths=[
                MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
                    prim_expr="/World/ground/terrain_.*", track_mesh_transforms=False),
            ],
        )
        print("[COURSE] depth camera added (raycast, 60x106, A2 head pose)", flush=True)
    except Exception as _e:
        print(f"[COURSE] depth camera add FAILED ({_e})", flush=True)

    os.makedirs(args_cli.out_dir, exist_ok=True)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    obs, extras = env.reset()
    uenv = env.unwrapped

    import imageio.v2 as iio
    ov = uenv.scene["overview_cam"]
    fc = uenv.scene["follow_cam"]
    eye = torch.tensor([[0.0, -36.0, 24.0]], device=uenv.device)
    tgt = torch.tensor([[0.0, 2.0, 0.0]], device=uenv.device)
    ov.set_world_poses_from_view(eye, tgt)
    w_ov = iio.get_writer(os.path.join(args_cli.out_dir, "parkour_overview.mp4"), fps=50)
    w_fc = iio.get_writer(os.path.join(args_cli.out_dir, "parkour_follow.mp4"), fps=50)

    robot = uenv.scene["robot"]

    # --- align SPAWN heading to the S-curve tangent -------------------------------------------
    # The env spawns the robot ON the curve (curve_map(x,0)) but with rot=identity (yaw=0, +x).
    # At the spawn x the curve tangent is curve_theta(x) (~ -3 deg), so the robot starts a few
    # degrees off the path. That small offset triggers a large recovery-steer early, which this
    # under-distilled policy answers by folding the front legs and pitching forward. Rotate the
    # spawn quat to face the curve tangent so delta_yaw ~ 0 at t=0 and no recovery-lunge fires.
    if int(os.environ.get("SPAWN_ALIGN", "1")):
        import numpy as _np0
        _rs = robot.data.root_state_w.clone()
        _sx = float(_rs[0, 0]); _sy = float(_rs[0, 1])
        # recover course arc-x from world x (curve_map's X = arc_x - w*sin(th); w=0 at spawn -> X=arc_x)
        _spawn_th = float(ct.curve_theta(_sx))
        _hz = math.sin(_spawn_th / 2.0); _hw = math.cos(_spawn_th / 2.0)   # yaw-only quat (w,x,y,z)
        _rs[0, 3] = _hw; _rs[0, 4] = 0.0; _rs[0, 5] = 0.0; _rs[0, 6] = _hz
        robot.write_root_state_to_sim(_rs)
        uenv.sim.step(render=False)
        print(f"[COURSE] spawn heading aligned to curve tangent {_spawn_th:+.3f} rad "
              f"({math.degrees(_spawn_th):+.1f} deg) at x={_sx:+.2f}", flush=True)
    else:
        print("[COURSE] SPAWN_ALIGN=0: default identity spawn heading (+x)", flush=True)

    # --- force a STATIONARY initial state (zero root lin/ang velocity) -------------------------
    # root_state layout: [px,py,pz, qw,qx,qy,qz, vx,vy,vz, wx,wy,wz]. Zero the 6 velocity terms
    # so the robot starts fully at rest (no residual spawn velocity feeding the first obs).
    _rs0 = robot.data.root_state_w.clone()
    _rs0[0, 7:13] = 0.0
    robot.write_root_state_to_sim(_rs0)
    uenv.sim.step(render=False)
    print("[COURSE] initial state forced STATIONARY (root lin/ang vel = 0)", flush=True)

    names = list(robot.joint_names)
    # jpos/jvel: our adapter reads the FULL 20-joint proprio in per-leg order (legs then arm).
    # course robot.joint_names is articulation order; build a per-leg index list = LEG_ORDER + arm.
    arm_names = [n for n in names if n not in LEG_ORDER]     # joint1..joint8
    perleg_names = list(LEG_ORDER) + arm_names               # 20, Task A per-leg order
    perleg_ids = torch.tensor([names.index(n) for n in perleg_names], device=uenv.device)
    n_act = uenv.action_manager.total_action_dim

    solution = AlgSolution()
    if hasattr(solution, "reset"):
        solution.reset()

    # TEACHER_SCANDOTS mode: build the teacher's 132-pt scandots grid template (base-frame x,y) using
    # IsaacLab's own GridPattern so the ray ORDER matches the teacher's height_scanner exactly, then
    # each step raycast it against the course terrain -> true scandots for the raw teacher policy.
    _teacher_scandots = int(os.environ.get("TEACHER_SCANDOTS", "0"))
    _no_scan = int(os.environ.get("TEACHER_NO_SCAN", "0"))   # load teacher but feed NO scandots
    _scan_local = None
    _raymesh = None
    if _teacher_scandots and not _no_scan:
        from isaaclab.sensors.ray_caster import patterns
        _gp = patterns.GridPatternCfg(resolution=0.15, size=[1.65, 1.5])
        _starts, _ = patterns.grid_pattern(_gp, device=uenv.device)   # (132, 3) ray starts (x,y,z)
        # scanner offset pos=(0.375,0,20) forward of base; add the 0.375 m forward shift to x.
        _scan_local = _starts[:, :2].clone()
        _scan_local[:, 0] += 0.375
        _raymesh = _build_course_raymesh()
        print(f"[COURSE] TEACHER_SCANDOTS on: {_scan_local.shape[0]}-pt grid, "
              f"raymesh={'OK' if _raymesh is not None else 'MISSING'}", flush=True)

    # locate the lidar sensor to read its 5760-ray scan each step (student exteroception)
    lidar = uenv.scene.sensors.get("lidar_sensor") if hasattr(uenv.scene, "sensors") else None
    if lidar is None:
        try:
            lidar = uenv.scene["lidar_sensor"]
        except Exception:
            lidar = None
    print(f"[COURSE] lidar sensor: {'FOUND' if lidar is not None else 'MISSING'}", flush=True)
    rail_lidar = None
    if int(os.environ.get("LIDAR_INCLUDE_RAILS", "1")):
        try:
            rail_lidar = uenv.scene.sensors.get("rail_lidar") if hasattr(uenv.scene, "sensors") else None
            if rail_lidar is None:
                rail_lidar = uenv.scene["rail_lidar"]
        except Exception:
            rail_lidar = None
    print(f"[COURSE] rail lidar: {'FOUND' if rail_lidar is not None else 'MISSING'}", flush=True)

    # locate the depth camera (student exteroception). Reads raw distance_to_camera each step.
    depth_camera = None
    try:
        depth_camera = uenv.scene.sensors.get("depth_camera") if hasattr(uenv.scene, "sensors") else None
        if depth_camera is None:
            depth_camera = uenv.scene["depth_camera"]
    except Exception:
        depth_camera = None
    print(f"[COURSE] depth camera: {'FOUND' if depth_camera is not None else 'MISSING'}", flush=True)

    def _depth_raw():
        # raw distance_to_camera (H0,W0). The adapter._depth crops/resizes/normalizes to (1,58,87).
        if depth_camera is None:
            return None
        out = depth_camera.data.output["distance_to_camera"][0]   # (H0,W0,1) or (H0,W0)
        return out.squeeze(-1) if out.dim() == 3 else out

    def _lidar_scan():
        # mdp.height_scan returns sensor_z - hit_z - offset; the student adapter's _lidar expects
        # obs['extero'] = that raw scan (it does /max_dist - 0.5 itself). Reproduce height_scan.
        if lidar is None:
            return torch.zeros(1, 5760, device=uenv.device)
        hits = lidar.data.ray_hits_w[0]                    # (5760, 3) world hit points
        sz = lidar.data.pos_w[0, 2]                         # sensor world z
        h = sz - hits[:, 2] - 0.5                           # offset 0.5 (matches LIDAR_OFFSET)
        h = torch.nan_to_num(h, nan=10.0, posinf=10.0, neginf=10.0)
        return h.view(1, -1)

    # --- RULE-BASED GUIDE-LINE (red belt/post rail) detection from raw LiDAR ---
    # The course boundary is a red webbing belt at BELT_Z=0.40 m and rigid posts (H=0.44 m) down
    # BOTH edges. The LiDAR is a real 3D scan (16 ch, vfov +-20 deg, hfov 360), so it returns those
    # rails. height_scan (used by the student) collapses them away, but here we read the raw hits,
    # keep only points at belt height above the local ground, split into left/right of the robot,
    # and fit the near rail's heading -> a deployable "walk parallel to the guide line" steer.
    # GUIDE_STEER=1 turns it on; otherwise it only logs (detection-first validation).
    _BELT_Z_LO = float(os.environ.get("GUIDE_Z_LO", "0.20"))   # min height above local ground [m]
    _BELT_Z_HI = float(os.environ.get("GUIDE_Z_HI", "0.55"))   # max height above local ground [m]
    _GUIDE_FWD = float(os.environ.get("GUIDE_FWD", "3.0"))     # only use rail hits within +-this x (robot frame)

    def _detect_rail():
        """Return (heading_off, lat, n_left, n_right) in the ROBOT/SENSOR frame.
        heading_off: yaw offset [rad] to make the robot parallel to the near rail (0 = parallel).
        lat: signed lateral distance to the near rail [m] (>0 rail on left).

        REAL-WORLD ONLY inputs: this uses solely the LiDAR's own returns, expressed in the SENSOR
        frame via the sensor's own pose (pos_w/quat_w) -- exactly what a physical LiDAR hands you
        (points in its own frame). No ground-truth world yaw and no world position are used: the
        hit-minus-sensor vector is rotated by the INVERSE of the sensor quaternion, so the result is
        the same body-frame point cloud a real unit reports. Height is taken relative to the sensor,
        also a real onboard quantity."""
        _rl = rail_lidar if rail_lidar is not None else lidar
        if _rl is None:
            return None
        hits = _rl.data.ray_hits_w[0]                           # (N,3) world (rail prims only)
        s_pos = _rl.data.pos_w[0]                               # (3,) sensor world position
        s_quat = _rl.data.quat_w[0]                             # (4,) sensor world orientation (wxyz)
        # relative vector hit-minus-sensor (this is the LiDAR range vector) -> rotate into the
        # SENSOR body frame with the inverse sensor rotation. Both are onboard quantities.
        rel = hits - s_pos.unsqueeze(0)                        # (N,3) world-frame range vectors
        body = math_utils.quat_rotate_inverse(s_quat.unsqueeze(0).repeat(hits.shape[0], 1), rel)
        rx = body[:, 0]                                        # forward (sensor +x)
        ry = body[:, 1]                                        # left (sensor +y)
        dz = body[:, 2]                                        # height above the sensor [m] (onboard)
        # belt/post sit ~0.40 m above ground; the sensor rides ~0.30 m above ground, so a rail is
        # roughly [BELT_Z_LO-0.30 .. BELT_Z_HI-0.30] relative to the sensor. Shift the window down
        # by the mounting height so the same GUIDE_Z_* env values keep meaning "height above ground".
        _z_mount = float(os.environ.get("GUIDE_SENSOR_H", "0.30"))
        dz = dz + _z_mount                                     # approx height above local ground
        m = (dz > _BELT_Z_LO) & (dz < _BELT_Z_HI) & (rx.abs() < _GUIDE_FWD) & torch.isfinite(rx) & torch.isfinite(ry)
        if int(m.sum()) < 4:
            return (None, None, 0, 0)
        rxm, rym = rx[m], ry[m]
        left = rym > 0; right = ~left
        nL, nR = int(left.sum()), int(right.sum())
        # near rail = the side with more hits closer; pick the side whose |ry| median is smaller
        def _fit(side):
            if int(side.sum()) < 3:
                return None
            X = rxm[side]; Y = rym[side]
            # line Y = a*X + b ; slope a = tan(rail heading rel to robot). heading_off = -atan(a)
            Xm = X - X.mean(); Ym = Y - Y.mean()
            denom = float((Xm * Xm).sum())
            if denom < 1e-6:
                return None
            a = float((Xm * Ym).sum() / denom)
            b = float(Y.mean() - a * X.mean())                 # lateral intercept at robot (x=0)
            return (a, b, int(side.sum()))
        fL, fR = _fit(left), _fit(right)
        # choose the near rail (smaller |b|)
        cand = [f for f in (fL, fR) if f is not None]
        if not cand:
            return (None, None, nL, nR)
        a, b, _ = min(cand, key=lambda f: abs(f[1]))
        # a = d(ry)/d(rx) of the rail in ROBOT frame. To make the rail parallel to the robot's
        # forward (+x) axis, rotate the robot by +atan(a) (turn toward the rail's own heading).
        heading_off = math.atan(a)
        return (heading_off, b, nL, nR)

    # FIXED FINISH point (world xy): the course end (arc-x = COURSE_X0 + COURSE_LEN) mapped onto
    # the course. The adapter steers straight to this single point instead of tracking the curve.
    import numpy as _np0
    _finish_arcx = float(ct.COURSE_X0 + ct.COURSE_LEN)
    _fgx, _fgy = ct.curve_map(_finish_arcx, 0.0)
    _FINAL_GOAL_XY = (float(_fgx), float(_fgy))
    print(f"[COURSE] fixed final goal (finish) = ({_FINAL_GOAL_XY[0]:.2f}, {_FINAL_GOAL_XY[1]:.2f})",
          flush=True)

    with torch.inference_mode():
        for i in range(args_cli.steps):
            data = robot.data
            # --- assemble Task-A-style proprio (72) in the layout our adapter reads ---
            lin_vel = data.root_lin_vel_b[0]                 # (3,)
            ang_vel = data.root_ang_vel_b[0]                 # (3,)
            grav = data.projected_gravity_b[0]               # (3,)
            jpos = (data.joint_pos[0] - data.default_joint_pos[0])[perleg_ids]   # (20,) per-leg
            jvel = data.joint_vel[0][perleg_ids]             # (20,) per-leg
            cmd = torch.zeros(3, device=uenv.device)         # our adapter ignores cmd slot
            act_slot = torch.zeros(20, device=uenv.device)   # our adapter ignores act slot
            proprio = torch.cat([lin_vel, ang_vel, cmd, grav, jpos, jvel, act_slot]).view(1, -1)

            # --- inject live pose ---
            # pos + yaw are REQUIRED to compute the bearing to the goal (can't be derived from
            # proprio alone). roll/pitch are OPTIONAL: with INJECT_ATTITUDE=0 we DON'T inject them,
            # so the adapter reconstructs roll/pitch from the proprio gravity vector instead -- a
            # step toward using only on-robot sensing (real-world has an IMU, not ground-truth euler).
            rq = data.root_quat_w[0]
            r, p, y = math_utils.euler_xyz_from_quat(rq.unsqueeze(0))
            solution._live_yaw = float(math_utils.wrap_to_pi(y)[0])
            if int(os.environ.get("INJECT_ATTITUDE", "0")):
                solution._live_roll = float(math_utils.wrap_to_pi(r)[0])
                solution._live_pitch = float(math_utils.wrap_to_pi(p)[0])
            else:
                solution._live_roll = None       # -> adapter uses gravity-based roll/pitch
                solution._live_pitch = None
            pos = data.root_pos_w[0]
            px_w, py_w = float(pos[0]), float(pos[1])
            solution._live_pos = (px_w, py_w)
            # STRAIGHT-TO-GOAL steering: the course curvature is mild, so instead of tracking the
            # S-curve tangent (which lagged and left the robot chronically off-heading -> asymmetric
            # gait), just steer on the straight bearing from the current position to the FIXED
            # finish point. _final_goal is computed once (below the loop setup) as the course end.
            import numpy as _np
            _th = float(ct.curve_theta(px_w))
            _cy = float(ct.curve_y(px_w))
            solution._live_offset = _np.cos(_th) * (py_w - _cy)   # keep offset just for logging
            # STEER_MODE: goal = steer to fixed finish; spawn = just hold the spawn heading and
            # walk forward (no goal point at all); curve = old pure-pursuit (unused here).
            # RULE-BASED GUIDE (real-world only): detect the red rail from the onboard LiDAR and
            # steer parallel to it. The heading correction _rel is RELATIVE (rail-frame), computed
            # purely from LiDAR returns -- no world position, no ground-truth yaw -- and injected
            # DIRECTLY as delta_yaw. No goal-direction fallback: if the rail is momentarily unseen,
            # inject 0 (hold heading, walk straight) rather than reaching for a world goal point.
            _rail = _detect_rail()
            if _rail is not None and _rail[0] is not None:
                _hoff, _lat, _nL, _nR = _rail
                # parallel to rail (_hoff -> 0) + hold a set lateral offset: nudge heading toward/away.
                _keep = float(os.environ.get("GUIDE_KEEP_D", "0.9"))   # desired dist to near rail [m]
                _kd = float(os.environ.get("GUIDE_KD", "0.4"))          # lateral-correction gain
                _clamp = float(os.environ.get("GUIDE_CLAMP", "0.40"))   # max total heading correction
                _laterr_clamp = float(os.environ.get("GUIDE_LATERR_CLAMP", "0.5"))  # max lat-err used
                _sign = 1.0 if _lat > 0 else -1.0                       # rail on left(+) or right(-)
                _lat_err = (abs(_lat) - _keep)                          # >0 too far from rail
                _rel = _hoff + _sign * _kd * max(-_laterr_clamp, min(_laterr_clamp, _lat_err))
                _rel = max(-_clamp, min(_clamp, _rel))                  # clamp total heading correction
                solution._dbg_guide = (_hoff, _rel, _lat)
            else:
                _rel = 0.0                                             # rail unseen -> walk straight
                solution._dbg_guide = None
            solution._dbg_rail = _rail
            if int(os.environ.get("GUIDE_STEER", "0")):
                # DIRECT relative-steer injection: LiDAR-only, real-world-deployable direction.
                solution._live_delta_yaw = _rel
                solution._final_goal = None
                solution._live_goal_point = None

            # TEACHER_SCANDOTS: raycast the true 132-pt scandots grid against the course terrain and
            # hand them to the adapter, which writes them into obs[77:209] for the raw teacher policy.
            # TEACHER_NO_SCAN=1: run the teacher policy WITHOUT any scandots (obs[77:209] stays 0 ->
            # the teacher perceives flat/blind terrain); only the env_idx rough flag (ENV_MODE) tells
            # it "this is rough". Isolates the teacher gait with NO terrain exteroception at all.
            if _teacher_scandots and not _no_scan and _raymesh is not None and _scan_local is not None:
                solution._live_scandots = _course_live_scan(robot, _raymesh, _scan_local, uenv.device)

            # student uses the DEPTH camera: feed proprio + raw depth to the adapter (which
            # crops/resizes/normalizes to 58x87). extero(lidar) still passed for TEACHER_SCANDOTS/debug.
            depth_raw = _depth_raw()
            resp = solution.predicts({"proprio": proprio, "depth": depth_raw}, 0.0)
            a20 = torch.tensor(resp["action"], device=uenv.device, dtype=torch.float32).view(-1)
            # student adapter returns 20 actions in per-leg order (legs 0..11, arm 12..19 at its
            # hold target). course action = joint_leg(12) + joint_arm(8), both per-leg order.
            act = torch.zeros(1, n_act, device=uenv.device)
            act[0, :min(20, n_act)] = a20[:min(20, n_act)]
            env.step(act)

            rp = data.root_pos_w[0]
            fc.set_world_poses_from_view(
                (rp + torch.tensor([-2.4, -2.4, 1.6], device=uenv.device)).unsqueeze(0),
                (rp + torch.tensor([0.0, 0.0, 0.3], device=uenv.device)).unsqueeze(0),
            )
            w_ov.append_data(ov.data.output["rgb"][0].cpu().numpy())
            w_fc.append_data(fc.data.output["rgb"][0].cpu().numpy())
            # --- motor power (torque) utilisation vs the actuator effort limits ---
            # computed_torque is the PD torque the actuator WANTS (pre-clip); joint_effort_limits is
            # the per-joint effort_limit_sim (hip/thigh 120, calf 180). ratio>1 => the motor would
            # saturate. Track the running max ratio over the 12 LEG joints (arm is held, ignore it).
            try:
                _ct = data.computed_torque[0][perleg_ids][:12].abs()      # 12 leg joints
                _el = data.joint_effort_limits[0][perleg_ids][:12]
                _ratio = (_ct / _el.clamp(min=1e-6))
                _rmax = float(_ratio.max()); _rmean = float(_ratio.mean())
                _pw_run_max = max(getattr(solution, "_pw_run_max", 0.0), _rmax)
                solution._pw_run_max = _pw_run_max
                _argmax = int(_ratio.argmax())
            except Exception as _te:
                _rmax = _rmean = -1.0; _pw_run_max = -1.0; _argmax = -1
            if i % 200 == 0:
                _jn = perleg_names[_argmax] if _argmax >= 0 else "?"
                _rl = getattr(solution, "_dbg_rail", None)
                if _rl is not None and _rl[0] is not None:
                    _railstr = f"RAIL hoff={_rl[0]:+.2f}rad lat={_rl[1]:+.2f}m L={_rl[2]} R={_rl[3]}"
                elif _rl is not None:
                    _railstr = f"RAIL none (L={_rl[2]} R={_rl[3]})"
                else:
                    _railstr = "RAIL n/a"
                _gd = getattr(solution, "_dbg_guide", None)
                if _gd is not None:
                    _railstr += f" GUIDE hoff={_gd[0]:+.2f} rel={_gd[1]:+.2f} lat={_gd[2]:+.2f}"
                print(f"[PARKOUR] step {i}: x={float(rp[0]):.2f} y={float(rp[1]):.2f} "
                      f"z={float(rp[2]):.2f} yaw={solution._live_yaw:+.2f} "
                      f"offset={solution._live_offset:+.2f} | "
                      f"power now max={_rmax*100:.0f}% mean={_rmean*100:.0f}% ({_jn}) "
                      f"run_max={_pw_run_max*100:.0f}% | {_railstr}", flush=True)
            # FINISH: the robot walks off the end of the course terrain past the finish line and
            # falls (z drops below the ground). Stop recording there so the video ends at the
            # finish instead of showing the fall into the void.
            if float(rp[2]) < 0.1:
                print(f"[PARKOUR] FINISH reached (walked off course end) at step {i}, "
                      f"x={float(rp[0]):.2f} y={float(rp[1]):.2f}", flush=True)
                break
    w_ov.close()
    w_fc.close()
    print(f"[PARKOUR] done: final x={float(robot.data.root_pos_w[0,0]):.2f} "
          f"y={float(robot.data.root_pos_w[0,1]):.2f}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
