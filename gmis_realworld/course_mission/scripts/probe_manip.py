"""Manipulation-only probe: skip the search, teleport the robot in front of a chosen
object on a chosen section, then run crouch -> stability gate -> RL grasp -> basket
deposit. Isolates the manipulation half of the mission from the perception half so a
failure can be attributed before committing to a 40-minute full run.

Run: PROFILE unused; pick with --cls and --section.
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

# Shared machine: GPUs 1-7 are other people's training runs. Pin to GPU 0 BEFORE any
# CUDA-aware import so forgetting the CUDA_VISIBLE_DEVICES prefix cannot land this on
# someone else's device.
import os as _os
_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Grasp + deposit probe on the course.")
parser.add_argument("--speed", type=str, default="s075")
parser.add_argument("--depth", type=str, default="0.24")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--per_type", type=int, default=4)
parser.add_argument("--walk_ckpt", type=str, default="")
parser.add_argument("--grasp_ckpt", type=str, default="")
parser.add_argument("--record_head", action="store_true")
parser.add_argument("--trace_dir", type=str, default="",
                    help="collect per-step grasp values AND frames here, for the\n                          overlay video (scripts/make_grasp_overlay.py)")
parser.add_argument("--trials", type=str, default="can:grass,rind:grass,lunchbox:mud,redbag:grass")
parser.add_argument("--repeat", type=int, default=1,
                    help="run the trial list this many times; before every trial the "
                         "article is put back at its settled spawn pose (and taken out "
                         "of the basket bookkeeping), so one launch samples an object's "
                         "grasp rate instead of paying the 3-minute start-up per sample")
parser.add_argument("--out_dir", type=str, default=str(RW / "course_mission" / "videos"))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import imageio.v2 as iio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as mu  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import course_mission  # noqa: F401, E402
import course_sim.terrain as ct  # noqa: E402
from course_mission.launch import head_cam_cfg, HEAD_POS, HEAD_ROT, TASK, resolve_ckpts  # noqa: E402
from course_mission.env_cfg import scatter_layout  # noqa: E402
from course_mission.runner import MissionRunner, Profile, ground_z_arc  # noqa: E402
from course_mission.mission import arc_of  # noqa: E402


def _close_writers_on_exit(writers):
    """An mp4 only becomes playable when its writer closes and stamps the moov atom.
    Killing a probe mid-run left files that no player would open (the user hit this), so
    the writers are closed on SIGTERM/SIGINT and on any exception, not just on success."""
    import atexit
    import signal

    def shut(*_):
        for w in writers.values():
            try:
                w.close()
            except Exception:
                pass
        raise SystemExit(1)

    atexit.register(lambda: [w.close() for w in writers.values()
                             if getattr(w, "_closed", False) is False])
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, shut)
        except ValueError:
            pass


def main():
    device = args_cli.device or "cuda:0"
    out = Path(args_cli.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ck = resolve_ckpts(args_cli)

    _os.environ["MISSION_LAYOUT_SEED"] = str(args_cli.seed)  # must precede parse_env_cfg
    env_cfg = parse_env_cfg(TASK, device=device, num_envs=1)
    env_cfg.per_type = args_cli.per_type
    env_cfg.per_type = args_cli.per_type
    rec = dict(update_period=0.04, height=720, width=1280, data_types=["rgb"],
               spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.1, 250.0)))
    env_cfg.scene.overview_cam = CameraCfg(prim_path="/World/overview_cam", **rec)
    env_cfg.scene.follow_cam = CameraCfg(prim_path="/World/follow_cam", **rec)
    # A camera for WATCHING THE GRASP. The follow cam sits behind and above the robot, so
    # the basket on its back fills the frame and hides the one thing worth seeing. This one
    # is re-aimed at the OBJECT every step from a low side angle, which keeps the gripper,
    # the object and the gap between them all in view.
    env_cfg.scene.grasp_cam = CameraCfg(
        prim_path="/World/grasp_cam",
        update_period=0.0, height=720, width=1280, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.02, 60.0)))
    env_cfg.scene.head_cam = head_cam_cfg()

    env = gym.make(TASK, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    env.reset()
    cams = {"overview": uenv.scene["overview_cam"], "follow": uenv.scene["follow_cam"],
            "head": uenv.scene["head_cam"], "grasp": uenv.scene["grasp_cam"]}
    cams["overview"].set_world_poses_from_view(
        torch.tensor([[0.0, -34.0, 26.0]], device=uenv.device),
        torch.tensor([[0.0, 2.0, 0.0]], device=uenv.device))
    class _FrameSink:
        """Writer-shaped sink that keeps frames in memory instead of encoding them.

        _capture() feeds every registered writer, so borrowing that path is how the frames
        get paired with the per-step numbers without touching the capture code.
        """

        cam_key = "grasp"           # which camera runner._capture should feed it

        def __init__(self):
            self.frames = []

        def append_data(self, img):
            # Downscale on the way in. Full 720p frames made a single trial's npz 670 MB,
            # which took minutes to write and left a truncated, CRC-broken file whenever
            # anything interrupted it. The overlay renders at 640 wide anyway.
            a = np.asarray(img, dtype=np.uint8)
            self.frames.append(a[::2, ::2])

        def close(self):
            pass

    writers = {"follow": iio.get_writer(str(out / "probe_manip_follow.mp4"), fps=25)}
    sink = None
    if args_cli.trace_dir:
        sink = _FrameSink()
        writers["trace"] = sink
    _close_writers_on_exit(writers)

    # PROBE_RETRIES: match the venue profile (4) to measure the full attempt budget
    # incl. the IK fallback; default 1 keeps the fast single-shot A/B rounds comparable.
    prof = Profile(name="probe", max_terrain_rank=3, max_picks=99,
                   grasp_retries=int(_os.environ.get("PROBE_RETRIES", "1")))
    r = MissionRunner(env, uenv, env_cfg, prof, ck, cams, writers, out, device)
    r.cfg_depth = float(args_cli.depth)
    robot = uenv.scene["robot"]

    # GRASP-MODULE ISOLATION (PROBE_NAV_TRUE=1): the teleport probe never localises,
    # so NavEst's decision frame is garbage here and the honest IK servo missed its
    # hover by 350-475 mm — the probe could not exercise the honest grasp at all
    # (0908, ikdiag). Seeding the frame with truth swaps the NAV module for identity
    # while every downstream consumer (S-frame targets, decision-frame servos,
    # arc_lat gates) runs its real honest code path. This is a bench-only switch:
    # the mission never sets it, and the grasp being tested still sees only its
    # frozen camera-style target.
    if _os.environ.get("PROBE_NAV_TRUE", "0") == "1":
        # The naive world-identity pin broke every approach (gmod1: S==world makes
        # arc_lat return world XY, so the |lat|>1.2 edge gates fired instantly).
        # The S frame is COURSE (arc, lateral) — pin it to the TRUE course pose:
        # est (arc, lat) = arc_of(true), yaw_S = true heading minus the course
        # tangent, Rd per the estimator's own convention.
        from course_mission.navest import _rz as _nav_rz
        from course_sim import terrain as _ct
        _nav = r.nav

        def _true_course():
            p_t, R_t, yaw_t = _nav._true_root()
            a, w = arc_of(float(p_t[0]), float(p_t[1]))
            th = float(_ct.curve_theta(a))
            return p_t, R_t, yaw_t, a, w, th

        def _frames_true():
            p_t, R_t, yaw_t, a, w, th = _true_course()
            return (np.array([a, w, float(p_t[2])]), _nav_rz(-th), R_t, p_t)

        _nav._frames = _frames_true
        _nav.arc_lat = lambda: (lambda *v: (v[3], v[4]))(*_true_course())
        _nav.yaw_S = lambda: (lambda *v: float(v[2] - v[5]))(*_true_course())
        # ...and the HEALTH signals. Pinning only the frame left since_fix growing
        # forever (no real corridor fits fire in a teleported scene), so the runner's
        # starving/reacquire watchdogs ate every trial in nav-health theatre
        # (rmud1: "navigation starving; relocalizing" -> reacquire budget spent ->
        # approach=False, zero grasps reached). A pinned frame IS a fresh fix by
        # definition — pin since_fix at 0 with a swallow-writes property.
        _nav.__class__ = type("NavPinned", (type(_nav),), {
            "since_fix": property(lambda s: 0, lambda s, v: None),
        })
        print("[PROBE] NavEst pinned to TRUE course pose + since_fix=0 "
              "(grasp-module isolation)", flush=True)

    wanted = [t.split(":") for t in args_cli.trials.split(",")]
    picked = []
    for cls, sec in wanted:
        for name, c, s_, ax, w, _z in env_cfg.mission_layout:
            if c == cls and s_ == sec and name not in [p[0] for p in picked]:
                picked.append((name, cls, sec, ax, w))
                break
    print(f"[PROBE] trials: {picked}", flush=True)

    results = []
    with torch.inference_mode():
        for _ in range(150):                      # let the trash settle
            r._step(r.legs.step(torch.zeros(1, 3, device=device)))
        # settled spawn poses, so --repeat can put an article back exactly as it
        # was found (an upright can stays upright, a lying one stays lying)
        spawn_pose = {name: (uenv.scene[name].data.root_pos_w[0].clone(),
                             uenv.scene[name].data.root_quat_w[0].clone())
                      for name, *_ in picked}
        trial_list = [(rep, *t) for rep in range(max(1, args_cli.repeat)) for t in picked]
        for rep, name, cls, sec, ax, w in trial_list:
            obj = uenv.scene[name]
            if rep > 0:
                sp, sq = spawn_pose[name]
                obj.write_root_pose_to_sim(torch.cat([sp.unsqueeze(0), sq.unsqueeze(0)], dim=1))
                obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))
                r.carried = [(n_, c_) for n_, c_ in r.carried if n_ != name]
                r._hand_item = None
                for _ in range(60):
                    r._write_arm(r.travel_q, open_=False)
                    r._step(r.legs.step(torch.zeros(1, 3, device=device)))
                print(f"[PROBE] repeat {rep + 1}: {name} back at its spawn pose", flush=True)
            # PROBE_LAY=1: put the article on its face before the trial. Bags spawn
            # upright here, but 399 of 428 bag encounters in the mission record are
            # LYING — the probe was measuring the 7 % case. Roll 90 deg about the
            # bag's own x (the handle bar), which is the attitude the record shows.
            if _os.environ.get("PROBE_LAY") == "1":
                _q0 = obj.data.root_quat_w[0]
                _hx = torch.tensor([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0],
                                   device=device)
                _qn = mu.quat_mul(_q0.unsqueeze(0), _hx.unsqueeze(0))[0]
                _pn = obj.data.root_pos_w[0].clone()
                _pn[2] = _pn[2] - 0.10          # settle it down onto the deck
                obj.write_root_pose_to_sim(torch.cat([_pn.unsqueeze(0),
                                                      _qn.unsqueeze(0)], dim=1))
                obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))
                for _ in range(120):
                    r._write_arm(r.travel_q, open_=False)
                    r._step(r.legs.step(torch.zeros(1, 3, device=device)))
                print(f"[PROBE] laid {name} on its face before the trial", flush=True)
            p = obj.data.root_pos_w[0].cpu().numpy()
            _Rz = mu.matrix_from_quat(obj.data.root_quat_w[0].unsqueeze(0))[0][2]
            print(f"[PROBE] {name} ({cls}, {sec}) rests at z {p[2]:.3f} with local axes·z "
                  f"{[round(float(v), 2) for v in _Rz]} "
                  f"({'STANDING' if abs(float(_Rz[2])) > 0.7 else 'lying'})", flush=True)
            oax, ow = arc_of(float(p[0]), float(p[1]))
            th = float(ct.curve_theta(oax))
            # stand so the object sits at the pregrasp bank anchor in front of the robot
            back = float(r.grasp_anchor[0])
            ex, ey = ct.curve_map(oax - back, ow)
            root = torch.tensor([[float(ex), float(ey), ground_z_arc(oax) + 0.50]], device=device)
            quat = mu.quat_from_euler_xyz(torch.zeros(1, device=device),
                                          torch.zeros(1, device=device),
                                          torch.tensor([th], device=device))
            # Carry the basket's contents along with the teleport. Only the robot body is
            # moved by write_root_pose_to_sim, so anything resting in the basket lost its
            # support and dropped on the spot — on video that looks like the robot being
            # flung away while the payload vanishes. Re-place each carried object at the
            # same pose relative to the new base.
            old_p = robot.data.root_pos_w[0].clone()
            old_q = robot.data.root_quat_w[0].clone()
            payload = []
            for pname, _pcls in r.carried:
                po = uenv.scene[pname]
                rel = mu.quat_apply_inverse(old_q.unsqueeze(0),
                                            po.data.root_pos_w[0].unsqueeze(0) - old_p.unsqueeze(0))
                payload.append((po, rel))
            if _os.environ.get("PROBE_NO_APPROACH") == "1":
                # STANCE-DIRECT TELEPORT (0909): the mission walk approach inside the
                # bench kept dying to frame/steering theatre five harness rounds in a
                # row (starving loops, loop-arounds spiralling to lat +10.7). The
                # bench's job is the GRASP, so place the robot exactly at the stance
                # the approach would have built — object at the pregrasp bank anchor
                # in the base frame, heading along the course tangent — and go
                # straight to crouch + attempt.
                from course_sim import terrain as _ct2
                _th = float(_ct2.curve_theta(oax))
                _axo = float(_os.environ.get("PROBE_STANCE_X", "0.568"))
                _ayo = float(_os.environ.get("PROBE_STANCE_Y", "-0.011"))
                _c, _s = np.cos(_th), np.sin(_th)
                _rx = float(p[0]) - (_c * _axo - _s * _ayo)
                _ry = float(p[1]) - (_s * _axo + _c * _ayo)
                # stand height from the LOCAL terrain — reusing the old root z put the
                # feet floating/buried at the new spot and every crouch read unstable
                _gz2 = float(ground_z_arc(oax))
                root = torch.tensor([[_rx, _ry, _gz2 + 0.52]], device=device)
                quat = torch.tensor([[float(np.cos(_th / 2)), 0.0, 0.0,
                                      float(np.sin(_th / 2))]], device=device)
            robot.write_root_pose_to_sim(torch.cat([root, quat], dim=1))
            robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))
            for po, rel in payload:
                new_pos = root + mu.quat_apply(quat, rel)
                po.write_root_pose_to_sim(torch.cat(
                    [new_pos, torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)], dim=1))
                po.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))
            r.legs.fresh = True
            for _ in range(80):                   # regain balance after the teleport
                r._write_arm(r.travel_q, open_=False)
                r._step(r.legs.step(torch.zeros(1, 3, device=device)))

            det = {"cls": cls, "pos_w": p + np.random.normal(0, 0.012, 3),
                   "dist": 0.5, "section": sec, "arc": oax, "lat": ow}
            got = dep = stable = False
            if sink is not None:
                # Trace from the APPROACH, not from run_grasp. The bag never enters the
                # grasp loop at all (runner.py:991 skips it for being 0.50 m outside the
                # pregrasp bank), so a grasp-only trace records nothing for exactly the
                # case we are trying to see.
                r.grasp_cam_target = obj
                r.grasp_trace = []
                sink.frames.clear()
                _f0 = r.frame_every
                r.frame_every = 3
            # Fixed feed-forward, measured over 132 unbiased probe first-crouches:
            # median error (-0.091, -0.155) with sd 0.05/0.06 — the shift is repeatable
            # in the probe's controlled setting, so predict it instead of the old
            # crouch-measure-stand-recrouch dance (which made x WORSE 81% of the time;
            # see runner.py aim_bias0 for the paired-retry numbers).
            # y only: x-close is harmless (champion HELD at x 0.24) but lateral
            # offset kills; the tight-lateral acceptance in approach() now actually
            # realises this aim instead of absorbing it in the radial stop circle.
            _bias = np.array([0.0, -0.155])
            _noapp = _os.environ.get("PROBE_NO_APPROACH") == "1"
            for corr in range(2):
                if _noapp:
                    # stance-direct: the teleport above already built the stance
                    ok_app, tw = True, np.asarray(det["pos_w"], float).copy()
                else:
                    ok_app, tw = r.approach(det, _bias)
                stable = r.crouch_and_check(sec)
                if not stable:
                    break
                b = r.to_base_frame(tw)
                err = b[:2] - r.grasp_anchor[:2]
                print(f"[PROBE] post-crouch alignment error {np.round(err,3).tolist()}", flush=True)
                if _noapp:
                    break        # stance-direct: no walking retry dance to run
                # component-wise bank-window gate, same thresholds as the mission loop
                if not (err[0] > (0.32 if det["cls"] == "redbag" else 0.20)
                        or abs(err[1]) > 0.18) or corr == 1:
                    break
                # Bounded, for the same reason as the mission loop: an object that has
                # been knocked away (or fallen off the course) reports a metres-large
                # "alignment error", and folding that in makes every later attempt aim
                # wide. One run reached an aim bias of [4.85, 4.01] m.
                if np.linalg.norm(err) < 0.6:
                    _bias = np.clip(_bias + err, -0.45, 0.45)
                    print(f"[PROBE] retry with aim bias {np.round(_bias,3).tolist()}",
                          flush=True)
                else:
                    print(f"[PROBE] ignoring implausible alignment error "
                          f"{np.round(err,2).tolist()} m", flush=True)
                r.stand_up()
                print("[PROBE] backing out 0.3 m before the re-approach", flush=True)
                import torch as _torch
                for _ in range(220):
                    c, _, _ = r.walk_cmd(0.0, vx=-0.30)
                    r._write_arm(r.travel_q, open_=False)
                    r._step(r.legs.step(c))
            if stable:
                if sink is not None:
                    r.frame_every = 1        # every step through the grasp itself
                # Same attempt machine the venue mission runs: RL tries then the
                # scripted-IK fallback (attempt count = PROBE_RETRIES + 1).
                got = r.attempt_grasp(name, {**det, "pos_w": tw})
            if sink is not None:
                r.frame_every = _f0
                td = Path(args_cli.trace_dir)
                td.mkdir(parents=True, exist_ok=True)
                tr = np.asarray([[t["step"], t["reach"], t["width"],
                                  1.0 if t["cmd_close"] else 0.0,
                                  t["a_rot"], t["a_xyz"], t["lift"],
                                  t.get("tilt", np.nan), t.get("foot_d", np.nan),
                                  t.get("in_grasp", 0.0)]
                                 for t in (r.grasp_trace or [])], dtype=np.float32)
                np.savez(
                    td / f"trace_{name}_{cls}_{sec}.npz",
                    frames=np.asarray(sink.frames, dtype=np.uint8),
                    trace=tr,
                    meta=np.asarray([name, cls, sec], dtype=object))
                print(f"[PROBE] trace {name}: {len(tr)} grasp steps, "
                      f"{len(sink.frames)} frames", flush=True)
                r.grasp_trace = None
                r.grasp_cam_target = None
            if got:
                if cls in r.p.basket_classes:
                    dep = r.deposit_to_basket(name)
                else:
                    # END-TO-END for hand classes: mission-style gripper carry to the
                    # bin, so a HELD proves the SCORE, not just the lift.
                    # SLOW LIFT FIRST, exactly like the mission FSM: a shallow lid
                    # pinch peels out under acceleration — both s5_600 lunchboxes were
                    # HELD and then found at their spawn point (the old shortcut went
                    # straight to stand_up and the tuck flung them out).
                    from course_mission.catalogue import grasp_hint as _gh
                    from course_mission.env_cfg import OBJ_DIMS as _OD
                    _, _depth = _gh(cls, _OD[cls])
                    _slow = 420 if _depth < 0.012 else 200
                    _wp = [r.retrieve["q_mid"]]
                    if r.carry_q is not None and r.carry_q != r.retrieve["q_mid"]:
                        _wp.append(r.carry_q)
                    r.play_traj(_wp, grip_open=False, steps_each=_slow)
                    _obj_b = r.to_base_frame(
                        r.objects[name].data.root_pos_w[0].cpu().numpy())
                    _tcp_b = r.to_base_frame(r.tcp_w()[0].cpu().numpy())
                    _gap = float(np.linalg.norm(_obj_b - _tcp_b))
                    # A BAG HANGS FURTHER OUT than a rigid article. Its handle bar sits
                    # at bag-local z 0.205, so a bag held correctly by the bar reads a
                    # TCP-to-centre gap of ~0.20 m by construction — bagL lifted one
                    # 601 mm off the deck and this check called it lost at 0.204.
                    if _gap > (0.32 if cls in ("redbag", "purplebag") else 0.20):
                        print(f"[PROBE] lift lost the {cls} (TCP gap {_gap:.3f} m) — "
                              f"counting the grasp as failed", flush=True)
                        got = False
                        r.stand_up()
                        results.append((name, cls, sec, ok_app, stable, got, dep))
                        print(f"[PROBE] {name:9s} {cls:10s} {sec:7s} approach={ok_app} "
                              f"stable={stable} grasp={got} basket={dep}", flush=True)
                        continue
                    r._hand_item = name
                    r.travel_q = torch.tensor(r.carry_q, dtype=torch.float32,
                                              device=r.device) \
                        if r.carry_q is not None else r.travel_q
                    r.stand_up()
                    colour = "blue"
                    if r.goto_bin(colour):
                        res = r.unload_one(name, colour)
                        print(f"[PROBE] bag delivery -> {res}", flush=True)
                        dep = (res == "correct")
                    else:
                        print("[PROBE] bag delivery: bin unreachable", flush=True)
                    r._hand_item = None
            r.stand_up()
            results.append((name, cls, sec, ok_app, stable, got, dep))
            print(f"[PROBE] {name:9s} {cls:10s} {sec:7s} approach={ok_app} stable={stable} "
                  f"grasp={got} basket={dep}", flush=True)

    print("[PROBE] ===== summary =====", flush=True)
    for row in results:
        print(f"[PROBE] {row}", flush=True)
    n = len(results)
    print(f"[PROBE] stable {sum(r[4] for r in results)}/{n} | "
          f"grasp {sum(r[5] for r in results)}/{n} | basket {sum(r[6] for r in results)}/{n}",
          flush=True)
    for wtr in writers.values():
        wtr.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
