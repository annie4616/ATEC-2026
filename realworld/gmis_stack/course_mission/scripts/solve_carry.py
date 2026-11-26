"""Solve a LOW CARRY pose: hold a grasped object tucked at the back/side instead of high.

WHY: the mission currently carries every object at retrieve_traj q_mid, whose TCP sits at
base-frame z = +0.548 m. That pose was chosen to clear the BASKET, and the basket has since
been removed from the robot (MISSION_NO_BASKET=1), so the constraint that forced the arm
high no longer exists. Carrying a mass 0.55 m above the base for the whole 30 m haul raises
the CoM and adds a pendulum the walk policy never saw in training — it costs balance for
nothing.

This solver searches the arm joint space for a pose that holds the object LOW and BEHIND /
BESIDE the torso (roughly where the basket used to sit), subject to: the object must not
intersect the torso, must not sweep the leg envelope, must keep enough ground clearance to
survive terrain relief, and must stay tool-down so a top-down pinch keeps its grip.

Outputs carry_traj.json: q_carry + its TCP, with the same keepout/body checks the retrieve
solver applies.

Run: CUDA_VISIBLE_DEVICES=0 python course_mission/scripts/solve_carry.py --headless --device cuda:0
"""

import argparse
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve()
CM = _SCRIPT.parents[1]
RW = _SCRIPT.parents[2]
for _p in (str(RW / "source" / "atec_rl_lab"), str(RW / "eunha_rl"), str(RW / "a2pro_rl"), str(RW)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Solve basket retrieve -> bin drop trajectory.")
parser.add_argument("--depth", type=str, default="0.24")
parser.add_argument("--mode", type=str, default="carry", choices=["carry", "tuck"],
                    help="carry = hold an object low/aft; tuck = empty-handed walking pose")
parser.add_argument("--probe", type=str, default="",
                    help="comma-separated 6 joint values: print its TCP and exit")
parser.add_argument("--poses_json", type=str, default=str(RW / "eunha_rl" / "s075" / "banks" / "crouch_poses.json"))
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--rounds", type=int, default=400)
parser.add_argument("--out", type=str, default=str(CM / "carry_traj.json"))

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.assets import ArticulationCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

import atec_rl_lab  # noqa: E402, F401
from a2pkg.a2piper import UNITREE_A2_PIPER_CFG  # noqa: E402

ARM_JOINTS = [f"arm_joint{i}" for i in range(1, 7)]
EE_BODY = "gripper_base"
TCP_OFF = 0.135

# Head sensors / front mast: same keepout box the retrieve solver uses.
KO_LO = np.array([0.218, -0.101, -0.037]) - 0.035
KO_HI = np.array([0.438, 0.169, 0.163]) + 0.035
TORSO_TOP_Z = 0.09          # torso top surface, base frame
HANG = 0.17                 # how far the object's bottom hangs below the TCP

# What we are solving FOR.
#   - TCP low: the whole point. 0.548 today; anything under ~0.30 is a real improvement.
#   - Ground clearance: the base rides ~0.50 m up when walking, so the object's bottom sits
#     at base_z + tcp_z - HANG. Terrain relief reaches ~0.03 m on pebble and the stairs
#     section is worse, so demand >= 0.18 m of clearance under the object.
WALK_BASE_Z = 0.50
MIN_GROUND_CLEAR = 0.30   # stairs/slope relief plus margin; 0.18 scraped on the first solve
# Leg swing envelope in the base frame: hips sit near x = +-0.28, y = +-0.19, and the
# thighs sweep forward/back through it. Keep the object out of that column entirely.
LEG_X = 0.20                # |x| beyond this is over a hip
LEG_Y = 0.13
# Minimum centre-to-centre distance between any arm link and any body link. Link origins
# are not surfaces, so this has to absorb both radii: the A2 torso is ~0.15 m half-width
# and the arm links ~0.05 m, hence 0.20 m rather than something that looks generous.
MIN_LINK_CLEAR = 0.20


@configclass
class SceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = None


def main():
    device = args_cli.device if getattr(args_cli, "device", None) else "cuda:0"
    torch.manual_seed(0)

    poses = json.loads(Path(args_cli.poses_json).read_text())
    crouch = poses["poses"][args_cli.depth]
    crouch_z = crouch["base_height_mean"]
    leg_pose = {k: (max(min(v, 1.0), -1.0) if "_hip_" in k else v)
                for k, v in crouch["joint_pos"].items()}

    robot_cfg = UNITREE_A2_PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # NO BASKET. That is the whole point — the basket is what forced the carry pose high,
    # and the shipping profile removes it (solution_v5_venue.py sets MISSION_NO_BASKET=1).
    robot_cfg.spawn.articulation_props.fix_root_link = True
    robot_cfg.init_state = robot_cfg.init_state.replace(
        pos=(0.0, 0.0, crouch_z),
        joint_pos={**leg_pose, **{j: 0.0 for j in ARM_JOINTS},
                   "arm_joint7": 0.05, "arm_joint8": -0.05},
        joint_vel={".*": 0.0},
    )

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=0.005, device=device, gravity=(0.0, 0.0, 0.0)))
    scene_cfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0)
    scene_cfg.robot = robot_cfg
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    robot = scene["robot"]
    dt = sim.get_physics_dt()

    names = list(robot.joint_names)
    arm_ids = torch.tensor([names.index(j) for j in ARM_JOINTS], device=device)
    ee_idx = robot.body_names.index(EE_BODY)
    # REAL LINK GEOMETRY, not one axis-aligned box. The first tuck solve returned a pose
    # with the TCP at z=0.061 — inside the torso shell — because the body model was a
    # single AABB with a 1 mm margin at its edge. Use the articulation's own link origins
    # as collision proxies instead: every ARM link must stay clear of every BODY link.
    bn = list(robot.body_names)
    # Only the DISTAL arm links can be swung into the body. arm_base and arm_link1 are
    # bolted to the torso, so their distance to base_link is fixed and small — including
    # them made every sampled configuration fail the clearance test and the solver returned
    # nothing at all. Check from link3 outwards.
    _DISTAL = ("arm_link3", "arm_link4", "arm_link5", "arm_link6", "arm_link7", "arm_link8")
    arm_link_ids = [i for i, b in enumerate(bn) if b in _DISTAL or b == EE_BODY]
    body_link_ids = [i for i, b in enumerate(bn)
                     if not b.startswith("arm_") and "gripper" not in b]
    print(f"[GEOM] arm links {[bn[i] for i in arm_link_ids]}", flush=True)
    print(f"[GEOM] body links {[bn[i] for i in body_link_ids]}", flush=True)
    limits = robot.data.soft_joint_pos_limits[0, arm_ids, :]
    center = (limits[:, 0] + limits[:, 1]) / 2
    half = (limits[:, 1] - limits[:, 0]) / 2 * 0.95
    q_template = robot.data.default_joint_pos.clone()
    zeros = torch.zeros_like(q_template)
    root_pos = robot.data.root_pos_w.clone()
    root_quat = robot.data.root_quat_w.clone()
    N = args_cli.num_envs

    def fk_batch(q_arm):
        q = q_template.clone()
        q[:, arm_ids] = q_arm
        robot.write_joint_state_to_sim(q, zeros)
        robot.set_joint_position_target(q)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(dt)
        ee_p = robot.data.body_pos_w[:, ee_idx]
        ee_q = robot.data.body_quat_w[:, ee_idx]
        ee_b = math_utils.quat_apply_inverse(root_quat, ee_p - root_pos)
        axis = math_utils.quat_apply(ee_q, torch.tensor([0.0, 0.0, 1.0], device=device).expand(N, 3))
        axis_b = math_utils.quat_apply_inverse(root_quat, axis)
        tcp_b = ee_b + axis_b * TCP_OFF
        # closest approach between any arm link and any body link, per sample
        bp = robot.data.body_pos_w
        al = bp[:, arm_link_ids, :]
        bl = bp[:, body_link_ids, :]
        dmin = torch.cdist(al, bl).amin(dim=(1, 2))
        return (ee_b.cpu().numpy(), axis_b.cpu().numpy(), tcp_b.cpu().numpy(),
                dmin.cpu().numpy())

    if args_cli.probe:
        q = [float(x) for x in args_cli.probe.split(",")]
        with torch.inference_mode():
            ee, ax, t, _dm = fk_batch(torch.tensor(q, device=device,
                                              dtype=torch.float32).unsqueeze(0).expand(N, 6))
        import math as _m
        tilt = _m.degrees(_m.acos(min(1.0, max(-1.0, -float(ax[0][2])))))
        print(f"[PROBE] q    = {q}", flush=True)
        print(f"[PROBE] TCP  = {np.round(t[0], 3).tolist()}  (base frame)", flush=True)
        print(f"[PROBE] axis = {np.round(ax[0], 2).tolist()}  tilt {tilt:.1f} deg from straight down",
              flush=True)
        return

    best = None
    with torch.inference_mode():
        for it in range(args_cli.rounds):
            q = center + (torch.rand(N, 6, device=device) * 2 - 1) * half
            ee, ax, t, dmin = fk_batch(q)
            qn = q.cpu().numpy()
            for i in range(N):
                tp, a = t[i], ax[i]
                if args_cli.mode == "tuck":
                    # EMPTY-HANDED WALKING POSE. Nothing is held, so there is no tool-down
                    # or ground-clearance requirement — the only goal is to get the arm's
                    # mass close to the base. This matters MORE than the carry pose because
                    # the robot walks empty for most of a run, and the pose it used had TCP
                    # [0.579, -0.096, 0.545]: 58 cm FORWARD and 55 cm UP, despite the code
                    # comment claiming it "folds the arm tight against the body". Mass that
                    # far forward pitches the robot on every step.
                    pts_t = np.stack([ee[i], tp])
                    if np.all((pts_t > KO_LO) & (pts_t < KO_HI), axis=1).any():
                        continue
                    # arm links must keep real clearance from the body/leg links
                    if dmin[i] < MIN_LINK_CLEAR:
                        continue
                    if tp[2] < -0.05:
                        continue        # below the belly, into the leg planes
                    r_t = float(np.hypot(tp[0], tp[1]))
                    score_t = r_t + 0.7 * abs(tp[2]) + 0.5 * max(0.0, tp[0])
                    if best is None or score_t < best[0]:
                        best = (score_t, qn[i].tolist(), tp.tolist(), a.tolist())
                    continue
                # tool must point DOWN or the pinch does not hold under gait accelerations
                # TOOL-DOWN, and strictly. The first solve returned axis z = -0.59, i.e. the
                # gripper tilted 54 deg off vertical — a lunchbox is held by an 8 mm pinch
                # on its lid taper and would slide straight out of that. Require within
                # ~25 deg of straight down.
                if a[2] > -0.90:
                    continue
                # object bottom must clear the ground with margin
                if WALK_BASE_Z + tp[2] - HANG < MIN_GROUND_CLEAR:
                    continue
                # never over the torso at a height where the hanging object would hit it
                over_torso = (-0.34 < tp[0] < 0.20) and abs(tp[1]) < 0.18
                if over_torso and (tp[2] - HANG) < TORSO_TOP_Z:
                    continue
                # Keep the object clear of the legs. The hips sit near x = +-0.28, so the
                # danger is a load hanging beside or between them low enough to be swept.
                # Anything below the torso underside must stay well inboard OR well outboard
                # of the leg planes, never in them.
                if (tp[2] - HANG) < -0.10 and 0.13 < abs(tp[1]) < 0.30 and abs(tp[0]) > 0.15:
                    continue
                # head sensor keepout, for both the wrist and the TCP
                pts = np.stack([ee[i], tp])
                if np.all((pts > KO_LO) & (pts < KO_HI), axis=1).any():
                    continue
                # SCORE: low CoM first, then a short moment arm, then behind rather than
                # in front (a load carried aft counteracts the forward pitch of walking).
                r = float(np.hypot(tp[0], tp[1]))
                score = tp[2] + 0.6 * r + 0.4 * max(0.0, tp[0])
                if best is None or score < best[0]:
                    best = (score, qn[i].tolist(), tp.tolist(), a.tolist())
            if best is not None and it % 5 == 4:
                print(f"[CARRY] iter {it + 1}: best TCP {np.round(best[2], 3).tolist()} "
                      f"score {best[0]:.3f}", flush=True)

    if best is None:
        raise SystemExit("NO carry candidate — relax the constraints")
    _, q_carry, t_carry, a_carry = best
    print(f"[CARRY] q_carry  = {np.round(q_carry, 3).tolist()}", flush=True)
    print(f"[CARRY] tcp_b    = {np.round(t_carry, 3).tolist()}  axis {np.round(a_carry, 2).tolist()}",
          flush=True)

    # validate the transit path from the grasp-lift waypoint to this carry pose
    rt = json.loads((Path(__file__).resolve().parents[1] / "retrieve_traj.json").read_text())

    def validate(qa, qb, steps=80):
        qa = torch.tensor(qa, device=device, dtype=torch.float32)
        qb = torch.tensor(qb, device=device, dtype=torch.float32)
        bad_ko, bad_body = 0, 0
        for s in range(steps + 1):
            qq = qa + (qb - qa) * s / steps
            ee_b, _, tcp_b, _dm = fk_batch(qq.unsqueeze(0).expand(N, 6))
            pts = np.stack([ee_b[0], tcp_b[0]])
            bad_ko += int(np.all((pts > KO_LO) & (pts < KO_HI), axis=1).any())
            tp = tcp_b[0]
            over_torso = (-0.34 < tp[0] < 0.20) and abs(tp[1]) < 0.18
            if over_torso and (tp[2] - HANG) < TORSO_TOP_Z:
                bad_body += 1
        return bad_ko, bad_body

    with torch.inference_mode():
        ko_in, body_in = validate(rt["q_mid"], q_carry)
        ko_out, body_out = validate(q_carry, rt["q_drop"])
    print(f"[CARRY] mid->carry keepout={ko_in} body={body_in}", flush=True)
    print(f"[CARRY] carry->drop keepout={ko_out} body={body_out}", flush=True)
    ok = (ko_in + ko_out) == 0 and (body_in + body_out) <= 2

    out = {
        "depth": args_cli.depth,
        "arm_joints": ARM_JOINTS,
        "q_carry": q_carry, "mode": args_cli.mode,
        "tcp_carry_b": t_carry,
        "axis_carry_b": a_carry,
        "tcp_mid_b_for_reference": rt["tcp_mid_b"],
        "checks": {"keepout_in": ko_in, "body_in": body_in,
                   "keepout_out": ko_out, "body_out": body_out, "valid": ok},
        "notes": ("low tucked carry, solved WITHOUT the basket; replaces retrieve q_mid "
                  "(TCP z 0.548) as the pose held while walking to the bin"),
    }
    dst = Path(__file__).resolve().parents[1] / ("carry_traj.json" if args_cli.mode == "carry" else "tuck_traj.json")
    dst.write_text(json.dumps(out, indent=2))
    print(f"[CARRY] wrote {dst}  valid={ok}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
