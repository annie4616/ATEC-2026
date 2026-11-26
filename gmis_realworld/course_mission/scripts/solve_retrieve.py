"""Solve the basket->bin RETRIEVE trajectory (the reverse of deposit_traj.json).

deposit_traj.json drops the object from above the basket opening (TCP z=0.255). To put
it in a competition bin at the end of the course the arm has to reach back DOWN INTO the
basket, close on the settled object, lift it out over the torso and hold it in front,
above the 0.55 m bin rim. Same sampling-FK method as solve_deposit.py.

Outputs retrieve_traj.json: q_pick (deep in the basket), q_mid, q_drop (front-high, TCP
well past the bin rim), plus the settle-height statistics used to pick the pick depth.

Run: CUDA_VISIBLE_DEVICES=0 python course_mission/scripts/solve_retrieve.py --headless --device cuda:0
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
parser.add_argument("--poses_json", type=str, default=str(RW / "eunha_rl" / "s075" / "banks" / "crouch_poses.json"))
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--rounds", type=int, default=400)
parser.add_argument("--out", type=str, default=str(CM / "retrieve_traj.json"))
parser.add_argument("--scan", action="store_true", help="Scan reachable depths inside the basket only.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import itertools  # noqa: E402
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
import os as _os
BASKET_C = np.array([float(_os.environ.get("BASKET_CX", "-0.07")),
                     float(_os.environ.get("BASKET_CY", "-0.27"))])
SCAN_HX = float(_os.environ.get("BASKET_SCAN_HX", "0.12"))
SCAN_HY = float(_os.environ.get("BASKET_SCAN_HY", "0.09"))
BASKET_OPEN_Z = 0.218
BASKET_FLOOR_Z = 0.098          # basket interior floor, base frame
KO_LO = np.array([0.218, -0.101, -0.037]) - 0.035
KO_HI = np.array([0.438, 0.169, 0.163]) + 0.035
TORSO_TOP_Z = 0.09
BOTTLE_HANG = 0.17
# bin rim is 0.55 m in WORLD; standing base z ~0.52 -> need TCP base-z > 0.10 with margin.
# Keep the drop pose front-high so the object clears the rim by >=0.2 m while standing.
# The robot cannot stand closer than ~0.5 m from the bin centre — its front feet reach
# the bin's base plate — so the DROP pose has to carry the TCP 0.55-0.72 m ahead for the
# object to fall INSIDE the 0.40 x 0.26 m opening instead of just in front of it.
DROP_X = (0.55, 0.72)
DROP_Z = (0.28, 0.46)


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
    robot_cfg.spawn.usd_path = _os.environ.get(
        "MISSION_BASKET_USD", str(RW / "eunha_rl" / "assets" / "a2_piperx_basket.usd"))
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
        return ee_b.cpu(), axis_b.cpu(), tcp_b.cpu()

    if args_cli.scan:
        # how deep into the basket can the TCP go, tool pointing down?
        pts = []
        with torch.inference_mode():
            for _ in range(200):
                q_arm = center + (torch.rand(N, 6, device=device) * 2 - 1) * half
                _, axis_b, tcp_b = fk_batch(q_arm)
                t = tcp_b.numpy(); a = axis_b.numpy()
                m = (np.abs(t[:, 0] - BASKET_C[0]) < SCAN_HX) & (np.abs(t[:, 1] - BASKET_C[1]) < SCAN_HY) \
                    & (t[:, 2] > BASKET_FLOOR_Z) & (t[:, 2] < BASKET_OPEN_Z + 0.12) & (a[:, 2] < -0.70)
                pts.append(np.concatenate([t[m], a[m, 2:3]], axis=1))
        P = np.concatenate(pts)
        print(f"[SCAN] in-basket tool-down samples: {len(P)}", flush=True)
        for zlo in np.arange(BASKET_FLOOR_Z, BASKET_OPEN_Z + 0.12, 0.02):
            m = (P[:, 2] >= zlo) & (P[:, 2] < zlo + 0.02)
            if m.sum():
                q = P[m]
                print(f"[SCAN] z[{zlo:.3f},{zlo+0.02:.3f}): n={int(m.sum())} "
                      f"x {q[:,0].min():.3f}~{q[:,0].max():.3f} y {q[:,1].min():.3f}~{q[:,1].max():.3f} "
                      f"best_vert={q[:,3].min():.2f}", flush=True)
        raise SystemExit(0)

    cand = {"pick": [], "mid": [], "drop": []}
    with torch.inference_mode():
        for _ in range(args_cli.rounds):
            q_arm = center + (torch.rand(N, 6, device=device) * 2 - 1) * half
            _, axis_b, tcp_b = fk_batch(q_arm)
            t = tcp_b.numpy(); a = axis_b.numpy(); qn = q_arm.cpu().numpy()
            # pick: TCP LOW inside the basket (close on the settled object), tool down
            m = (np.abs(t[:, 0] - BASKET_C[0]) < 0.085) & (np.abs(t[:, 1] - BASKET_C[1]) < 0.055) \
                & (t[:, 2] > BASKET_FLOOR_Z + 0.03) & (t[:, 2] < BASKET_FLOOR_Z + 0.10) & (a[:, 2] < -0.80)
            for i in np.nonzero(m)[0]:
                # prefer deepest + most vertical + closest to basket center
                score = -(t[i, 2]) - 0.3 * abs(a[i, 2] + 1) - 0.3 * abs(t[i, 0] - BASKET_C[0])
                cand["pick"].append((score, qn[i].tolist(), t[i].tolist()))
            # mid: high transit over the torso (same box as the deposit solve)
            m = (t[:, 0] > -0.10) & (t[:, 0] < 0.12) & (np.abs(t[:, 1]) < 0.10) \
                & (t[:, 2] > 0.38) & (t[:, 2] < 0.55)
            for i in np.nonzero(m)[0]:
                cand["mid"].append((t[i, 2], qn[i].tolist(), t[i].tolist()))
            # drop: front-high over the bin opening, tool down so the object falls in
            m = (t[:, 0] > DROP_X[0]) & (t[:, 0] < DROP_X[1]) & (np.abs(t[:, 1]) < 0.09) \
                & (t[:, 2] > DROP_Z[0]) & (t[:, 2] < DROP_Z[1]) & (a[:, 2] < -0.6)
            for i in np.nonzero(m)[0]:
                cand["drop"].append((t[i, 2] - abs(t[i, 1]), qn[i].tolist(), t[i].tolist()))
            if all(len(v) > 40 for v in cand.values()):
                break

    for k, v in cand.items():
        if not v:
            raise SystemExit(f"NO candidates for '{k}' — relax the box or move the basket")
        v.sort(key=lambda x: -x[0])
        print(f"[RET] {k}: {len(v)} candidates, best TCP={[round(x, 3) for x in v[0][2]]}", flush=True)

    best = None
    for (_, qp, tp), (_, qm, tm), (_, qd, td) in itertools.product(
            cand["pick"][:12], cand["mid"][:12], cand["drop"][:12]):
        d = np.abs(np.array(qp) - np.array(qm)).sum() + np.abs(np.array(qm) - np.array(qd)).sum()
        if best is None or d < best[0]:
            best = (d, qp, qm, qd, tp, tm, td)
    _, q_pick, q_mid, q_drop, t_pick, t_mid, t_drop = best
    print(f"[RET] chosen travel={best[0]:.2f} rad", flush=True)

    def validate(qa, qb, steps=80):
        qa = torch.tensor(qa, device=device); qb = torch.tensor(qb, device=device)
        bad_ko, bad_body = 0, 0
        for s in range(steps + 1):
            q = qa + (qb - qa) * s / steps
            ee_b, _, tcp_b = fk_batch(q.unsqueeze(0).expand(N, 6))
            pts = np.stack([ee_b[0].numpy(), tcp_b[0].numpy()])
            bad_ko += int(np.all((pts > KO_LO) & (pts < KO_HI), axis=1).any())
            t = tcp_b[0].numpy()
            over_torso = (-0.34 < t[0] < 0.20) and abs(t[1]) < 0.18
            in_basket_col = (abs(t[0] - BASKET_C[0]) < 0.16 and abs(t[1] - BASKET_C[1]) < 0.14)
            if over_torso and (t[2] - BOTTLE_HANG) < TORSO_TOP_Z and not in_basket_col:
                bad_body += 1
        return bad_ko, bad_body

    with torch.inference_mode():
        ko1, body1 = validate(q_pick, q_mid)
        ko2, body2 = validate(q_mid, q_drop)
    print(f"[RET] path pick->mid: keepout_hits={ko1} body_hits={body1}", flush=True)
    print(f"[RET] path mid->drop: keepout_hits={ko2} body_hits={body2}", flush=True)
    ok = (ko1 + ko2) == 0 and (body1 + body2) <= 2

    out = {
        "depth": args_cli.depth,
        "arm_joints": ARM_JOINTS,
        "q_pick": q_pick, "q_mid": q_mid, "q_drop": q_drop,
        "tcp_pick_b": t_pick, "tcp_mid_b": t_mid, "tcp_drop_b": t_drop,
        "basket_center_b": BASKET_C.tolist(), "basket_floor_z_b": BASKET_FLOOR_Z,
        "checks": {"keepout_hits": ko1 + ko2, "body_hits": body1 + body2, "valid": ok},
        "notes": "basket pick -> over-torso mid -> front-high drop above the bin rim",
    }
    Path(args_cli.out).write_text(json.dumps(out, indent=2))
    print(f"[RET] {'VALID' if ok else 'INVALID'} — written {args_cli.out}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
