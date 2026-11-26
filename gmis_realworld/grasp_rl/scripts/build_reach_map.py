"""Build the arm reachability map and the pre-grasp joint bank for B2+Piper in a crouch.

Method: spawn UNITREE_B2_PIPER_CFG with the root FIXED at the chosen crouch pose (from
banks/crouch_poses.json), gravity off, no ground. Legs are written to the crouch joint
angles; the 6 arm joints are sampled uniformly inside their soft limits (fingers open).
One physics step per batch updates link transforms; we read the gripper_base pose.

Outputs (grasp_rl/banks/):
  reach_map.npz      voxelized (2 cm) occupancy of reachable EE positions in base frame
  pregrasp_bank.npz  top-down-approach configs binned on a 2 cm ground-plane (x,y) grid:
                     per bin up to K arm-joint solutions + EE pose (for RSI resets)

Run:
  cd real_world_preliminary/ATEC2026_Simulation_Challenge
  CUDA_VISIBLE_DEVICES=5 ~/.conda/envs/isaaclab/bin/python \
      grasp_rl/scripts/build_reach_map.py --headless --device cuda:0 --depth 0.2
"""

import argparse
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve()
GRASP_RL = _SCRIPT.parents[1]
RW = _SCRIPT.parents[2]
for _p in (str(RW / "source" / "atec_rl_lab"), str(GRASP_RL)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Build B2+Piper reach map + pregrasp bank at a crouch pose.")
parser.add_argument("--depth", type=str, default="0.2", help="Crouch depth tag in crouch_poses.json.")
parser.add_argument("--poses_json", type=str, default=str(GRASP_RL / "banks" / "crouch_poses.json"))
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--rounds", type=int, default=400, help="Sampling rounds (samples = rounds * num_envs).")
parser.add_argument("--voxel", type=float, default=0.02, help="Voxel / ground-bin size [m].")
parser.add_argument("--topdown_deg", type=float, default=35.0, help="Max tilt from vertical for pregrasp bank.")
parser.add_argument("--bank_per_bin", type=int, default=5)
parser.add_argument("--ee_z_range", type=str, default="0.03,0.35", help="EE height band (base frame z + crouch z).")
parser.add_argument("--limit_margin", type=float, default=0.95, help="Fraction of soft joint range to sample.")
parser.add_argument("--out_dir", type=str, default=str(GRASP_RL / "banks"),
                    help="Output dir for reach_map/pregrasp_bank (new lineages use their own tree).")
parser.add_argument("--horizontal", action="store_true", default=False,
                    help="Bank of SIDE-approach configs (tool axis near horizontal) for rim pinches.")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json  # noqa: E402
import math  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.assets import ArticulationCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

import atec_rl_lab  # noqa: E402
from atec_rl_lab.assets.robots import UNITREE_B2_PIPER_CFG  # noqa: E402

if __import__("os").environ.get("GRASP_ROBOT") == "a2piper":
    _A2PRO = str(RW / "a2pro_rl")
    if _A2PRO not in sys.path:
        sys.path.insert(0, _A2PRO)
    from a2pkg.a2piper import UNITREE_A2_PIPER_CFG as UNITREE_B2_PIPER_CFG  # noqa: F811, E402
    print("[REACH] robot lineage: A2 Pro + Piper", flush=True)

assert str(RW) in atec_rl_lab.__file__, f"atec_rl_lab resolved OUTSIDE the working repo: {atec_rl_lab.__file__}"

ARM_JOINTS = ["arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5", "arm_joint6"]
_GO = float(__import__("os").environ.get("GRASP_GRIP_OPEN", "0.035"))
FINGER_OPEN = {"arm_joint7": _GO, "arm_joint8": -_GO}
EE_BODY = "gripper_base"


@configclass
class ReachSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = None  # set in main()


def main():
    device = args_cli.device if getattr(args_cli, "device", None) else "cuda:0"
    torch.manual_seed(args_cli.seed)

    poses = json.loads(Path(args_cli.poses_json).read_text())
    if args_cli.depth not in poses["poses"]:
        raise SystemExit(f"depth {args_cli.depth} not in {args_cli.poses_json}; have {sorted(poses['poses'])}")
    crouch = poses["poses"][args_cli.depth]
    crouch_z = crouch["base_height_mean"]
    # clamp hips just inside the B2 limit (+-0.87): extreme poses (e.g. depth 0.26) sit
    # exactly ON the limit and fail Articulation cfg validation
    leg_pose = {k: (max(min(v, 0.868), -0.868) if "_hip_" in k else v)
                for k, v in crouch["joint_pos"].items()}
    print(f"[REACH] crouch depth {args_cli.depth}: base z={crouch_z:.3f}", flush=True)

    # robot with FIXED root at the crouch pose, gravity off (pure kinematics + drives)
    robot_cfg = UNITREE_B2_PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot_cfg.spawn.articulation_props.fix_root_link = True
    robot_cfg.init_state = robot_cfg.init_state.replace(
        pos=(0.0, 0.0, crouch_z),
        joint_pos={**leg_pose, **{j: 0.0 for j in ARM_JOINTS}, **FINGER_OPEN},
        joint_vel={".*": 0.0},
    )

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=device, gravity=(0.0, 0.0, 0.0)))
    scene_cfg = ReachSceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0)
    scene_cfg.robot = robot_cfg
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    robot = scene["robot"]
    dt = sim.get_physics_dt()

    joint_names = list(robot.joint_names)
    arm_ids = torch.tensor([joint_names.index(j) for j in ARM_JOINTS], device=device)
    ee_idx = robot.body_names.index(EE_BODY)
    limits = robot.data.soft_joint_pos_limits[0, arm_ids, :]  # [6, 2]
    center = (limits[:, 0] + limits[:, 1]) / 2
    half = (limits[:, 1] - limits[:, 0]) / 2 * args_cli.limit_margin
    print(f"[REACH] arm soft limits (rad): {[tuple(round(v, 3) for v in row) for row in limits.tolist()]}", flush=True)

    # full-state template: crouch legs + open fingers, arm overwritten per round
    q_template = robot.data.default_joint_pos.clone()  # [N, J] (from init_state)
    zeros = torch.zeros_like(q_template)

    N = args_cli.num_envs
    ee_pos_all, ee_axis_all, q_all = [], [], []
    root_pos = robot.data.root_pos_w.clone()
    root_quat = robot.data.root_quat_w.clone()
    with torch.inference_mode():
        for r in range(args_cli.rounds):
            q = q_template.clone()
            q[:, arm_ids] = center + (torch.rand(N, 6, device=device) * 2 - 1) * half
            robot.write_joint_state_to_sim(q, zeros)
            robot.set_joint_position_target(q)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(dt)
            ee_pos_w = robot.data.body_pos_w[:, ee_idx]
            ee_quat_w = robot.data.body_quat_w[:, ee_idx]
            ee_pos_b = math_utils.quat_apply_inverse(root_quat, ee_pos_w - root_pos)
            # gripper approach axis (tool z) in base frame
            axis_w = math_utils.quat_apply(ee_quat_w, torch.tensor([0.0, 0.0, 1.0], device=device).expand(N, 3))
            axis_b = math_utils.quat_apply_inverse(root_quat, axis_w)
            ee_pos_all.append(ee_pos_b.cpu())
            ee_axis_all.append(axis_b.cpu())
            q_all.append(q[:, arm_ids].cpu())
            if (r + 1) % 100 == 0:
                print(f"[REACH] {(r + 1) * N} samples", flush=True)

    ee_pos = torch.cat(ee_pos_all).numpy()    # [M, 3] base frame
    ee_axis = torch.cat(ee_axis_all).numpy()  # [M, 3]
    q_arm = torch.cat(q_all).numpy()          # [M, 6]
    M = ee_pos.shape[0]

    # ---- reach map: voxel occupancy in base frame -------------------------------------------
    vox = args_cli.voxel
    lo = ee_pos.min(axis=0)
    hi = ee_pos.max(axis=0)
    dims = np.maximum(((hi - lo) / vox).astype(int) + 1, 1)
    idx = ((ee_pos - lo) / vox).astype(int)
    flat = np.ravel_multi_index(idx.T, dims)
    counts = np.bincount(flat, minlength=int(np.prod(dims))).reshape(dims)
    out_map = Path(args_cli.out_dir) / f"reach_map_{args_cli.depth}.npz"
    out_map.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_map, counts=counts, origin=lo, voxel=vox, bounds_lo=lo, bounds_hi=hi,
                        samples=M, crouch_depth=args_cli.depth, crouch_z=crouch_z)

    # ---- pregrasp bank: top-down configs binned on ground-plane (x, y) ----------------------
    z_lo, z_hi = (float(s) for s in args_cli.ee_z_range.split(","))
    world_z = ee_pos[:, 2] + crouch_z          # EE height above ground (root at crouch_z, flat)
    vert = -ee_axis[:, 2]                      # 1.0 = straight down
    if args_cli.horizontal:
        keep = (abs(vert) <= math.sin(math.radians(25.0))) & (world_z >= z_lo) & (world_z <= z_hi)
    else:
        keep = (vert >= math.cos(math.radians(args_cli.topdown_deg))) & (world_z >= z_lo) & (world_z <= z_hi)
    kp, ka, kq, kz = ee_pos[keep], ee_axis[keep], q_arm[keep], world_z[keep]
    kv = vert[keep]
    print(f"[REACH] samples={M}  top-down graspable={keep.sum()} ({keep.mean():.1%})", flush=True)

    bins = np.floor(kp[:, :2] / vox).astype(int)
    order = np.lexsort((-kv, bins[:, 1], bins[:, 0]))  # per bin, most-vertical first
    bins_s, kp_s, kq_s, kz_s, kv_s = bins[order], kp[order], kq[order], kz[order], kv[order]
    uniq, start = np.unique(bins_s, axis=0, return_index=True)
    sel = []
    for i, s in enumerate(start):
        e = start[i + 1] if i + 1 < len(start) else len(bins_s)
        sel.extend(range(s, min(s + args_cli.bank_per_bin, e)))
    sel = np.array(sel, dtype=int)
    out_bank = Path(args_cli.out_dir) / f"pregrasp_bank_{args_cli.depth}.npz"
    np.savez_compressed(out_bank, bin_xy=bins_s[sel], ee_pos_b=kp_s[sel], ee_world_z=kz_s[sel],
                        arm_q=kq_s[sel], verticality=kv_s[sel], voxel=vox,
                        crouch_depth=args_cli.depth, crouch_z=crouch_z,
                        arm_joint_names=np.array(ARM_JOINTS))

    occ = (counts > 0).sum()
    print(f"[REACH] reach map: {occ} voxels occupied, bbox x[{lo[0]:.2f},{hi[0]:.2f}] "
          f"y[{lo[1]:.2f},{hi[1]:.2f}] z[{lo[2]:.2f},{hi[2]:.2f}] (base frame) -> {out_map}", flush=True)
    print(f"[REACH] pregrasp bank: {len(uniq)} ground bins ({len(uniq) * vox * vox * 1e4:.0f} cm^2 coverage), "
          f"{len(sel)} configs -> {out_bank}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
