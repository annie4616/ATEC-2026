"""Smoke test: build the mission env, walk a little, run the detector, dump frames.

Verifies (cheaply, before a 20-minute full run):
  scene builds | robot + basket spawn on the course | cameras produce images |
  the colour detector finds the competition objects and rejects the red strips |
  arc/section geometry agrees with the ground truth object layout
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

parser = argparse.ArgumentParser(description="Mission env smoke test.")
parser.add_argument("--speed", type=str, default="s075")
parser.add_argument("--depth", type=str, default="0.24")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--per_type", type=int, default=4)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--walk_ckpt", type=str, default="")
parser.add_argument("--grasp_ckpt", type=str, default="")
parser.add_argument("--record_head", action="store_true")
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
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import course_mission  # noqa: F401, E402
from course_mission.launch import head_cam_cfg, HEAD_POS, HEAD_ROT, TASK, resolve_ckpts  # noqa: E402
from course_mission.mission import (  # noqa: E402
    ArmDriver, HeadCamMount, LegDriver, Perception, arc_of, section_of,
)
from course_mission.runner import ground_z_world  # noqa: E402
from course_mission.env_cfg import BIN_OF  # noqa: E402


def main():
    device = args_cli.device or "cuda:0"
    out = Path(args_cli.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ck = resolve_ckpts(args_cli)
    for k, v in ck.items():
        print(f"[SMOKE] {k}: {v} exists={Path(v).exists()}", flush=True)

    env_cfg = parse_env_cfg(TASK, device=device, num_envs=1)
    env_cfg.layout_seed = args_cli.seed
    env_cfg.per_type = args_cli.per_type
    env_cfg.scene.head_cam = head_cam_cfg()
    env_cfg.scene.overview_cam = CameraCfg(
        prim_path="/World/overview_cam", update_period=0.0, height=720, width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.1, 250.0)))

    env = gym.make(TASK, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    env.reset()
    robot = uenv.scene["robot"]
    print(f"[SMOKE] action dim = {uenv.action_manager.total_action_dim} "
          f"(terms {list(uenv.action_manager.active_terms)})", flush=True)
    print(f"[SMOKE] bodies: basket present = "
          f"{'basket' in robot.body_names} | joints={len(robot.joint_names)}", flush=True)

    # ground-truth layout vs the arc/section helpers
    bad = 0
    for name, cls, sec, ax, w, _z in env_cfg.mission_layout:
        obj = uenv.scene[name]
        p = obj.data.root_pos_w[0].cpu().numpy()
        ax2, w2 = arc_of(float(p[0]), float(p[1]))
        s2 = section_of(ax2)
        if s2 != sec or abs(ax2 - ax) > 0.6:
            bad += 1
            print(f"[SMOKE] MISMATCH {name} {cls}: planned {sec}@{ax:.2f}/{w:.2f} "
                  f"-> actual {s2}@{ax2:.2f}/{w2:.2f}", flush=True)
    print(f"[SMOKE] layout check: {len(env_cfg.mission_layout) - bad}/"
          f"{len(env_cfg.mission_layout)} objects land in their planned section", flush=True)

    legs = LegDriver(robot, device, ck["walk"], ck["crouch"], ck["adapt"])
    arm = ArmDriver(robot, device)
    head_mount = HeadCamMount(uenv.scene["head_cam"], robot, device, HEAD_POS, HEAD_ROT)
    percep = Perception(uenv.scene["head_cam"], device, mount=head_mount)
    travel_q = torch.tensor([0.0, 1.2, -1.2, 0.0, 0.6, 0.0], device=device)
    act = torch.zeros(1, uenv.action_manager.total_action_dim, device=device)

    ov = uenv.scene["overview_cam"]
    ov.set_world_poses_from_view(torch.tensor([[0.0, -34.0, 26.0]], device=uenv.device),
                                 torch.tensor([[0.0, 2.0, 0.0]], device=uenv.device))

    hits, frames = {}, 0
    tally = {"tp": 0, "fp": 0}
    perdet = []
    true_pos, visible = set(), set()
    with torch.inference_mode():
        for i in range(args_cli.steps):
            d = robot.data
            X, Y = float(d.root_pos_w[0, 0]), float(d.root_pos_w[0, 1])
            ax, w = arc_of(X, Y)
            import isaaclab.utils.math as mu
            yaw = float(mu.euler_xyz_from_quat(d.root_quat_w)[2][0])
            import course_sim.terrain as ct
            th = float(ct.curve_theta(ax))
            err = (th - 0.8 * np.clip(w, -0.8, 0.8) - yaw + np.pi) % (2 * np.pi) - np.pi
            cmd = torch.zeros(1, 3, device=device)
            cmd[0, 0] = 0.6
            cmd[0, 1] = float(np.clip(-0.5 * w, -0.3, 0.3))
            cmd[0, 2] = float(np.clip(1.2 * err, -0.7, 0.7))
            arm.set_joints(travel_q)
            arm.set_gripper(False)
            arm.write_gripper()
            head_mount.sync()
            act[:, :12] = legs.step(cmd)
            env.step(act)
            if i % 25 == 0:
                dets = percep.detect(ground_z_world(X, Y))
                for dd in dets:
                    hits[dd["cls"]] = hits.get(dd["cls"], 0) + 1
                    # precision: is there a real object of this class within 45 cm?
                    ok = False
                    for nm, cl, *_ in env_cfg.mission_layout:
                        if cl != dd["cls"]:
                            continue
                        gp = uenv.scene[nm].data.root_pos_w[0].cpu().numpy()
                        if float(np.linalg.norm(gp[:2] - dd["pos_w"][:2])) < 0.45:
                            ok = True
                            true_pos.add(nm)
                            break
                    tally["tp" if ok else "fp"] += 1
                    # Range-resolved accounting. A detection can fail the 45 cm test for
                    # two very different reasons: wrong class, or right class but a bad
                    # range estimate. The mission only ever acts on close detections, so
                    # separate the two before blaming the classifier.
                    best, bestnm = 1e9, ""
                    for nm, cl, *_ in env_cfg.mission_layout:
                        if cl != dd["cls"]:
                            continue
                        gp = uenv.scene[nm].data.root_pos_w[0].cpu().numpy()
                        e = float(np.linalg.norm(gp[:2] - dd["pos_w"][:2]))
                        if e < best:
                            best, bestnm = e, nm
                    perdet.append((float(dd["dist"]), dd["cls"], best, bestnm))
                    print(f"[DET] r={dd['dist']:.2f} cls={dd['cls']:9s} "
                          f"nearest_same_class_err={best:.2f} ({bestnm})", flush=True)
                # recall: objects inside the camera cone that were NOT reported
                import isaaclab.utils.math as _mu
                for nm, cl, *_ in env_cfg.mission_layout:
                    gp = uenv.scene[nm].data.root_pos_w[0].cpu().numpy()
                    rel = _mu.quat_apply_inverse(
                        d.root_quat_w,
                        torch.tensor(gp, dtype=torch.float32, device=device).reshape(1, 3)
                        - d.root_pos_w)[0].cpu().numpy()
                    if rel[0] > 0.5 and np.linalg.norm(rel[:2]) < 2.5 and abs(
                            np.arctan2(rel[1], rel[0])) < 0.45:
                        visible.add(nm)
                if i in (50, 200, 350) and frames < 3:
                    iio.imwrite(str(out / f"smoke_head_{i}.png"),
                                uenv.scene["head_cam"].data.output["rgb"][0, ..., :3].cpu().numpy())
                    iio.imwrite(str(out / f"smoke_overview_{i}.png"),
                                ov.data.output["rgb"][0, ..., :3].cpu().numpy())
                    frames += 1
                print(f"[SMOKE] step {i} arc={ax:.2f} lat={w:.2f} sec={section_of(ax)} "
                      f"z={float(d.root_pos_w[0,2]):.2f} dets={[ (x['cls'], round(x['dist'],2), round(x['above'],2)) for x in dets ]}",
                      flush=True)
    print("[SMOKE] ===== position error vs range =====", flush=True)
    for lo, hi in ((0.0, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 9.9)):
        b = [d for d in perdet if lo <= d[0] < hi]
        if b:
            errs = sorted(d[2] for d in b)
            good = sum(e < 0.45 for e in errs)
            print(f"[SMOKE] range {lo:.1f}-{hi:.1f} m: n={len(b):3d} "
                  f"within-45cm={good:3d} ({100*good/len(b):3.0f}%) "
                  f"median_err={errs[len(errs)//2]:.2f} m", flush=True)
    print(f"[SMOKE] detection tally over the run: {hits}", flush=True)
    tot = tally["tp"] + tally["fp"]
    print(f"[SMOKE] precision: {tally['tp']}/{tot} detections matched a real object "
          f"({100.0 * tally['tp'] / max(tot,1):.0f}%)", flush=True)
    print(f"[SMOKE] recall: {len(true_pos & visible)}/{len(visible)} of the objects that "
          f"passed through the camera cone were detected "
          f"(missed: {sorted(visible - true_pos)})", flush=True)
    print(f"[SMOKE] targets seen: "
          f"{[c for c in hits if BIN_OF.get(c)]} | distractors seen: "
          f"{[c for c in hits if not BIN_OF.get(c)]}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
