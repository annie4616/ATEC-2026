"""Bin-phase probe: load the basket, walk the last metres to a chosen bin, retrieve the
object out of the basket and drop it in. Isolates the final stage of the mission so it
can be fixed without paying for a 45-minute full run.

Objects are teleported into the basket (the grasp+deposit stage is already verified by
probe_manip.py); everything from goto_bin onwards is the real code path.
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

parser = argparse.ArgumentParser(description="Bin approach + unload probe.")
parser.add_argument("--speed", type=str, default="s075")
parser.add_argument("--depth", type=str, default="0.24")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--per_type", type=int, default=4)
parser.add_argument("--walk_ckpt", type=str, default="")
parser.add_argument("--grasp_ckpt", type=str, default="")
parser.add_argument("--record_head", action="store_true")
parser.add_argument("--pairs", type=str, default="trash_5:green,trash_2:blue")
parser.add_argument("--out_dir", type=str, default=str(RW / "course_mission" / "videos_bins"))
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
from course_mission.runner import MissionRunner, Profile, ground_z_arc  # noqa: E402
from course_mission.mission import bin_targets  # noqa: E402


def main():
    device = args_cli.device or "cuda:0"
    out = Path(args_cli.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ck = resolve_ckpts(args_cli)

    env_cfg = parse_env_cfg(TASK, device=device, num_envs=1)
    env_cfg.layout_seed = args_cli.seed
    env_cfg.per_type = args_cli.per_type
    rec = dict(update_period=0.04, height=720, width=1280, data_types=["rgb"],
               spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.1, 250.0)))
    env_cfg.scene.overview_cam = CameraCfg(prim_path="/World/overview_cam", **rec)
    env_cfg.scene.follow_cam = CameraCfg(prim_path="/World/follow_cam", **rec)
    env_cfg.scene.head_cam = head_cam_cfg()

    env = gym.make(TASK, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    env.reset()
    cams = {"overview": uenv.scene["overview_cam"], "follow": uenv.scene["follow_cam"],
            "head": uenv.scene["head_cam"]}
    cams["overview"].set_world_poses_from_view(
        torch.tensor([[16.0, -6.0, 6.0]], device=uenv.device),
        torch.tensor([[19.0, 0.5, 0.3]], device=uenv.device))
    writers = {"follow": iio.get_writer(str(out / "probe_bins_follow.mp4"), fps=25),
               "overview": iio.get_writer(str(out / "probe_bins_overview.mp4"), fps=25)}

    prof = Profile(name="bins", max_picks=0)
    r = MissionRunner(env, uenv, env_cfg, prof, ck, cams, writers, out, device)
    r.cfg_depth = float(args_cli.depth)
    robot = uenv.scene["robot"]

    pairs = [p.split(":") for p in args_cli.pairs.split(",")]
    # stand the robot just before the bin zone, facing along the course
    start_ax = ct.SECTION_CENTERS["bins"] - 3.2
    th = float(ct.curve_theta(start_ax))
    sx, sy = ct.curve_map(start_ax, 0.0)
    root = torch.tensor([[float(sx), float(sy), ground_z_arc(start_ax) + 0.50]], device=device)
    quat = mu.quat_from_euler_xyz(torch.zeros(1, device=device), torch.zeros(1, device=device),
                                  torch.tensor([th], device=device))
    robot.write_root_pose_to_sim(torch.cat([root, quat], dim=1))
    robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))
    r.legs.fresh = True      # the RMA history must not carry pre-teleport state

    with torch.inference_mode():
        for _ in range(120):
            r._write_arm(r.travel_q, open_=False)
            r._step(r.legs.step(torch.zeros(1, 3, device=device)))
        # drop the payload into the basket (grasp+deposit already verified elsewhere)
        dep = r.deposit
        bc = dep["basket_center_b"]
        for k, (name, _colour) in enumerate(pairs):
            obj = uenv.scene[name]
            rp = robot.data.root_pos_w[0]
            rq = robot.data.root_quat_w[0]
            off = torch.tensor([[bc[0], bc[1] + 0.03 * k, 0.16]], device=device)
            pos = rp.unsqueeze(0) + mu.quat_apply(rq.unsqueeze(0), off)
            obj.write_root_pose_to_sim(torch.cat(
                [pos, torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)], dim=1))
            obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))
            r.carried.append((name, r.obj_cls[name]))
        for _ in range(80):
            r._write_arm(r.travel_q, open_=False)
            r._step(r.legs.step(torch.zeros(1, 3, device=device)))
        print(f"[BINS] payload loaded: {r.carried}", flush=True)

        results = []
        for name, colour in pairs:
            ok = r.goto_bin(colour)
            print(f"[BINS] goto_bin({colour}) -> {ok}", flush=True)
            if not ok:
                results.append((name, colour, "unreached"))
                continue
            res = r.unload_one(name, colour)
            print(f"[BINS] unload {name} -> {res}", flush=True)
            results.append((name, colour, res))

    print("[BINS] ===== summary =====", flush=True)
    for row in results:
        print(f"[BINS] {row}", flush=True)
    for w in writers.values():
        w.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
