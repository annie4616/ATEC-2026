"""Dump head-camera frames FACING THE BINS at a ladder of ranges, for offline
detect_bins gate autopsy: basket38 stood at the bin row through 3,000+ steps of the
shadow estimator's 250-step detection cadence and logged ZERO fixes."""
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
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--speed", type=str, default="s075")
parser.add_argument("--depth", type=str, default="0.24")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--per_type", type=int, default=4)
parser.add_argument("--walk_ckpt", type=str, default="")
parser.add_argument("--grasp_ckpt", type=str, default="")
parser.add_argument("--out_dir", type=str, default=str(RW / "course_mission" / "videos_deploy" / "bins_frames"))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app = AppLauncher(args_cli).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
import isaaclab.utils.math as mu  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import course_mission  # noqa: F401, E402
import course_sim.terrain as ct  # noqa: E402
from course_mission.launch import head_cam_cfg, TASK, resolve_ckpts  # noqa: E402
from course_mission.runner import MissionRunner, Profile, ground_z_arc, bin_targets  # noqa: E402
from course_mission.mission import arc_of  # noqa: E402


def main():
    device = args_cli.device or "cuda:0"
    out = Path(args_cli.out_dir); out.mkdir(parents=True, exist_ok=True)
    ck = resolve_ckpts(args_cli)
    _os.environ["MISSION_LAYOUT_SEED"] = str(args_cli.seed)
    env_cfg = parse_env_cfg(TASK, device=device, num_envs=1)
    env_cfg.scene.head_cam = head_cam_cfg()
    env = gym.make(TASK, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    env.reset()
    cams = {"head": uenv.scene["head_cam"]}
    r = MissionRunner(env, uenv, env_cfg, Profile(name="binsframe"), ck, cams, {}, out, device)
    robot = uenv.scene["robot"]
    zero = torch.zeros(1, 3, device=device)
    bx, _ = bin_targets()["green"]
    with torch.inference_mode():
        for _ in range(150):
            r._write_arm(r.travel_q, open_=False)
            r._step(r.legs.step(zero))
        for dist in (8.0, 5.0, 3.0, 1.5):
            ax = bx - dist
            ex, ey = ct.curve_map(ax, 0.0)
            th = float(ct.curve_theta(ax))
            root = torch.tensor([[float(ex), float(ey), ground_z_arc(ax) + 0.42]], device=device)
            quat = mu.quat_from_euler_xyz(torch.zeros(1, device=device), torch.zeros(1, device=device),
                                          torch.tensor([th], device=device))
            robot.write_root_pose_to_sim(torch.cat([root, quat], dim=1))
            robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))
            r.legs.fresh = True
            for _ in range(120):
                r._write_arm(r.travel_q, open_=False)
                r._step(r.legs.step(zero))
            cd = r.percep.cam.data
            cam_p = r.percep.mount.pos_w if r.percep.mount is not None else cd.pos_w[0].cpu().numpy()
            cam_R = r.percep.mount.R_ros if r.percep.mount is not None else cd.quat_w_ros[0].cpu().numpy()
            np.savez_compressed(out / f"bins_{dist:.1f}.npz",
                                rgb=cd.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8),
                                K=cd.intrinsic_matrices[0].cpu().numpy(),
                                cam_p=np.asarray(cam_p), cam_R=np.asarray(cam_R),
                                gz=ground_z_arc(ax))
            print(f"[BINSF] saved d={dist}", flush=True)
    print("[BINSF] DONE", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
