"""Dump head-cam frames WHILE WALKING for offline detect_belt autopsy: the honest
nav runs measured an 8 % belt-fix rate on the move (vs 52 % benchmark), which starves
navfuse and sends the robot straight off the first bend."""
import argparse, sys
from pathlib import Path
_S = Path(__file__).resolve()
RW = _S.parents[2]
for _p in (str(RW / "source" / "atec_rl_lab"), str(RW / "eunha_rl"), str(RW / "a2pro_rl"), str(RW)):
    if _p in sys.path: sys.path.remove(_p)
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
parser.add_argument("--out_dir", type=str, default=str(RW / "course_mission" / "videos_deploy" / "belt_walk"))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app = AppLauncher(args_cli).app
import numpy as np, torch, gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import course_mission  # noqa: F401, E402
from course_mission.launch import head_cam_cfg, TASK, resolve_ckpts  # noqa: E402
from course_mission.runner import MissionRunner, Profile, ground_z_arc  # noqa: E402
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
    r = MissionRunner(env, uenv, env_cfg, Profile(name="beltwalk"), ck, cams, {}, out, device)
    zero = torch.zeros(1, 3, device=device)
    with torch.inference_mode():
        for _ in range(150):
            r._write_arm(r.travel_q, open_=False)
            r._step(r.legs.step(zero))
        n = 0
        for i in range(1400):
            cmd, ax, w = r.walk_cmd(0.0, vx=0.55)
            r._write_arm(r.travel_q, open_=False)
            r._step(r.legs.step(cmd))
            if i % 20 == 0 and n < 60:
                cd = r.percep.cam.data
                cam_p = r.percep.mount.pos_w if r.percep.mount is not None else cd.pos_w[0].cpu().numpy()
                cam_R = r.percep.mount.R_ros if r.percep.mount is not None else cd.quat_w_ros[0].cpu().numpy()
                np.savez_compressed(out / f"bw_{n:03d}.npz",
                    rgb=cd.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8),
                    K=cd.intrinsic_matrices[0].cpu().numpy(),
                    cam_p=np.asarray(cam_p), cam_R=np.asarray(cam_R),
                    gz=ground_z_arc(ax), ax=ax, w=w)
                n += 1
        print(f"[BELTWALK] saved {n} frames, final arc {ax:.1f}", flush=True)
    env.close()

if __name__ == "__main__":
    main()
    app.close()
