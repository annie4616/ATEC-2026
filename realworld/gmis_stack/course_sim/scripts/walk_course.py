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
    robot = uenv.scene["robot"]
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
            env.step(act)
            # follow-cam: smooth position follow behind the robot, horizon-stable
            rp = data.root_pos_w[0]
            fc.set_world_poses_from_view(
                (rp + torch.tensor([-2.4, -2.4, 1.6], device=uenv.device)).unsqueeze(0),
                (rp + torch.tensor([0.0, 0.0, 0.3], device=uenv.device)).unsqueeze(0),
            )
            w_ov.append_data(ov.data.output["rgb"][0].cpu().numpy())
            w_fc.append_data(fc.data.output["rgb"][0].cpu().numpy())
            if i % 200 == 0:
                x = data.root_pos_w[0, 0].item()
                z = data.root_pos_w[0, 2].item()
                print(f"[WALK] step {i}: x={x:.2f} z={z:.2f}", flush=True)
    w_ov.close()
    w_fc.close()
    print(f"[WALK] done: final x={robot.data.root_pos_w[0, 0].item():.2f}", flush=True)
    print("[WALK] videos: walk_overview.mp4 + walk_follow.mp4 (single run)", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
