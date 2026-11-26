"""Walk the course and save a labelled head-camera dataset for offline detector work.

Every sim run costs 5-10 minutes on a GPU slot, which is a terrible iteration loop for a
perception bug. This records the raw inputs the detector sees (rgb, depth, intrinsics,
camera pose) together with the ground truth it should have produced (every object's world
position and class), so a detector change can be scored in under a second with
tune_detector.py and only the winner has to be confirmed in sim.
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

parser = argparse.ArgumentParser(description="Record a labelled head-cam dataset.")
parser.add_argument("--speed", type=str, default="s075")
parser.add_argument("--depth", type=str, default="0.24")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--per_type", type=int, default=4)
parser.add_argument("--steps", type=int, default=4000)
parser.add_argument("--every", type=int, default=25)
# Alternate walking and standing still. The projection analysis pointed at a
# pose/image mismatch while the base pitches, so frames have to be labelled
# with how fast the base was actually moving when each one was captured.
parser.add_argument("--stop_cycle", type=int, default=0,
                    help="walk N steps, then stand still N/4 steps, repeating")
parser.add_argument("--walk_ckpt", type=str, default="")
parser.add_argument("--grasp_ckpt", type=str, default="")
parser.add_argument("--out", type=str, default=str(RW / "course_mission" / "dataset"))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import course_mission  # noqa: F401, E402
import course_sim.terrain as ct  # noqa: E402
from course_mission.launch import head_cam_cfg, HEAD_POS, HEAD_ROT, TASK, resolve_ckpts  # noqa: E402
from course_mission.mission import (  # noqa: E402
    ArmDriver, HeadCamMount, LegDriver, arc_of, section_of,
)
from course_mission.runner import ground_z_world  # noqa: E402


def main():
    device = args_cli.device or "cuda:0"
    out = Path(args_cli.out)
    out.mkdir(parents=True, exist_ok=True)
    ck = resolve_ckpts(args_cli)

    env_cfg = parse_env_cfg(TASK, device=device, num_envs=1)
    env_cfg.layout_seed = args_cli.seed
    env_cfg.per_type = args_cli.per_type
    env_cfg.scene.head_cam = head_cam_cfg()

    env = gym.make(TASK, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    env.reset()
    robot = uenv.scene["robot"]
    cam = uenv.scene["head_cam"]
    legs = LegDriver(robot, device, ck["walk"], ck["crouch"], ck["adapt"])
    arm = ArmDriver(robot, device)
    head_mount = HeadCamMount(cam, robot, device, HEAD_POS, HEAD_ROT)
    travel_q = torch.tensor([0.0, 1.2, -1.2, 0.0, 0.6, 0.0], device=device)
    act = torch.zeros(1, uenv.action_manager.total_action_dim, device=device)

    names = [n for n, *_ in env_cfg.mission_layout]
    classes = [c for _n, c, *_ in env_cfg.mission_layout]
    saved = 0
    with torch.inference_mode():
        for i in range(args_cli.steps):
            d = robot.data
            X = float(d.root_pos_w[0, 0])
            Y = float(d.root_pos_w[0, 1])
            ax, w = arc_of(X, Y)
            yaw = float(torch.atan2(
                2 * (d.root_quat_w[0, 0] * d.root_quat_w[0, 3]
                     + d.root_quat_w[0, 1] * d.root_quat_w[0, 2]),
                1 - 2 * (d.root_quat_w[0, 2] ** 2 + d.root_quat_w[0, 3] ** 2)))
            th = float(ct.curve_theta(ax))
            err = (th - 0.8 * np.clip(w, -0.8, 0.8) - yaw + np.pi) % (2 * np.pi) - np.pi
            standing = False
            if args_cli.stop_cycle:
                phase = i % (args_cli.stop_cycle + args_cli.stop_cycle // 4)
                standing = phase >= args_cli.stop_cycle
            cmd = torch.zeros(1, 3, device=device)
            if not standing:
                cmd[0, 0] = 0.6
                cmd[0, 1] = float(np.clip(-0.5 * w, -0.3, 0.3))
                cmd[0, 2] = float(np.clip(1.2 * err, -0.7, 0.7))
            arm.set_joints(travel_q)
            arm.set_gripper(False)
            arm.write_gripper()
            head_mount.sync()
            act[:, :12] = legs.step(cmd)
            env.step(act)

            if i % args_cli.every:
                continue
            if args_cli.stop_cycle and standing and phase < args_cli.stop_cycle + 20:
                continue          # let the gait settle before calling it "still"
            objs = np.stack([uenv.scene[n].data.root_pos_w[0].cpu().numpy() for n in names])
            np.savez_compressed(
                str(out / f"f{saved:04d}.npz"),
                rgb=cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8),
                depth=cam.data.output["distance_to_image_plane"][0].cpu().numpy()
                .astype(np.float32).reshape(360, 640),
                K=cam.data.intrinsic_matrices[0].cpu().numpy(),
                # the sensor's own pose buffer never updates for this camera; the mount
                # holds the pose the image was actually rendered from
                cam_pos=head_mount.pos_w,
                cam_rot=head_mount.R_ros,
                root_pos=d.root_pos_w[0].cpu().numpy(),
                root_quat=d.root_quat_w[0].cpu().numpy(),
                ground_z=np.float32(ground_z_world(X, Y)),
                standing=np.bool_(standing),
                ang_vel=d.root_ang_vel_b[0].cpu().numpy(),
                lin_vel=d.root_lin_vel_b[0].cpu().numpy(),
                arc=np.float32(ax), lat=np.float32(w),
                section=np.array(section_of(ax)),
                names=np.array(names), classes=np.array(classes), objs=objs)
            saved += 1
            if saved % 10 == 0:
                print(f"[COLLECT] {saved} frames (step {i}, arc {ax:.1f}, "
                      f"{section_of(ax)})", flush=True)

    print(f"[COLLECT] done: {saved} frames in {out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
