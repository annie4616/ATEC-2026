"""Compare the URDF-derived arm kinematics against the simulator's PhysX ones.

`course_mission/armkin.py` has to reproduce two things the robot cannot get off-simulator:

  * FK   — `ArmDriver.frame_pose()` / `tcp_w()`, currently PhysX body poses
  * J    — `ArmDriver._jacobian()`, currently `root_physx_view.get_jacobians()`

Self-consistency was already proved without a simulator (armkin._selftest: the Jacobian
is the finite difference of the FK to 4e-7). What that CANNOT prove is that the URDF
chain in armkin.py is the same chain PhysX is simulating. This script does, by driving
the real articulation to random arm poses and comparing.

    bash course_mission/scripts/run_check_armkin.sh
"""
import argparse
import os as _os
import sys
from pathlib import Path as _P

_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
_RW = _P(__file__).resolve().parents[2]
for _p in (str(_RW / "source" / "atec_rl_lab"), str(_RW / "eunha_rl"),
           str(_RW / "a2pro_rl"), str(_RW)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--poses", type=int, default=25)
parser.add_argument("--speed", type=str, default="s075")
parser.add_argument("--depth", type=str, default="0.24")
parser.add_argument("--per_type", type=int, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--walk_ckpt", type=str, default="")
parser.add_argument("--grasp_ckpt", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# The mission env config spawns a head camera, so Isaac refuses to build the scene
# without this even though nothing here renders.
args_cli.enable_cameras = True
app = AppLauncher(args_cli).app

import gymnasium as gym          # noqa: E402
import numpy as np               # noqa: E402
import torch                     # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from course_mission import armkin  # noqa: E402
from course_mission.launch import TASK  # noqa: E402
from course_mission.mission import ARM_JOINTS, ArmDriver  # noqa: E402


def main():
    device = "cuda:0"
    env_cfg = parse_env_cfg(TASK, device=device, num_envs=1)
    env_cfg.layout_seed = args_cli.seed
    env = gym.make(TASK, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    env.reset()
    robot = uenv.scene["robot"]
    arm = ArmDriver(robot, device)
    act = torch.zeros(1, uenv.action_manager.total_action_dim, device=device)

    rng = np.random.default_rng(0)
    dp, dR, dJl, dJa = [], [], [], []
    with torch.inference_mode():
        for k in range(args_cli.poses):
            q = np.array([rng.uniform(lo + 0.15, hi - 0.15)
                          for lo, hi in armkin.JOINT_LIMITS])
            tgt = torch.tensor(q, dtype=torch.float32, device=device).reshape(1, 6)
            for _ in range(90):                 # let the arm servo settle on the target
                arm.set_joints(tgt)
                arm.write_gripper()
                env.step(act)
            q_now = robot.data.joint_pos[0, arm.joint_ids].cpu().numpy()

            # --- FK: sim TCP in the base frame vs armkin
            p_b, q_b = arm.frame_pose()
            p_sim = p_b[0].cpu().numpy()
            p_mine, R_mine, _, _ = armkin.fk(q_now)
            dp.append(float(np.linalg.norm(p_sim - p_mine)))

            import isaaclab.utils.math as mu
            R_sim = mu.matrix_from_quat(q_b)[0].cpu().numpy()
            dR.append(float(np.abs(R_sim - R_mine).max()))

            # --- Jacobian
            J_sim = arm._jacobian()[0].cpu().numpy()
            J_mine = armkin.jacobian(q_now)
            dJl.append(float(np.abs(J_sim[:3] - J_mine[:3]).max()))
            dJa.append(float(np.abs(J_sim[3:] - J_mine[3:]).max()))
            print(f"[ARMKIN] pose {k:2d}  |dp| {dp[-1]*1000:6.2f} mm   "
                  f"max|dR| {dR[-1]:.4f}   max|dJ_lin| {dJl[-1]:.4f}   "
                  f"max|dJ_ang| {dJa[-1]:.4f}", flush=True)

    def s(name, v, scale=1.0, unit=""):
        a = np.array(v) * scale
        print(f"[ARMKIN] {name:22s} median {np.median(a):8.4f}{unit}  "
              f"max {a.max():8.4f}{unit}", flush=True)

    print("[ARMKIN] ===== summary =====", flush=True)
    s("TCP position error", dp, 1000.0, " mm")
    s("EE rotation error", dR)
    s("Jacobian linear err", dJl)
    s("Jacobian angular err", dJa)
    print("[ARMKIN] DONE", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
