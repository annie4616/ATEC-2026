"""Task A straight-walking demo collector (heading-corrected oracle).

Teacher = the pretrained B2 walking policy (demo/policy.pt via WalkingOracle) with
a WORLD-YAW heading-feedback controller: every step we read the robot's true world
yaw (a privilege only available in the sim) and command
    wz = clip(-kyaw * (yaw - target_yaw), -wz_max, wz_max)
so the robot walks straight down +x. We record the observations a STUDENT policy
could actually see at eval time (head RGB + depth + proprio state) and the teacher's
3-D velocity command (vx, vy, wz) as the imitation target -- i.e. the student learns
to reproduce the heading correction WITHOUT world yaw.

Episode randomization: each episode perturbs the initial yaw by a small random
amount so the student sees a range of heading errors to correct.
"""
import math
import numpy as np
import torch

from task_d.walking_oracle import WalkingOracle  # reused as-is
from task_d.collector import _reset_asset_root, WARMUP_STEPS


def _world_yaw(robot) -> float:
    q = robot.data.root_quat_w[0]                       # w,x,y,z
    return math.atan2(2 * (float(q[0]) * float(q[3]) + float(q[1]) * float(q[2])),
                      1 - 2 * (float(q[2]) ** 2 + float(q[3]) ** 2))


def collect_one_demo(
    env,
    oracle: WalkingOracle,
    device: str,
    camera=None,
    depth_camera=None,
    vx: float = 2.0,
    kyaw: float = 2.0,
    wz_max: float = 1.0,
    target_yaw: float = 0.0,
    init_yaw_jitter: float = 0.0,
    n_steps: int = 500,
    fallen_z: float = 0.30,
    fallen_patience: int = 40,
    save_images: bool = True,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
) -> dict | None:
    """Roll out one straight-walking demo. Returns recorded dict or None if fallen."""
    env.reset()
    robot = env.unwrapped.scene["robot"]
    _reset_asset_root(robot)

    # Optionally perturb the initial yaw so the student learns to correct a range
    # of heading errors. Rotate the root quaternion about world +z by dyaw.
    if init_yaw_jitter > 0.0:
        dyaw = float((rng.random() * 2 - 1) * init_yaw_jitter) if rng is not None \
            else 0.0
        if abs(dyaw) > 1e-6:
            root = robot.data.default_root_state.clone()[0:1]
            half = dyaw * 0.5
            qz = torch.tensor([math.cos(half), 0.0, 0.0, math.sin(half)],
                              device=device, dtype=root.dtype)
            w0, x0, y0, z0 = root[0, 3], root[0, 4], root[0, 5], root[0, 6]
            w1, x1, y1, z1 = qz[0], qz[1], qz[2], qz[3]
            # quat multiply qz * q0
            root[0, 3] = w1 * w0 - x1 * x0 - y1 * y0 - z1 * z0
            root[0, 4] = w1 * x0 + x1 * w0 + y1 * z0 - z1 * y0
            root[0, 5] = w1 * y0 - x1 * z0 + y1 * w0 + z1 * x0
            root[0, 6] = w1 * z0 + x1 * y0 - y1 * x0 + z1 * w0
            robot.write_root_pose_to_sim(root[:, 0:7])
            robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))

    oracle.reset()
    for _ in range(WARMUP_STEPS):
        oracle.set_velocity_command(vx, 0.0, 0.0)
        env.step(oracle.compute_action())

    qpos_buf, qvel_buf, action_buf, velcmd_buf = [], [], [], []
    root_pos_buf, root_quat_buf = [], []
    frames_buf = [] if (save_images and camera is not None) else None
    depth_buf = [] if (save_images and depth_camera is not None) else None

    fallen = 0
    reached = False
    for step in range(n_steps):
        # --- teacher heading feedback (privileged world yaw) ---
        yaw = _world_yaw(robot)
        err = math.atan2(math.sin(yaw - target_yaw), math.cos(yaw - target_yaw))
        wz = max(-wz_max, min(wz_max, -kyaw * err))
        oracle.set_velocity_command(vx, 0.0, wz)
        action = oracle.compute_action()

        # --- record obs (student-visible) + teacher target BEFORE stepping ---
        qpos_buf.append(robot.data.joint_pos[0].cpu().numpy())
        qvel_buf.append(robot.data.joint_vel[0].cpu().numpy())
        action_buf.append(action[0].cpu().numpy())
        velcmd_buf.append(np.array([vx, 0.0, wz], dtype=np.float32))
        root_pos_buf.append(robot.data.root_pos_w[0].cpu().numpy())
        root_quat_buf.append(robot.data.root_quat_w[0].cpu().numpy())
        if frames_buf is not None:
            rgba = camera.data.output["rgb"][0].cpu().numpy()
            frames_buf.append(rgba[:, :, :3].astype(np.uint8))
        if depth_buf is not None:
            d = depth_camera.data.output["depth"][0].cpu().numpy()
            depth_buf.append(np.squeeze(d).astype(np.float32))

        _, _, terminated, truncated, _ = env.step(action)

        if verbose and step % 200 == 0:
            _rp = robot.data.root_pos_w[0]
            print(f"    [step {step:5d}] x={_rp[0].item():+.1f} y={_rp[1].item():+.2f} "
                  f"z={_rp[2].item():+.2f} yaw={math.degrees(yaw):+.0f}deg wz={wz:+.2f}",
                  flush=True)

        if robot.data.root_pos_w[0, 2].item() < fallen_z:
            fallen += 1
            if fallen >= fallen_patience:
                _rp = robot.data.root_pos_w[0]
                if verbose:
                    print(f"    [ABORT step {step}] FALLEN at x={_rp[0].item():+.2f} "
                          f"y={_rp[1].item():+.2f} z={_rp[2].item():+.2f} -> drop demo",
                          flush=True)
                return None
        else:
            fallen = 0

        if bool(terminated.any()) or bool(truncated.any()):
            reached = True                       # x>145 goal or time-out
            break

    if not qpos_buf:
        return None
    result = {
        "qpos":      np.stack(qpos_buf),          # (T,20)
        "qvel":      np.stack(qvel_buf),          # (T,20)
        "action":    np.stack(action_buf),        # (T,20)
        "vel_cmd":   np.stack(velcmd_buf),        # (T,3)  <-- ACT target
        "root_pos":  np.stack(root_pos_buf),      # (T,3)
        "root_quat": np.stack(root_quat_buf),     # (T,4)
        "reached":   reached,
    }
    if frames_buf is not None:
        result["frames"] = np.stack(frames_buf)   # (T,H,W,3) uint8
    if depth_buf is not None:
        result["depth"] = np.stack(depth_buf)     # (T,H,W) float32
    return result
