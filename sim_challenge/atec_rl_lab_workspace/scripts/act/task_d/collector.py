"""Single-episode demo collection and success checking for Task D."""

from __future__ import annotations

import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv

from .config import (
    SUCCESS_X_THRESHOLD,
    MAX_EPISODE_STEPS,
    WARMUP_STEPS,
    FULL_ACTION_DIM,
)
from .walking_oracle import WalkingOracle
from .push_controller import PushIntoPitController


def _reset_asset_root(asset) -> None:
    """Force an asset's root pose+velocity back to its configured init_state.

    The Task D env registers a reset event only for the robot's JOINTS — neither
    the box nor the robot ROOT pose is reset, so both persist wherever the previous
    episode ended (box stuck in the pit -> 1-step false successes; robot left in the
    pit -> every following episode starts fallen and fails). We restore the root from
    default_root_state, which in this env is already the world-frame pose: the assets
    are spawned WITHOUT the env-origin offset (verified: box cfg x=-3, env_origin=-4.2,
    actual spawn x=-3), so we must NOT add env_origins (doing so flung the box to -7.2
    and broke physics). Velocities are zeroed.
    """
    root_state = asset.data.default_root_state[0:1].clone()
    root_state[:, 7:13] = 0.0  # zero linear + angular velocity
    asset.write_root_pose_to_sim(root_state[:, 0:7])
    asset.write_root_velocity_to_sim(root_state[:, 7:13])


# Success-criterion switch. Default = box-into-pit (for box-push-only demos). Set to
# True (e.g. by collect_demos_task_d.py --cross_runup) to require the robot to actually
# CROSS the pit and reach the finish line (robot x > 3.5) — the real Task D success.
REQUIRE_CROSSING = False


def robot_reached_goal(env: ManagerBasedRLEnv) -> bool:
    """Success criterion for keeping a demo.

    REQUIRE_CROSSING=True  -> robot crossed the pit (robot x > 3.5), the true finish.
    REQUIRE_CROSSING=False -> the BOX was pushed into the pit (box x in [-1.4, 0.7] and
                              settled, z < -0.5). Used for box-push-only demos.
    """
    robot = env.unwrapped.scene["robot"]
    if REQUIRE_CROSSING:
        return bool(robot.data.root_pos_w[0, 0].item() > SUCCESS_X_THRESHOLD)
    try:
        box = env.unwrapped.scene["box"]
    except (KeyError, Exception):
        return bool(robot.data.root_pos_w[0, 0].item() > SUCCESS_X_THRESHOLD)
    bx = box.data.root_pos_w[0, 0].item()
    bz = box.data.root_pos_w[0, 2].item()
    return bool(-1.4 <= bx <= 0.7 and bz < -0.5)


def collect_one_demo(
    env:    ManagerBasedRLEnv,
    oracle: WalkingOracle,
    device: str,
    camera=None,
    verbose: bool = False,
    push: PushIntoPitController | None = None,
    save_failed: bool = False,
    max_steps: int | None = None,
    frame_fn=None,
) -> dict | None:
    """Run one episode with the walking oracle and return recorded data.

    Returns a dict with keys, or None if the robot fell / episode ended early
    before reaching the goal:
      qpos     (T, 20)   absolute joint positions
      qvel     (T, 20)   joint velocities
      action   (T, 20)   env action fed to env.step
      root_pos (T, 3)    robot base position (world frame)
      root_quat(T, 4)    robot base orientation (w,x,y,z, world frame)
      frames   (T, H, W, 3)  RGB uint8 — only when camera is given
    """
    env.reset()
    robot = env.unwrapped.scene["robot"]
    try:
        box = env.unwrapped.scene["box"]
    except (KeyError, Exception):
        box = None
    # env.reset() does NOT restore the robot root pose or the box (only robot joints),
    # so each episode would otherwise start with both left wherever the last one ended
    # (robot fallen in the pit, box already in the pit). Restore both to init_state.
    _reset_asset_root(robot)
    if box is not None:
        _reset_asset_root(box)
    oracle.reset()
    if push is not None:
        push.reset()

    # Warm-up: let the robot settle on the terrain (not recorded).
    for _ in range(WARMUP_STEPS):
        action = oracle.compute_action()
        env.step(action)

    qpos_buf, qvel_buf, action_buf = [], [], []
    # Velocity command (vx, vy, wz) the push controller issued each step. This is the
    # HIERARCHICAL ACT target: ACT learns image -> velocity command, and a separate
    # walking RL policy (WalkingOracle) turns that command into joint actions. Learning
    # the 3-D command instead of the full 20-D action keeps ACT from having to reproduce
    # stable locomotion itself (which it can't from a handful of demos).
    velcmd_buf = []
    root_pos_buf, root_quat_buf = [], []
    # Record frames if either a robot-mounted camera or an external frame_fn
    # (e.g. a third-person follow viewport) is provided.
    capture_frames = (camera is not None) or (frame_fn is not None)
    frames_buf = [] if capture_frames else None
    # Also capture per-step DEPTH from a robot-mounted camera (not the third-person
    # frame_fn, which has no depth). RGB+D lets ACT estimate distance to the box/pit,
    # which the crossing run-up/charge timing depends on. Challenge obs include RGBD.
    capture_depth = (camera is not None) and ("depth" in camera.data.output)
    depth_buf = [] if capture_depth else None

    reached = False
    n_steps = max_steps if max_steps is not None else MAX_EPISODE_STEPS
    # Abort an episode once the robot has clearly fallen: the rough walking policy
    # cannot recover, so without this the robot lies on the terrain (root z low but
    # above the env's -0.4 fall threshold) and the episode spins until n_steps, which
    # previously left a hung process holding the GPU. If root z stays below
    # FALLEN_Z for FALLEN_PATIENCE consecutive steps, give up on this attempt.
    FALLEN_Z = 0.30
    FALLEN_PATIENCE = 60
    fallen_steps = 0
    # CROSSING-STALL abort: the cross-runup controller pushes the box in (~step 800)
    # then tries to cross; with the rough policy the cross usually fails and the robot
    # just spins at the pit edge until n_steps (~2 min wasted per attempt). Once the box
    # has SETTLED (z<-0.5), watch the robot's best forward x; if it doesn't improve for
    # STALL_PATIENCE steps, the cross has stalled -> give up early so we can try again.
    # (Only used when crossing is the success criterion; harmless otherwise.)
    STALL_PATIENCE = 250
    STALL_EPS = 0.05
    box_settled = False
    best_x = -1e9
    stall_steps = 0
    for step in range(n_steps):
        # Steer toward pushing the box into the pit (if a push controller is given).
        vx, vy, wz = 0.0, 0.0, 0.0
        if push is not None and box is not None:
            # The cross-runup controller also uses box_quat (to rotate the box so its
            # long side spans the pit); the plain controller's signature has no box_quat.
            # Pass it when accepted, otherwise fall back to the 3-arg call.
            try:
                vx, vy, wz = push.compute_velocity_command(
                    robot.data.root_pos_w[0],
                    box.data.root_pos_w[0],
                    robot.data.root_quat_w[0],
                    box.data.root_quat_w[0],
                )
            except TypeError:
                vx, vy, wz = push.compute_velocity_command(
                    robot.data.root_pos_w[0],
                    box.data.root_pos_w[0],
                    robot.data.root_quat_w[0],
                )
            oracle.set_velocity_command(vx, vy, wz)

        action = oracle.compute_action()

        if verbose and step % 50 == 0:
            rp = robot.data.root_pos_w[0]
            msg = (f"    [step {step:4d}] root=({rp[0].item():+.2f},{rp[1].item():+.2f},"
                   f"{rp[2].item():+.2f})")
            if box is not None:
                bp = box.data.root_pos_w[0]
                msg += (f" box=({bp[0].item():+.2f},{bp[1].item():+.2f},{bp[2].item():+.2f})")
            if push is not None:
                msg += f" state={push.state}"
            print(msg)

        # Record obs at time t and the action taken at time t (before stepping).
        qpos_buf.append(robot.data.joint_pos[0].cpu().numpy())
        qvel_buf.append(robot.data.joint_vel[0].cpu().numpy())
        root_pos_buf.append(robot.data.root_pos_w[0].cpu().numpy())
        root_quat_buf.append(robot.data.root_quat_w[0].cpu().numpy())
        action_buf.append(action[0].cpu().numpy())
        velcmd_buf.append(np.array([vx, vy, wz], dtype=np.float32))
        if frames_buf is not None:
            if frame_fn is not None:
                # Third-person follow frame from the simulation viewport.
                frame = frame_fn()
                if frame is not None:
                    frames_buf.append(frame[:, :, :3])
            else:
                rgba = camera.data.output["rgb"][0].cpu().numpy()
                frames_buf.append(rgba[:, :, :3])
        if depth_buf is not None:
            # (H, W) float32 metric depth; squeeze any trailing channel dim.
            d = camera.data.output["depth"][0].cpu().numpy()
            depth_buf.append(np.squeeze(d).astype(np.float32))

        _, _, terminated, truncated, _ = env.step(action)

        if robot_reached_goal(env):
            reached = True
            break

        # Fallen-robot abort (skip in record_attempt mode, which wants the full clip).
        if not save_failed:
            if robot.data.root_pos_w[0, 2].item() < FALLEN_Z:
                fallen_steps += 1
                if fallen_steps >= FALLEN_PATIENCE:
                    if verbose:
                        rp = robot.data.root_pos_w[0]
                        print(f"    [ABORT step {step}] robot fallen (z={rp[2].item():+.3f}) "
                              f"for {FALLEN_PATIENCE} steps -> giving up attempt")
                    break
            else:
                fallen_steps = 0

        # Crossing-stall abort (only when crossing is the success goal). Activates once
        # the box has settled in the pit; aborts if the robot's best forward x has not
        # advanced for STALL_PATIENCE steps.
        if REQUIRE_CROSSING and not save_failed and box is not None:
            if not box_settled and box.data.root_pos_w[0, 2].item() < -0.5:
                box_settled = True
            if box_settled:
                rx_now = robot.data.root_pos_w[0, 0].item()
                if rx_now > best_x + STALL_EPS:
                    best_x = rx_now
                    stall_steps = 0
                else:
                    stall_steps += 1
                    if stall_steps >= STALL_PATIENCE:
                        if verbose:
                            print(f"    [ABORT step {step}] cross stalled (best_x={best_x:+.2f}, "
                                  f"no progress {STALL_PATIENCE} steps) -> giving up attempt")
                        break

        # The env's own x_reached termination fires at x > 3.5 (= success),
        # so only treat a termination as failure when we have NOT reached goal.
        if bool(terminated.any()) or bool(truncated.any()):
            if robot_reached_goal(env):
                reached = True
            if verbose:
                rp = robot.data.root_pos_w[0]
                active = []
                tm = env.unwrapped.termination_manager
                for name in tm.active_terms:
                    try:
                        if bool(tm.get_term(name)[0].item()):
                            active.append(name)
                    except Exception:
                        pass
                print(f"    [END step {step}] terminated={bool(terminated.any())} "
                      f"truncated={bool(truncated.any())} reached={reached} "
                      f"root_x={rp[0].item():+.3f} root_z={rp[2].item():+.3f} "
                      f"active_terms={active}")
            break

    if not reached and not save_failed:
        return None
    if not qpos_buf:
        return None

    result = {
        "qpos":      np.stack(qpos_buf),
        "qvel":      np.stack(qvel_buf),
        "action":    np.stack(action_buf),
        "vel_cmd":   np.stack(velcmd_buf),   # (T, 3) push velocity command = ACT target
        "root_pos":  np.stack(root_pos_buf),
        "root_quat": np.stack(root_quat_buf),
        "reached":   reached,
    }
    if frames_buf is not None:
        result["frames"] = np.stack(frames_buf)
    if depth_buf is not None:
        result["depth"] = np.stack(depth_buf)   # (T, H, W) float32 metric depth
    return result
