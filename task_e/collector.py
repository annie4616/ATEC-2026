"""Single-episode demo collection and success checking for Task E."""
#/home/yewon/ATEC2026_Simulation_Challenge/scripts/act/task_e/collector.py

import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv
from atec_rl_lab.utils import CartesianController
from atec_rl_lab.tasks.task_e.env_cfg import (
    TABLE_CENTER_X, TABLE_CENTER_Y, TABLE_TOP_Z,
    BASKET_CENTER_X, BASKET_CENTER_Y,
)

from .config_all import (
    ACTION_SCALE,
    GRIPPER_OPEN_POS, GRIPPER_CLOSE_POS,
    RETRACT_POS_X, RETRACT_POS_Y, CARRY_Z,
    DEFAULT_PLACE_QUAT_W,
    BASKET_IN_X, BASKET_IN_Y,
    OBJ_SPAWN_X_MIN, OBJ_SPAWN_X_MAX, OBJ_SPAWN_Z, OBJ_SPAWN_Y_BANDS,
    OBJ_HALF_EXTENTS, OBJ_BBOX_MARGIN,
    WARMUP_STEPS, SETTLE_STEPS,
)
from .state_machine import PickPlaceStateMachine


def _rerandomize_objects(env: ManagerBasedRLEnv, rng: np.random.Generator) -> None:
    """Place each object randomly with AABB-based overlap rejection."""
    placed: dict[int, tuple[float, float]] = {}  # obj_idx -> (x, y)

    for obj_idx in [1, 2, 3]:
        obj = env.unwrapped.scene.rigid_objects[f"object_{obj_idx}"]
        y_min, y_max = OBJ_SPAWN_Y_BANDS[obj_idx]
        hx, hy = OBJ_HALF_EXTENTS[obj_idx]

        x = y = None
        for _ in range(200):
            cx = float(rng.uniform(OBJ_SPAWN_X_MIN, OBJ_SPAWN_X_MAX))
            cy = float(rng.uniform(y_min, y_max))
            # AABB overlap check against all already-placed objects
            ok = all(
                abs(cx - px) >= hx + OBJ_HALF_EXTENTS[pi][0] + OBJ_BBOX_MARGIN or
                abs(cy - py) >= hy + OBJ_HALF_EXTENTS[pi][1] + OBJ_BBOX_MARGIN
                for pi, (px, py) in placed.items()
            )
            if ok:
                x, y = cx, cy
                break

        if x is None:  # fallback: band centre
            x = (OBJ_SPAWN_X_MIN + OBJ_SPAWN_X_MAX) / 2.0
            y = (y_min + y_max) / 2.0

        placed[obj_idx] = (x, y)
        state = obj.data.default_root_state[0:1].clone()
        state[0, 0] = x
        state[0, 1] = y
        state[0, 2] = OBJ_SPAWN_Z
        state[0, 7:] = 0.0   # zero velocities
        obj.write_root_state_to_sim(state)

    env.unwrapped.scene.write_data_to_sim()
    env.unwrapped.sim.forward()


_BASKET_MAX_Z = TABLE_TOP_Z + 0.1   # object must be below this to count as inside

def check_objects_in_basket(env: ManagerBasedRLEnv, pick_objects: list[int]) -> bool:
    """Return True only if every picked object is inside the basket region and settled."""
    for obj_idx in pick_objects:
        pos = env.unwrapped.scene.rigid_objects[f"object_{obj_idx}"].data.root_pos_w[0]
        if (abs(pos[0].item() - BASKET_CENTER_X) > BASKET_IN_X or
                abs(pos[1].item() - BASKET_CENTER_Y) > BASKET_IN_Y or
                pos[2].item() > _BASKET_MAX_Z):
            return False
    return True


def collect_one_demo(
    env:         ManagerBasedRLEnv,
    robot,
    ik_ctrl:     CartesianController,
    arm_ids:     list[int],
    gripper_ids: list[int],
    pick_objects: list[int],
    device:      str,
    default_jpos: torch.Tensor,
    rng:         np.random.Generator,
    camera=None,
) -> dict | None:
    """Run one full episode and return recorded data, or None on early termination.

    Returns a dict with keys:
      qpos    (T, 8)        absolute joint positions
      qvel    (T, 8)        joint velocities
      ee_pos  (T, 3)        end-effector position (world frame)
      ee_quat (T, 4)        end-effector quaternion (w,x,y,z)
      action  (T, 8)        env action = (joint_target - default_jpos) / ACTION_SCALE
      frames  (T, H, W, 3)  RGB uint8 — only present when camera is given
    """
    env.reset()
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos,
        torch.zeros_like(robot.data.default_joint_vel),
    )

    _rerandomize_objects(env, rng)      # write new object positions to sim + sim.forward()
    default_jpos = robot.data.default_joint_pos.clone()

    ee_home = torch.tensor([[RETRACT_POS_X, RETRACT_POS_Y, CARRY_Z]],
                            dtype=torch.float32, device=device)
    eq_home = torch.tensor([DEFAULT_PLACE_QUAT_W], dtype=torch.float32, device=device)
    g_open  = torch.tensor([GRIPPER_OPEN_POS], dtype=torch.float32, device=device)

    robot.update(dt=env.unwrapped.physics_dt)
    ik_ctrl.reset()

    # Warm-up: drive arm to HOME position (not recorded)
    for _ in range(WARMUP_STEPS):
        _step_to(env, robot, ik_ctrl, arm_ids, gripper_ids,
                 ee_home, eq_home, g_open, default_jpos)

    # Pre-compute grasp quaternions from actual object orientations after reset.
    # compute_grasp_quat() keeps a vertical (top-down) approach while rotating the
    # gripper (joint6) so its jaw opens PERPENDICULAR to each object's long axis.
    sm = PickPlaceStateMachine(pick_objects, device)
    for obj_idx in pick_objects:
        obj_quat = env.unwrapped.scene.rigid_objects[f"object_{obj_idx}"] \
                       .data.root_state_w[0, 3:7]
        sm.set_grasp_quat(obj_idx, obj_quat)

    if 2 in pick_objects:
        from .state_machine import is_upright, compute_grasp_quat_upright, compute_grasp_quat_fallen
        from isaaclab.utils.math import matrix_from_quat
        mustard_quat = env.unwrapped.scene.rigid_objects["object_2"] \
                            .data.root_state_w[0, 3:7]
        R = matrix_from_quat(mustard_quat.unsqueeze(0)).squeeze(0)
        #print(f"[DEBUG] mustard quat: {mustard_quat.cpu().numpy()}")
        #print(f"[DEBUG] R[:,0] (local X in world): {R[:,0].cpu().numpy()}")
        #print(f"[DEBUG] R[:,1] (local Y in world): {R[:,1].cpu().numpy()}")
        #print(f"[DEBUG] R[:,2] (local Z in world): {R[:,2].cpu().numpy()}")   

        if is_upright(mustard_quat):
            print("[INFO] Mustard starts upright - skipping knock")
            sm.set_mustard_needs_knock(False)
            sm.set_mustard_upright(True)
            new_quat = compute_grasp_quat_upright(device)
            print(f"[DEBUG] upright grasp quat: {new_quat.cpu().numpy()}")  # ← 추가
            sm._grasp_quat_cache[2] = new_quat
        else:
            print("[INFO] Mustard starts fallen - need knock? No, already fallen")
            sm.set_mustard_needs_knock(False)  # 이미 누워있으니 knock 불필요
            sm.set_mustard_upright(False)
            sm._grasp_quat_cache[2] = compute_grasp_quat_fallen(mustard_quat, device)

    # Settle
    for _ in range(SETTLE_STEPS):
        _step_to(env, robot, ik_ctrl, arm_ids, gripper_ids,
                 ee_home, eq_home, g_open, default_jpos)

    ik_ctrl.reset()

    # ---- Recording loop ---- #
    qpos_buf, qvel_buf, ee_pos_buf, ee_quat_buf, action_buf = [], [], [], [], []
    frames_buf = [] if camera is not None else None
    prev_state = None

    while not sm.done:
        obj_pos_w = env.unwrapped.scene.rigid_objects[sm.current_object_key] \
                        .data.root_pos_w[0].clone()
        cur_state = sm._current_state_order()[sm._state_idx]

        # WAIT → REACH 전환 시점에 쓰러진 병 quat 재계산
        if prev_state == "WAIT" and cur_state == "REACH":
            fallen_state = env.unwrapped.scene.rigid_objects["object_2"].data.root_state_w[0]
            fallen_quat = fallen_state[3:7]
            
            from .state_machine import is_upright, compute_grasp_quat_upright, compute_grasp_quat_fallen
            
            if is_upright(fallen_quat):
                # knock 후에도 안 쓰러진 경우 (드물지만)
                sm.set_mustard_upright(True)
                new_quat = compute_grasp_quat_upright(device)
            else:
                sm.set_mustard_upright(False)
                new_quat = compute_grasp_quat_fallen(fallen_quat, device)
            
            sm._grasp_quat_cache[2] = new_quat

        prev_state = cur_state
        ee_pos_des, ee_quat_des, gripper_cmd = sm.tick(obj_pos_w)
        #print(f"[DEBUG] ee_quat_des: {ee_quat_des.cpu().numpy()}")
        #print(f"[DEBUG] actual ee_quat: {ik_ctrl.ee_quat_w[0].cpu().numpy()}")
        #print(f"[DEBUG] state={cur_state} gripper_cmd={gripper_cmd}")
        arm_jpos_des   = ik_ctrl.compute(ee_pos_des.unsqueeze(0), ee_quat_des.unsqueeze(0))
        gripper_vals   = GRIPPER_OPEN_POS if gripper_cmd == "open" else GRIPPER_CLOSE_POS
        gripper_target = torch.tensor([gripper_vals], dtype=torch.float32, device=device)
        if cur_state in ("OPEN", "RETRACT"):
            cur_grip = robot.data.joint_pos[0, gripper_ids].cpu().numpy()
            #print(f"[DEBUG] state={cur_state} target={gripper_vals} current={cur_grip}")

        full_target = robot.data.joint_pos.clone()
        full_target[:, arm_ids]     = arm_jpos_des
        full_target[:, gripper_ids] = gripper_target
        env_action = (full_target - default_jpos) / ACTION_SCALE
        #robot.set_joint_position_target(gripper_target.unsqueeze(0) if gripper_target.dim() == 1 else gripper_target,joint_ids=gripper_ids)

        # Record BEFORE stepping (obs at time t, action at time t)
        qpos_buf.append(robot.data.joint_pos[0].cpu().numpy())
        qvel_buf.append(robot.data.joint_vel[0].cpu().numpy())
        ee_pos_buf.append(ik_ctrl.ee_pos_w[0].cpu().numpy())
        ee_quat_buf.append(ik_ctrl.ee_quat_w[0].cpu().numpy())
        action_buf.append(env_action[0].cpu().numpy())
        if frames_buf is not None:
            rgba = camera.data.output["rgb"][0].cpu().numpy()
            frames_buf.append(rgba[:, :, :3])
        
        gripper_tensor = torch.tensor([gripper_vals], dtype=torch.float32, device=device)
        robot.set_joint_position_target(gripper_tensor, joint_ids=gripper_ids)

        #if cur_state == "OPEN":
        #    print(f"[DEBUG] env_action shape={env_action.shape}")
        #    print(f"[DEBUG] env_action full={env_action[0].cpu().numpy()}")
        #    print(f"[DEBUG] env_action gripper part={env_action[0, gripper_ids].cpu().numpy()}")
        _, _, terminated, truncated, _ = env.step(env_action)
        #print(f"[DEBUG] state={cur_state} terminated={terminated.item()} truncated={truncated.item()}")


        #if terminated.any() or truncated.any():
        if truncated.any():
            print("[WARN] Episode ended early — skipping demo.")
            return None

    result = {
        "qpos":    np.stack(qpos_buf),
        "qvel":    np.stack(qvel_buf),
        "ee_pos":  np.stack(ee_pos_buf),
        "ee_quat": np.stack(ee_quat_buf),
        "action":  np.stack(action_buf),
    }
    if frames_buf is not None:
        result["frames"] = np.stack(frames_buf)
    return result


# ------------------------------------------------------------------ #
# Internal helper
# ------------------------------------------------------------------ #

def _step_to(env, robot, ik_ctrl, arm_ids, gripper_ids,
             ee_pos, ee_quat, gripper_target, default_jpos):
    """Single IK step toward a target pose (utility used during warm-up/settle)."""
    arm_des = ik_ctrl.compute(ee_pos, ee_quat)
    tgt = robot.data.joint_pos.clone()
    tgt[:, arm_ids]     = arm_des
    tgt[:, gripper_ids] = gripper_target
    env.step((tgt - default_jpos) / ACTION_SCALE)
    robot.update(dt=env.unwrapped.physics_dt)
