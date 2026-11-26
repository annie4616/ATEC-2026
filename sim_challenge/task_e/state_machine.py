"""Pick-place state machine and grasp-quaternion solver for Task E."""
#/home/yewon/ATEC2026_Simulation_Challenge/scripts/act/task_e/state_machine.py

import torch
from isaaclab.utils.math import matrix_from_quat, quat_from_matrix

from .config_all import (
    STEPS, STATE_ORDER, STATE_ORDER_MUSTARD,
    CARRY_Z, PLACE_HEIGHT,
    RETRACT_POS_X, RETRACT_POS_Y,
    GRASP_Z_OFFSET_PER_OBJ,
    MUSTARD_KNOCK_Y_OFFSET, MUSTARD_KNOCK_Z_OFFSET,
    MUSTARD_FALLEN_Z_OFFSET,
    MUSTARD_UPRIGHT_Z_OFFSET, MUSTARD_UPRIGHT_X_OFFSET, MUSTARD_UPRIGHT_Y_OFFSET,
    BASKET_CENTER_X, BASKET_CENTER_Y,
    PLACE_X_OFFSET, PLACE_Y_OFFSET,
    DEFAULT_PLACE_QUAT_W,
    get_step_count, get_pose_offset,
)

def _build_grasp_matrix(long_axis: torch.Tensor, grip_z: torch.Tensor) -> torch.Tensor:
    jaw_dir   = torch.linalg.cross(long_axis, grip_z)
    jaw_dir   = jaw_dir / jaw_dir.norm().clamp(min=1e-6)
    align_dir = torch.linalg.cross(jaw_dir, grip_z)
    align_dir = align_dir / align_dir.norm().clamp(min=1e-6)
    return torch.stack([align_dir, jaw_dir, grip_z], dim=1)


def compute_grasp_quat(obj_quat_w: torch.Tensor, device: str) -> torch.Tensor:
    R_obj        = matrix_from_quat(obj_quat_w.unsqueeze(0)).squeeze(0)
    grip_z       = torch.tensor([0.0, 0.0, -1.0], device=device)
    default_quat = torch.tensor(DEFAULT_PLACE_QUAT_W, dtype=torch.float32, device=device)

    norms, axes_xy = [], []
    for col in range(3):
        ax = torch.tensor([R_obj[0, col].item(), R_obj[1, col].item(), 0.0], device=device)
        norms.append(ax.norm().item())
        axes_xy.append(ax)

    best_norm  = max(norms)
    candidates = [
        axes_xy[c] / max(norms[c], 1e-6)
        for c in range(3)
        if norms[c] >= best_norm - 1e-3
    ]

    best_cos  = -2.0
    long_axis = candidates[0]
    for cand in candidates:
        q_cand  = quat_from_matrix(_build_grasp_matrix(cand, grip_z).unsqueeze(0)).squeeze(0)
        cos_sim = torch.abs((q_cand * default_quat).sum()).item()
        if cos_sim > best_cos:
            best_cos  = cos_sim
            long_axis = cand

    R_grip = _build_grasp_matrix(long_axis, grip_z)
    return quat_from_matrix(R_grip.unsqueeze(0)).squeeze(0)

# state_machine.py에 추가
import math

def is_upright(obj_quat_w: torch.Tensor, threshold: float = 0.7) -> bool:
    """병이 세워져 있는지 판단 (local Z가 world Z와 거의 정렬)"""
    R = matrix_from_quat(obj_quat_w.unsqueeze(0)).squeeze(0)
    # local Z axis의 world Z 성분
    z_alignment = abs(R[2, 1].item())
    return z_alignment > threshold


def compute_grasp_quat_upright(device: str) -> torch.Tensor:
    """세워진 병: 그냥 top-down으로 잡기"""
    #return torch.tensor([0.707, 0.707, 0.0, 0.0], dtype=torch.float32, device=device)
    #return torch.tensor([0.707, 0.0, 0.707, 0.0], dtype=torch.float32, device=device)
    #return torch.tensor([0.707, 0.0, 0.0, 0.707], dtype=torch.float32, device=device)
    #return torch.tensor([0.0, -0.707, -0.707, 0.0], dtype=torch.float32, device=device)
    return torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32, device=device)




def compute_grasp_quat_fallen(obj_quat_w: torch.Tensor, device: str) -> torch.Tensor:
    """누워진 병: long axis(=병의 원래 Z축)에 평행하게 그리퍼 정렬"""
    R_obj = matrix_from_quat(obj_quat_w.unsqueeze(0)).squeeze(0)
    
    # 누워진 병의 long axis는 local Y를 world frame에 옮긴 것
    long_axis_world = R_obj[:, 1].clone()
    long_axis_world[2] = 0.0  # XY 평면으로 사영
    long_axis_world = long_axis_world / long_axis_world.norm().clamp(min=1e-6)
    
    grip_z = torch.tensor([0.0, 0.0, -1.0], device=device)
    R_grip = _build_grasp_matrix(long_axis_world, grip_z)
    return quat_from_matrix(R_grip.unsqueeze(0)).squeeze(0)



class PickPlaceStateMachine:

    def __init__(self, object_indices: list[int], device: str):
        self._obj_indices = object_indices
        self._device      = device
        self._grasp_quat_cache: dict[int, torch.Tensor] = {}
        self.reset()

    def reset(self) -> None:
        self._ptr              = 0
        self._state_idx        = 0
        self._count            = 0
        self.done              = False
        self._cached_obj_pos: torch.Tensor | None = None
        self._grasp_quat_cache.clear()
        self._mustard_is_upright = True
        self._mustard_needs_knock = True
    
    def set_mustard_upright(self, upright: bool):
        self._mustard_is_upright = upright

    def set_mustard_needs_knock(self, needs_knock: bool):   # ← 추가
        self._mustard_needs_knock = needs_knock


    def set_grasp_quat(self, obj_idx: int, obj_quat_w: torch.Tensor) -> None:
        self._grasp_quat_cache[obj_idx] = compute_grasp_quat(obj_quat_w, self._device)

    def _current_state_order(self) -> list[str]:
        cur_idx = self._obj_indices[min(self._ptr, len(self._obj_indices) - 1)]
        if cur_idx == 2:
            # 머스타드: knock 필요 여부에 따라 분기
            return STATE_ORDER_MUSTARD if self._mustard_needs_knock else STATE_ORDER
        return STATE_ORDER

    def tick(self, obj_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, str]:
        cur_idx     = self._obj_indices[min(self._ptr, len(self._obj_indices) - 1)]
        state_order = self._current_state_order()
        s           = state_order[self._state_idx]
        d           = self._device

        # PRE_GRASP 진입 시 위치 캐시
        if s == "PRE_GRASP" and self._count == 0:
            self._cached_obj_pos = obj_pos.clone()

        # KNOCK 시 실시간 위치 사용 (캐시 무시)
        # WAIT 종료 → REACH 진입 시 쓰러진 위치로 캐시 갱신
        if s == "WAIT" and self._count == 0:
            self._cached_obj_pos = None  # 캐시 리셋

        if s == "REACH" and self._count == 0:
            self._cached_obj_pos = obj_pos.clone()  # 쓰러진 위치로 새로 캐시

        # REACH/CLOSE는 캐시된 위치 고정 사용
        if s in ("REACH", "CLOSE") and self._cached_obj_pos is not None:
            obj_pos = self._cached_obj_pos

        ee_pos, gripper = self._get_target_pos_gripper(s, obj_pos, d, cur_idx)
        ee_quat         = self._get_target_quat(s, d, cur_idx)

        self._count += 1
        if self._count >= get_step_count(cur_idx, s):
            self._count = 0
            if s == "RETRACT":
                self._ptr            += 1
                self._cached_obj_pos  = None
                if self._ptr >= len(self._obj_indices):
                    self.done = True
                    return ee_pos, ee_quat, gripper
                # 다음 물체는 항상 PRE_GRASP부터 시작
                next_order      = self._current_state_order()
                self._state_idx = next_order.index("PRE_GRASP")
            else:
                self._state_idx += 1

        return ee_pos, ee_quat, gripper

    @property
    def current_object_key(self) -> str:
        return f"object_{self._obj_indices[self._ptr]}"

    def _get_target_pos_gripper(
        self, s: str, obj_pos: torch.Tensor, d: str, cur_idx: int
    ) -> tuple[torch.Tensor, str]:
        if s == "INIT":
            return torch.tensor([RETRACT_POS_X, RETRACT_POS_Y, CARRY_Z], device=d), "open"
        elif s == "PRE_GRASP":
            p = obj_pos.clone()
            if cur_idx == 2:
                # 머스타드: 세워진 경우에만 X/Y 정렬 offset (누운 경우 보정 없음)
                if self._mustard_is_upright:
                    p[0] += MUSTARD_UPRIGHT_X_OFFSET
                    p[1] += MUSTARD_UPRIGHT_Y_OFFSET
            else:
                # 비-머스타드(sugar 등): config_all의 객체별 X/Y offset 적용
                ox, oy, _ = get_pose_offset(cur_idx, "PRE_GRASP")
                p[0] += ox
                p[1] += oy
            p[2] = CARRY_Z
            return p, "open"
        elif s == "KNOCK":
            p = obj_pos.clone()
            p[1] += MUSTARD_KNOCK_Y_OFFSET
            p[2] += MUSTARD_KNOCK_Z_OFFSET
            return p, "open"
        elif s == "WAIT":
            return torch.tensor([RETRACT_POS_X, RETRACT_POS_Y, CARRY_Z], device=d), "open"
        elif s == "REACH":
            p = obj_pos.clone()
            if cur_idx == 2:
                # 머스타드: 세워졌는지 누웠는지에 따라 다른 높이
                # is_upright를 미리 저장해두면 여기서 분기 가능
                if self._mustard_is_upright:
                    p[0] += MUSTARD_UPRIGHT_X_OFFSET
                    p[1] += MUSTARD_UPRIGHT_Y_OFFSET
                    p[2] += MUSTARD_UPRIGHT_Z_OFFSET   # 세워진 병: 위에서 잡기 (높게)
                else:
                    p[2] += MUSTARD_FALLEN_Z_OFFSET    # 누워진 병: 중앙 높이
            else:
                ox, oy, oz = get_pose_offset(cur_idx, "REACH")
                p[0] += ox
                p[1] += oy
                p[2] += oz if oz else GRASP_Z_OFFSET_PER_OBJ.get(cur_idx, 0.0)
            return p, "open"
        elif s == "CLOSE":
            p = obj_pos.clone()
            if cur_idx == 2:
                if self._mustard_is_upright:
                    p[0] += MUSTARD_UPRIGHT_X_OFFSET   # ← REACH와 동일
                    p[1] += MUSTARD_UPRIGHT_Y_OFFSET   # ← REACH와 동일
                    p[2] += MUSTARD_UPRIGHT_Z_OFFSET   # ← Z도 REACH와 같게 (FALLEN 아님!)
                else:
                    p[2] += MUSTARD_FALLEN_Z_OFFSET
            else:
                ox, oy, oz = get_pose_offset(cur_idx, "CLOSE")
                p[0] += ox
                p[1] += oy
                p[2] += oz if oz else GRASP_Z_OFFSET_PER_OBJ.get(cur_idx, 0.10)
            return p, "close"
        elif s == "LIFT":
            p = self._cached_obj_pos.clone() if self._cached_obj_pos is not None \
                else obj_pos.clone()
            p[2] = CARRY_Z
            return p, "close"
        elif s == "TRANSPORT":
            return torch.tensor([BASKET_CENTER_X + PLACE_X_OFFSET, BASKET_CENTER_Y + PLACE_Y_OFFSET, CARRY_Z], device=d), "close"
        elif s == "PLACE":
            return torch.tensor([BASKET_CENTER_X + PLACE_X_OFFSET, BASKET_CENTER_Y + PLACE_Y_OFFSET, PLACE_HEIGHT], device=d), "close"
        elif s == "OPEN":
            return torch.tensor([BASKET_CENTER_X + PLACE_X_OFFSET, BASKET_CENTER_Y + PLACE_Y_OFFSET, PLACE_HEIGHT + 0.3], device=d), "open"
        elif s == "LIFT_RETRACT":
            return torch.tensor([BASKET_CENTER_X + PLACE_X_OFFSET, BASKET_CENTER_Y + PLACE_Y_OFFSET, CARRY_Z], device=d), "open"
        elif s == "RETRACT":
            return torch.tensor([RETRACT_POS_X, RETRACT_POS_Y, CARRY_Z], device=d), "open"
        else:
            raise ValueError(f"Unknown state: {s}")

    def _get_target_quat(self, s: str, d: str, cur_idx: int) -> torch.Tensor:
        default_quat = torch.tensor(DEFAULT_PLACE_QUAT_W, dtype=torch.float32, device=d)
        # sugar(1): grasp yaw를 INIT부터 적용해 박스 위(높은 안전 위치)에서 회전을
        # 미리 끝내둔다. REACH 하강 중에 늦게 돌면서 박스를 치는 문제 방지.
        if cur_idx == 1:
            grasp_states = ("INIT", "PRE_GRASP", "REACH", "CLOSE", "LIFT")
        else:
            grasp_states = ("REACH", "CLOSE")
        if s in grasp_states:
            return self._grasp_quat_cache.get(cur_idx, default_quat)
        return default_quat