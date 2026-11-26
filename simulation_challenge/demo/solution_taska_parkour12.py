"""Task A (off-road) showcase of the SUCCESSFUL B2 12-DoF parkour teacher.

Policy: demo/policy_parkour_b2_12dof.pt  (== Isaaclab_Parkour run
2026-06-24_10-58-09_학습성공_0.45m / exported_teacher/policy.pt, md5 eaff25).
This is the teacher that CLIMBED the 0.45 m step. It is a PURE B2 (12-DoF, legs
only) teacher and outputs 12 leg actions. The exported forward() has the
normalizer baked in and calls actor(x, hist_encoding=True), so priv_latent is
ignored (zeros OK).

Obs = 753-dim (verified spec, memory atec2026-taskd-teacher-obs-spec):
  [0:53]   proprio 53 = ang_vel*0.25(3) imu(2) 0(1) delta_yaw(1) delta_next_yaw(1)
           0,0(2) vx_cmd(1) env_idx(1) invert_env_idx(1) joint_pos_rel(12)
           joint_vel*0.05(12) last_action(12) contact_fill(4)
  [53:185] scandot 132 = clip(base_z - terrain_h - 0.3, -1, 1), 12x11 grid,
           y-outer/x-inner, sensor at base+(0.375,0), rotated by yaw.
  [185:194] priv_explicit 9 = [base_lin_vel*2 (3), 0*6]
  [194:223] priv_latent 29 = zeros (ignored under hist_encoding)
  [223:753] history 530 = 10 x proprio53 (idx6:8 zeroed), init = 10x current.

Joint order: the teacher's 12 legs are in ARTICULATION order (DOF-type grouped:
FL_hip,FR_hip,RL_hip,RR_hip, FL_thigh..., FL_calf...). The Task A B2Piper proprio
and action leg order is PER-LEG (FR_hip,FR_thigh,FR_calf, FL_..., ...). So we
permute both ways (verified by name round-trip):
  teacher_leg[i] = proprio_leg[PROP_FROM_TEACHER[i]]
  env_action_leg[j] = teacher_out[TEACHER_FROM_ACTION[j]]
The 8 arm joints ride at default (action 0).

Scan: the driver (play_taska_parkour.py) raycasts the REAL Task A ground at the
132 grid points and injects self._live_scan.
"""

import os
import sys
import math
from typing import Any

import torch

current_path = os.path.dirname(os.path.abspath(__file__))
if current_path not in sys.path:
    sys.path.insert(0, current_path)

# --- scan grid (12x11=132, y-outer/x-inner), sensor at base_link + (0.375, 0) ---
SCAN_OFFSET_X = 0.375
SCAN_Z_OFFSET = 0.3
_SCAN_XS = [round(-0.825 + 0.15 * i, 4) for i in range(12)]
_SCAN_YS = [round(-0.75 + 0.15 * j, 4) for j in range(11)]
_SCAN_LOCAL = [(sx + SCAN_OFFSET_X, sy) for sy in _SCAN_YS for sx in _SCAN_XS]   # 132

# --- 12-DoF leg permutations (verified) ---
PROP_FROM_TEACHER = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
TEACHER_FROM_ACTION = [1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10]


class AlgSolution:

    ACTION_SCALE = 0.5                          # env spec leg scale

    NUM_LEG = 12
    NUM_ARM = 8
    NUM_JOINTS_ENV = 20                         # B2Piper env action dim

    # 12-DoF teacher obs layout
    NUM_PROP = 3 + 2 + 3 + 4 + 3 * 12 + 5       # 53
    NUM_SCAN = 132
    NUM_PRIV_E = 9
    NUM_PRIV_L = 29
    NUM_HIST = 10
    PARKOUR_OBS_DIM = NUM_PROP + NUM_SCAN + NUM_PRIV_E \
        + NUM_PRIV_L + NUM_HIST * NUM_PROP      # 753

    PARKOUR_TRAIN_SCALE = 0.25                  # teacher trained with action scale 0.25
    PARKOUR_OUT_RESCALE = PARKOUR_TRAIN_SCALE / ACTION_SCALE   # 0.5
    CROSS_VX = 1.0                              # forward vel command fed to the teacher
    STEP_DT = 0.02
    FLAT_SCAN_VALUE = 0.12                      # measured flat-ground scandot (spec memory)

    def __init__(self):
        self.device = 'cuda'
        self.parkour = torch.jit.load(current_path + '/policy_parkour_b2_12dof.pt',
                                      map_location=self.device)
        self.parkour.eval()

        self.scan_local = torch.tensor(_SCAN_LOCAL, device=self.device, dtype=torch.float32)
        self.prop_from_teacher = torch.tensor(PROP_FROM_TEACHER, device=self.device, dtype=torch.long)
        self.teacher_from_action = torch.tensor(TEACHER_FROM_ACTION, device=self.device, dtype=torch.long)

        self.leg_joint_indices = list(range(12))
        self.arm_joint_indices = list(range(12, 20))
        self.arm_default_action = torch.zeros((self.NUM_ARM,), device=self.device, dtype=torch.float32)

        self._live_scan = None
        self._init_episode_state()

    def _init_episode_state(self):
        self._ts = 0
        self.ryaw = 0.0
        self._parkour_hist = None
        self._teacher_last_action = torch.zeros(self.NUM_LEG, device=self.device)  # 12, teacher order

    def get_action_spec(self) -> dict[str, dict[str, Any]] | None:
        return {}

    def reset(self, **kwargs) -> None:
        self._init_episode_state()
        return None

    def _integrate_yaw(self, obs):
        wz = float(obs["proprio"].to(self.device)[0][5])
        self.ryaw += wz * self.STEP_DT

    def _scan(self) -> torch.Tensor:
        if self._live_scan is not None:
            return self._live_scan.to(self.device, torch.float32).view(-1)[:self.NUM_SCAN]
        return torch.full((self.NUM_SCAN,), self.FLAT_SCAN_VALUE,
                          device=self.device, dtype=torch.float32)

    def _build_proprio53(self, obs) -> torch.Tensor:
        proprio = obs["proprio"].to(self.device)[0]
        action_dim = (proprio.shape[-1] - 12) // 3            # 20 (env)
        base_ang_vel = proprio[3:6]
        gravity = proprio[9:12]
        roll = math.atan2(float(gravity[1]), float(gravity[2]) + 1e-6)
        pitch = math.atan2(-float(gravity[0]),
                           math.sqrt(float(gravity[1])**2 + float(gravity[2])**2) + 1e-6)
        # env proprio leg slice (per-leg order), first 12 of the 20 joints are legs.
        jpos_env = proprio[12:12 + action_dim][:self.NUM_LEG]
        jvel_env = proprio[12 + action_dim:12 + 2 * action_dim][:self.NUM_LEG]
        # permute per-leg -> teacher (articulation) order.
        jpos_t = jpos_env[self.prop_from_teacher]
        jvel_t = jvel_env[self.prop_from_teacher]

        delta_yaw = -self.ryaw
        p = torch.zeros(self.NUM_PROP, device=self.device, dtype=torch.float32)
        p[0:3] = base_ang_vel * 0.25
        p[3] = roll
        p[4] = pitch
        p[5] = 0.0
        p[6] = delta_yaw
        p[7] = delta_yaw                        # delta_next_yaw ~ delta_yaw
        p[8] = 0.0
        p[9] = 0.0
        p[10] = self.CROSS_VX                    # vx command
        p[11] = 1.0                              # env_idx: non-flat (step)
        p[12] = 0.0                              # invert_env_idx
        p[13:25] = jpos_t
        p[25:37] = jvel_t * 0.05
        p[37:49] = self._teacher_last_action     # 12, teacher order
        p[49:53] = 0.0                           # contact_fill unknown
        return p

    def _parkour_action(self, obs) -> torch.Tensor:
        proprio = obs["proprio"].to(self.device)[0]
        base_lin_vel = proprio[0:3]

        prop = self._build_proprio53(obs)                          # (53,)
        scan = self._scan()                                        # (132,)

        prop_hist = prop.clone()
        prop_hist[6:8] = 0.0
        if self._parkour_hist is None:
            self._parkour_hist = prop_hist.view(1, 1, -1).repeat(1, self.NUM_HIST, 1)
        else:
            self._parkour_hist = torch.cat(
                [self._parkour_hist[:, 1:], prop_hist.view(1, 1, -1)], dim=1)

        priv_e = torch.zeros(self.NUM_PRIV_E, device=self.device)
        priv_e[0:3] = base_lin_vel[:3] * 2.0
        priv_l = torch.zeros(self.NUM_PRIV_L, device=self.device)

        full = torch.cat([prop, scan, priv_e, priv_l,
                          self._parkour_hist.reshape(-1)]).view(1, -1)      # (1, 753)
        with torch.inference_mode():
            act_raw = self.parkour(full)                           # normalizer baked in
        if not isinstance(act_raw, torch.Tensor):
            act_raw = torch.as_tensor(act_raw, device=self.device)
        act_raw = act_raw.to(self.device, torch.float32)
        if act_raw.ndim == 1:
            act_raw = act_raw.unsqueeze(0)
        self._teacher_last_action = act_raw[0].clone()             # 12, teacher order (raw)

        # rescale, permute teacher -> env leg order, pad arm with default.
        leg_env = (act_raw[0] * self.PARKOUR_OUT_RESCALE).clamp(-9.6, 9.6)[self.teacher_from_action]
        action_env = torch.zeros(self.NUM_JOINTS_ENV, device=self.device, dtype=torch.float32)
        action_env[self.leg_joint_indices] = leg_env
        action_env[self.arm_joint_indices] = self.arm_default_action
        return action_env.view(1, -1)

    def predicts(self, obs, current_score):
        self._integrate_yaw(obs)
        action = self._parkour_action(obs)
        self._ts += 1
        if os.environ.get("ACT_DEBUG") and self._ts % 50 == 0:
            print(f"[TASKA-P12 t={self._ts:4d}] yaw={self.ryaw:+.2f} "
                  f"scan={'live' if self._live_scan is not None else 'flat'}", flush=True)
        return {'action': action.cpu().numpy().tolist(), 'giveup': False}
