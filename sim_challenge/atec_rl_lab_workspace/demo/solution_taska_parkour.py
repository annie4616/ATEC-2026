"""Task A (off-road navigation) showcase of the B2Piper PARKOUR teacher.

The parkour teacher (policy_parkour.pt = B2Piper model_12700, run 2026-06-29_08-27-56)
was trained on the Task D terrain (pit + platform + box). Here we drive it on the
Task A off-road arena purely to record a video. There is no box and no push phase:
the parkour policy runs from step 0.

Scan input: the teacher consumes a 132-D height "scandot" grid (12 x-cols x 11 y-rows,
sensor at base_link + (0.375, 0), rotated by yaw). We do NOT synthesize the Task D
terrain here -- instead the play driver raycasts the REAL Task A ground mesh at those
132 grid points every step and injects the clipped scan via `self._live_scan`. If no
live scan is provided we fall back to a flat scan (all clipped to the flat value).

Obs layout (must match the checkpoint): num_prop 77 ; scandots 132 ; priv_explicit 9 ;
priv_latent 29 ; history 10*77  ->  1017.
"""

import os
import sys
import math
from typing import Any

import torch

# =============================================================================
# Joint order: policy_parkour.pt (B2Piper teacher) is fed joints in the CHALLENGE
# proprio order directly, with NO permutation -- this matches the known-good
# deployment references demo/solution.py (Task D active submission) and
# submission_taskd_teacher/solution_teacher_b2piper.py, both of which slot
# proprio[12:32] straight into p[13:33] and write the raw 20-D output straight to
# the env action. (An earlier attempt to permute into articulation/".*" order made
# the robot collapse -- the exported policy expects challenge order.)
# =============================================================================
PROP_FROM_TEACHER = list(range(20))       # identity
TEACHER_FROM_ACTION = list(range(20))     # identity

current_path = os.path.dirname(os.path.abspath(__file__))
if current_path not in sys.path:
    sys.path.insert(0, current_path)

# Scan grid (teacher height-scanner): 12 x-cols x 11 y-rows = 132, y-outer/x-inner,
# sensor at base_link + (0.375, 0), rotates with yaw. Same layout as solution.py.
SCAN_OFFSET_X = 0.375
SCAN_Z_OFFSET = 0.3                  # clip(base_z - terrain_h - 0.3, -1, 1)
_SCAN_XS = [round(-0.825 + 0.15 * i, 4) for i in range(12)]
_SCAN_YS = [round(-0.75 + 0.15 * j, 4) for j in range(11)]
_SCAN_LOCAL = [(sx + SCAN_OFFSET_X, sy) for sy in _SCAN_YS for sx in _SCAN_XS]   # 132 (x,y)


class AlgSolution:

    ACTION_SCALE = 0.5

    # ---- parkour teacher obs layout (B2Piper, 20 DoF, 1017-D) --------------------
    NUM_PROP = 3 + 2 + 3 + 4 + 3 * 20 + 5      # 77
    NUM_SCAN = 132
    NUM_PRIV_E = 9
    NUM_PRIV_L = 29
    NUM_HIST = 10
    NUM_JOINTS = 20
    PARKOUR_OBS_DIM = NUM_PROP + NUM_SCAN + NUM_PRIV_E \
        + NUM_PRIV_L + NUM_HIST * NUM_PROP      # 1017
    PARKOUR_TRAIN_SCALE = 0.25
    PARKOUR_OUT_RESCALE = PARKOUR_TRAIN_SCALE / ACTION_SCALE   # 0.5
    CROSS_VX = 1.0                              # forward vel command fed to the teacher

    STEP_DT = 0.02                             # decimation 4 * sim.dt 0.005

    # Flat-ground scan value: base rides ~0.55 m over flat ground; clip(0.55 - 0 - 0.3).
    FLAT_SCAN_VALUE = 0.25

    def __init__(self):
        self.device = 'cuda'

        self.leg_joint_indices = list(range(12))
        self.arm_joint_indices = list(range(12, 20))

        # PARKOUR teacher policy (1017-D obs -> 20-D joint action).
        self.parkour = torch.jit.load(current_path + '/policy_parkour.pt',
                                      map_location=self.device)
        self.parkour.eval()

        # scan grid local coords (132, 2) -- used by the driver's raycast too.
        self.scan_local = torch.tensor(_SCAN_LOCAL, device=self.device, dtype=torch.float32)

        # joint-order permutation index tensors.
        self.prop_from_teacher = torch.tensor(PROP_FROM_TEACHER, device=self.device, dtype=torch.long)
        self.teacher_from_action = torch.tensor(TEACHER_FROM_ACTION, device=self.device, dtype=torch.long)

        # Live scan injected by the play driver each step (real Task A terrain).
        self._live_scan = None

        self._init_episode_state()

    def _init_episode_state(self):
        self._ts = 0
        self.ryaw = 0.0
        self._parkour_hist = None
        self._teacher_last_action = torch.zeros(self.NUM_JOINTS, device=self.device)

    def get_action_spec(self) -> dict[str, dict[str, Any]] | None:
        return {}                               # default action config (scale 0.5)

    def reset(self, **kwargs) -> None:
        self._init_episode_state()
        return None

    # ---------------------------------------------------------------- odometry (yaw only)
    def _integrate_yaw(self, obs):
        proprio = obs["proprio"].to(self.device)[0]
        wz = float(proprio[5])
        self.ryaw += wz * self.STEP_DT

    def _scan(self) -> torch.Tensor:
        """132-D height scan. Prefer the driver-injected real-terrain scan; else flat."""
        if self._live_scan is not None:
            return self._live_scan.to(self.device, torch.float32).view(-1)[:self.NUM_SCAN]
        return torch.full((self.NUM_SCAN,), self.FLAT_SCAN_VALUE,
                          device=self.device, dtype=torch.float32)

    # ---------------------------------------------------------------- parkour obs
    def _build_proprio77(self, obs, joint_pos_rel, joint_vel) -> torch.Tensor:
        proprio = obs["proprio"].to(self.device)[0]
        base_ang_vel = proprio[3:6]
        gravity = proprio[9:12]
        roll = math.atan2(float(gravity[1]), float(gravity[2]) + 1e-6)
        pitch = math.atan2(-float(gravity[0]),
                           math.sqrt(float(gravity[1])**2 + float(gravity[2])**2) + 1e-6)
        delta_yaw = -self.ryaw
        # permute challenge-proprio joint order -> teacher (articulation) order.
        jpos_t = joint_pos_rel[:self.NUM_JOINTS][self.prop_from_teacher]
        jvel_t = joint_vel[:self.NUM_JOINTS][self.prop_from_teacher]
        # Exact training layout (parkour_isaaclab observations.py __call__):
        #   [0:3] ang_vel*0.25 | [3:5] imu(roll,pitch) | [5] 0 | [6] delta_yaw
        #   [7] delta_next_yaw | [8:10] 0*cmd(=0,0) | wait: see below
        # cat order: ang_vel(3), imu(2), 0*dyaw(1)->5, dyaw(1)->6, dnext_yaw(1)->7,
        #   0*cmd[0:2](2)->8,9, cmd[0:1](1)->10, env_idx(1)->11, invert_env_idx(1)->12,
        #   joint_pos_rel(20)->13:33, joint_vel*0.05(20)->33:53, action_hist(20)->53:73,
        #   contact_fill(4)->73:77.
        # NOTE indices: cmd vx sits at 10, env_idx at 11, invert at 12 (each a float).
        p = torch.zeros(self.NUM_PROP, device=self.device, dtype=torch.float32)
        p[0:3] = base_ang_vel * 0.25
        p[3] = roll
        p[4] = pitch
        p[5] = 0.0
        p[6] = delta_yaw
        p[7] = delta_yaw                        # delta_next_yaw ~ delta_yaw (no path here)
        p[8] = 0.0
        p[9] = 0.0
        p[10] = self.CROSS_VX                   # commands[0] = forward vel command
        p[11] = 1.0                             # env_idx: non-flat terrain (True)
        p[12] = 0.0                             # invert_env_idx: not flat (False)
        p[13:33] = jpos_t
        p[33:53] = jvel_t * 0.05
        p[53:73] = self._teacher_last_action    # action_history[-1], teacher order
        p[73:77] = 0.0                          # contact_fill unknown -> 0
        return p

    def _parkour_action(self, obs, action_dim) -> torch.Tensor:
        proprio = obs["proprio"].to(self.device)[0]
        base_lin_vel = proprio[0:3]
        joint_pos_rel = proprio[12:12 + action_dim]
        joint_vel = proprio[12 + action_dim:12 + 2 * action_dim]

        prop = self._build_proprio77(obs, joint_pos_rel, joint_vel)               # (77,)
        scan = self._scan()                                                       # (132,)

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
                          self._parkour_hist.reshape(-1)]).view(1, -1)            # (1, 1017)
        with torch.inference_mode():
            act_raw = self.parkour(full)
        if not isinstance(act_raw, torch.Tensor):
            act_raw = torch.as_tensor(act_raw, device=self.device)
        act_raw = act_raw.to(self.device, torch.float32)
        if act_raw.ndim == 1:
            act_raw = act_raw.unsqueeze(0)
        self._teacher_last_action = act_raw[0].clone()      # keep teacher order for history
        # rescale (spec scale 0.5 vs trained 0.25) then permute teacher -> env action order.
        act_env = (act_raw[0] * self.PARKOUR_OUT_RESCALE).clamp(-9.6, 9.6)
        act_env = act_env[self.teacher_from_action]
        return act_env.view(1, -1)

    # ---------------------------------------------------------------- main step
    def predicts(self, obs, current_score):
        proprio = obs["proprio"].to(self.device)
        action_dim = (int(proprio.shape[-1]) - 12) // 3     # 20
        self._integrate_yaw(obs)
        action = self._parkour_action(obs, action_dim)
        self._ts += 1
        if os.environ.get("ACT_DEBUG") and self._ts % 50 == 0:
            print(f"[TASKA-PARKOUR t={self._ts:4d}] yaw={self.ryaw:+.2f} "
                  f"scan={'live' if self._live_scan is not None else 'flat'}", flush=True)
        return {'action': action.cpu().numpy().tolist(), 'giveup': False}
