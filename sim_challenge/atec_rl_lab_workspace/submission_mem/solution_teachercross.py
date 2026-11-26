"""Task D submission — TRACK-PUSH (+14) + B2Piper PARKOUR-TEACHER 0.4 m walk-across cross.

This is the COMBINED solution. It is `solution.py` (the verified +16 position-tracking
PUSH that settles the box into the -y-lane pit) with a NEW cross handoff that, instead of
replaying the brittle open-loop memorized cross actions (which diverge from the eval
server's air-drop start), drives all 20 joints with the VERIFIED B2Piper parkour TEACHER
(teacher_b2piper.pt, 1033 -> 20). The teacher walks the 0.4 m box-filled-pit step
CLOSED-LOOP (verified: it climbs a 0.4 m step 16/16), so the cross is robust.

PIPELINE (MEM_MODE="teachercross", the new default here):
  1. TRACK PUSH — identical to MODE=="track" in solution.py: _tracking_vel_cmd drives the
     walking RL policy (policy.pt, 12-D leg velocity tracker) to shove the box into the
     -y-lane pit for +14, while self.rx/ry/ryaw dead-reckoned odometry updates every step.
  2. TEACHER CROSS handoff — once current_score>=14 AND self.rx>=CROSS_TRIGGER_X (-0.80),
     set self._teacher_cross=True. From then on, every step we SYNTHESIZE the 1033-dim
     B2Piper teacher obs (proprio77 + scandot132 + priv_e9 + priv_l45 zeros + history770)
     from the dead-reckoned pose + the settled box terrain, run the teacher, and emit
     teacher_raw * 0.5 as the 20-dim env action. self._teacher_last_action and self.hist
     are tracked exactly as in solution_teacher_b2piper.py.

The teacher obs synthesis (terrain_height, scan grid, _build_proprio, _synth_scan, the
1033 assembly, _teacher_to_env_action) is REUSED AS-IS from
submission_taskd_teacher/solution_teacher_b2piper.py — the 1033 = 77+132+9+45+770 layout
and slicing match exactly.

SETTLED BOX (terrain_height during the teacher cross): the demo settles the box at world
x=-0.25, y=-0.49, z=-0.50 -> TOP z=-0.4. We fix box_cx=-0.25, box_cy=-0.49 once the
teacher cross starts, turning the 1.0 m pit into a 0.4 m step UP to the far flat ground.

MODES (env-var MEM_MODE): "teachercross" (default here), plus all of solution.py's modes
kept intact as fallbacks: "track", "oracle", "hybrid", "replay", "replayfb".

Files: vel_cmd_seq.npy (T,3), root_pos_seq.npy (T,3), joint_action_seq.npy (T,20),
policy.pt (12-D leg velocity tracker), teacher_b2piper.pt (1033->20 B2Piper parkour teacher).
"""

import math
import os
from typing import Any

import numpy as np
import torch

current_path = os.path.dirname(os.path.abspath(__file__))

LEG_PERM = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]          # identity (see header)
TRAIN_TO_ENV = [0.25, 0.5, 0.5, 0.25, 0.5, 0.5, 0.25, 0.5, 0.5, 0.25, 0.5, 0.5]
ENV_TO_TRAIN = [4.0, 2.0, 2.0, 4.0, 2.0, 2.0, 4.0, 2.0, 2.0, 4.0, 2.0, 2.0]

SPAWN_X, SPAWN_Y = -3.0, 0.0       # known deterministic spawn (robot faces +x, yaw 0)
STEP_DT = 0.02


# =============================================================================
# TEACHER TERRAIN + SCAN (reused AS-IS from solution_teacher_b2piper.py).
# Hard-coded terrain (world frame). tile->world: world_x = tile_x-6, world_y = tile_y-4;
# env_origin (-4.2,0,0). Robot spawns world (-3,0,~0.5) yaw 0. The box, once pushed into
# the -y-lane pit, sits half-submerged with its TOP at z=-0.4 (a 0.4 m step up to z=0.0).
# =============================================================================
PLATFORM_X0, PLATFORM_X1 = -0.65, 0.65
PLATFORM_Y0, PLATFORM_Y1 = 0.5, 3.5
PLATFORM_TOP_Z = 1.0                 # +y platform; not on the crossing path

PIT_X0, PIT_X1 = -0.65, 0.65
PIT_Y_MAX = 0.5                      # pit only on the -y side (y < +0.5)
PIT_FLOOR_Z = -1.0

BOX_HALF_X, BOX_HALF_Y = 0.4, 0.5
BOX_TOP_Z = -0.4                     # box half-submerged in pit -> top at z = -0.4
# Settled box location in the -y-lane pit (demo settle: x=-0.25, y=-0.49, top z=-0.4).
BOX_PARKED_CX = -0.25
BOX_PARKED_CY = -0.49
CROSS_LANE_Y = -0.45

SCAN_Z_OFFSET = 0.3                  # matches teacher: clip(base_z - terrain_h - 0.3,-1,1)


def terrain_height(x: float, y: float, box_cx: float, box_cy: float) -> float:
    """World-frame ground height at (x, y), including the box stepping-stone.

    The box, after the push, sits half-submerged in the -y-lane pit at (box_cx, box_cy)
    with its top at BOX_TOP_Z = -0.4, turning the 1.0 m pit into a 0.4 m step.
    """
    # Box (stepping stone in the pit), wherever it currently is.
    if (box_cx - BOX_HALF_X) <= x <= (box_cx + BOX_HALF_X) and \
       (box_cy - BOX_HALF_Y) <= y <= (box_cy + BOX_HALF_Y):
        return BOX_TOP_Z
    # Platform top (+y side; off the crossing path but modeled for completeness).
    if PLATFORM_X0 <= x <= PLATFORM_X1 and PLATFORM_Y0 <= y <= PLATFORM_Y1:
        return PLATFORM_TOP_Z
    # Pit (only on the -y side) — open pit where the box isn't.
    if PIT_X0 <= x <= PIT_X1 and y < PIT_Y_MAX:
        return PIT_FLOOR_Z
    # Flat ground.
    return 0.0


# Scan grid (teacher height-scanner), 132 points: GridPattern resolution 0.15,
# size [1.65,1.5] -> 12 x-cols x 11 y-rows, flattened y-outer/x-inner. Sensor at
# base_link + (0.375,0,*), rotates with yaw only. UNCHANGED from solution_teacher_b2piper.
SCAN_OFFSET_X = 0.375
_SCAN_XS = [round(-0.825 + 0.15 * i, 4) for i in range(12)]   # -0.825 .. +0.825
_SCAN_YS = [round(-0.75 + 0.15 * j, 4) for j in range(11)]    # -0.75  .. +0.75
_SCAN_LOCAL = [(sx + SCAN_OFFSET_X, sy) for sy in _SCAN_YS for sx in _SCAN_XS]  # 132 (x,y)


class AlgSolution:

    # MEM_MODE: "teachercross" (default here) = track push (+14) then B2Piper teacher cross;
    #           "track"  = position-tracking push only (robust to air-drop);
    #           "oracle" = open-loop vel_cmd replay (only works from demo start);
    #           "hybrid" = track + memorized open-loop joint_action over the cross window;
    #           "replay"/"replayfb" = pure / feedback-blended recorded joint_action replay.
    MODE = os.environ.get("MEM_MODE", "teachercross")
    # Position-feedback gain (world-error [m] -> body velocity command [m/s]).
    KP_POS = float(os.environ.get("MEM_KP", "1.5"))
    LOOKAHEAD = int(os.environ.get("MEM_LA", "1"))   # demo-path lookahead (steps)
    VXY_MAX = float(os.environ.get("MEM_VMAX", "3.0"))
    WZ_MAX = 1.5
    # Cross-window memorized replay (robot z<0.45 climbing box+platform), traj indices.
    CROSS_START = 950
    CROSS_END = 1200
    # POSITION-triggered cross handoff: once the box has scored (+14) AND the dead-reckoned
    # robot x reaches the demo's cross-start x, hand off the cross. In MODE=="hybrid" this
    # plays the memorized cross actions; in MODE=="teachercross" it switches to the teacher.
    CROSS_TRIGGER_X = float(os.environ.get("MEM_CROSS_X", "-0.80"))

    # ---- TEACHER CROSS forward command (proprio idx10 / vel cmd) ----
    TEACHER_VX = float(os.environ.get("MEM_TEACHER_VX", "1.8"))   # steady +x walk-across

    # ---- START ALIGNMENT (colleague's technique from demo/solution_b.py) ----
    SETTLE_LIN = 0.08            # base lin speed threshold [m/s]
    SETTLE_ANG = 0.10           # base ang speed threshold [rad/s]
    SETTLE_MIN = int(os.environ.get("MEM_SETTLE_MIN", "30"))   # let the air-drop LAND first
    SETTLE_MAX = int(os.environ.get("MEM_SETTLE_MAX", "120"))
    EASE_STEPS = int(os.environ.get("MEM_EASE", "60"))   # ease landed->demo-start
    MAX_LEG_DELTA = 0.04 / 0.5  # per-step leg action rate limit (rad/step / ACTION_SCALE)
    DO_ALIGN = os.environ.get("MEM_ALIGN", "1") == "1"
    DO_REALIGN = os.environ.get("MEM_REALIGN", "1") == "1"
    REALIGN_TGT_X = -3.0005
    REALIGN_TGT_Y = -0.0002
    REALIGN_TGT_YAW = 0.0
    REALIGN_POS_TOL = 0.03       # [m]
    REALIGN_YAW_TOL = 0.02       # [rad]
    REALIGN_MAX = int(os.environ.get("MEM_REALIGN_MAX", "300"))
    REALIGN_MIN = int(os.environ.get("MEM_REALIGN_MIN", "20"))
    WARMUP_N = int(os.environ.get("MEM_WARMUP_N", "10"))
    FB_KV = float(os.environ.get("MEM_FB_KV", "1.5"))    # pos err [m] -> corrective vel
    FB_GAIN = float(os.environ.get("MEM_FB_GAIN", "2.0"))  # err [m] -> blend weight
    FB_MAX = float(os.environ.get("MEM_FB_MAX", "0.7"))   # max corrective blend weight

    # ---- B2Piper teacher obs dims (total 1033). Layout MUST match the teacher file. ----
    NUM_PROP, NUM_SCAN, NUM_PRIV_E, NUM_PRIV_L, NUM_HIST = 77, 132, 9, 45, 10
    NUM_JOINTS = 20                     # 12 leg + 8 arm
    TEACHER_OBS_DIM = 77 + 132 + 9 + 45 + 10 * 77   # = 1033
    # teacher train JointPositionAction scale 0.25 vs env leg scale 0.5 (shared offset) ->
    # env action = teacher_raw * 0.5, applied to ALL 20 dims.
    TEACHER_TO_ENV_SCALE = 0.5

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.velcmd = np.load(os.path.join(current_path, "vel_cmd_seq.npy")).astype(np.float32)   # (T,3)
        self.rootpos = np.load(os.path.join(current_path, "root_pos_seq.npy")).astype(np.float32) # (T,3)
        self.seq = np.load(os.path.join(current_path, "joint_action_seq.npy")).astype(np.float32) # (T,20)
        try:
            # demo-start leg pose (joint_pos_rel, identity order); = (target-default)
            self.jprel_start = np.load(
                os.path.join(current_path, "proprio_jprel_start.npy")).astype(np.float32)
        except Exception:
            self.jprel_start = None
        self.T = self.velcmd.shape[0]

        self.policy = torch.jit.load(current_path + '/policy.pt', map_location=self.device)
        self.policy.eval()

        # B2Piper parkour teacher (1033 obs -> 20 action = leg12 + arm8).
        self.teacher = torch.jit.load(current_path + '/teacher_b2piper.pt',
                                      map_location=self.device)
        self.teacher.eval()

        self.leg_perm = torch.tensor(LEG_PERM, device=self.device, dtype=torch.long)
        self.train_to_env = torch.tensor(TRAIN_TO_ENV, device=self.device, dtype=torch.float32).view(1, -1)
        self.env_to_train = torch.tensor(ENV_TO_TRAIN, device=self.device, dtype=torch.float32).view(1, -1)

        # B2Piper default joint pose in proprio order (legs 12 + arm 8).
        leg_default = [-0.1, 0.8, -1.5,   0.1, 0.8, -1.5,
                       -0.1, 1.0, -1.5,   0.1, 1.0, -1.5]
        self.default_joint_pos = torch.tensor(
            [leg_default + [0.0] * 8], device=self.device, dtype=torch.float32)

        # precompute scan grid local coords as a tensor (132, 2)
        self.scan_local = torch.tensor(_SCAN_LOCAL, device=self.device, dtype=torch.float32)

        self._init_state()

    def _init_state(self):
        self._t = 0
        # dead-reckoned world pose (robot spawns at -3,0 facing +x)
        self.rx, self.ry, self.ryaw = SPAWN_X, SPAWN_Y, 0.0
        self.rz = 0.5                  # dead-reckoned base height (teacher scan only)
        self._cross_i = -1            # >=0 once the memorized cross replay is active
        # start-alignment phase machine: "settle" -> "ease" -> "run"
        self._phase = ("settle" if (self.DO_ALIGN and self.jprel_start is not None) else "run")
        self._phase_t = 0
        self._run_t = 0               # trajectory index during RUN
        self._ease_start = None       # captured landed leg pose (action units) at ease start
        self._prev_leg_cmd = None     # for rate-limiting
        # ----- TEACHER CROSS state -----
        self._teacher_cross = False   # True once we hand the cross off to the teacher
        # settled box location used by terrain_height during the teacher cross.
        self.box_cx = BOX_PARKED_CX
        self.box_cy = BOX_PARKED_CY
        self.hist = None              # (1, 10, 77) lazily filled at first teacher step
        self._teacher_last_action = torch.zeros(self.NUM_JOINTS, device=self.device)

    def get_action_spec(self) -> dict[str, dict[str, Any]] | None:
        return {}

    def reset(self, **kwargs) -> None:
        self._init_state()
        return None

    def _proprio(self, obs):
        pr = obs["proprio"]
        if hasattr(pr, "to"):
            return pr.to(self.device, torch.float32).view(1, -1)
        return torch.as_tensor(np.asarray(pr), device=self.device, dtype=torch.float32).view(1, -1)

    def _tracking_vel_cmd(self, meas_vx, meas_vy, meas_wz, t):
        """Advance odometry with measured body velocity, then return a body-frame velocity
        command = demo feed-forward + P correction toward the demo world position."""
        # integrate dead-reckoning
        self.ryaw += meas_wz * STEP_DT
        c, s = math.cos(self.ryaw), math.sin(self.ryaw)
        self.rx += (c * meas_vx - s * meas_vy) * STEP_DT
        self.ry += (s * meas_vx + c * meas_vy) * STEP_DT

        ti = t if t < self.T else self.T - 1
        la = min(ti + self.LOOKAHEAD, self.T - 1)
        ex = float(self.rootpos[la, 0]) - self.rx
        ey = float(self.rootpos[la, 1]) - self.ry
        # rotate world error into body frame
        bex = c * ex + s * ey
        bey = -s * ex + c * ey
        # feed-forward demo command + P correction
        vx = float(self.velcmd[ti, 0]) + self.KP_POS * bex
        vy = float(self.velcmd[ti, 1]) + self.KP_POS * bey
        wz = float(self.velcmd[ti, 2])
        # clamp
        sp = math.hypot(vx, vy)
        if sp > self.VXY_MAX:
            vx *= self.VXY_MAX / sp
            vy *= self.VXY_MAX / sp
        wz = max(-self.WZ_MAX, min(self.WZ_MAX, wz))
        return vx, vy, wz

    def _walk_leg(self, p, adim, vx=0.0, vy=0.0, wz=0.0):
        """Walking-policy leg action at a given velocity command (SETTLE uses zero)."""
        base_ang_vel = p[:, 3:6]
        projected_gravity = p[:, 9:12]
        joint_pos_leg = p[:, 12:24][:, self.leg_perm]
        joint_vel_leg = p[:, 12 + adim:12 + adim + 12][:, self.leg_perm]
        last_leg = p[:, 12 + 2 * adim:12 + 2 * adim + 12]
        actions_train_leg = last_leg * self.env_to_train
        vc = torch.tensor([[vx, vy, wz]], device=self.device, dtype=torch.float32)
        policy_obs = torch.cat([base_ang_vel * 0.25, projected_gravity, vc,
                                joint_pos_leg, joint_vel_leg * 0.05, actions_train_leg], dim=-1)
        with torch.inference_mode():
            at = self.policy(policy_obs)
        if not isinstance(at, torch.Tensor):
            at = torch.as_tensor(at, device=self.device, dtype=torch.float32)
        at = at.to(self.device, torch.float32)
        if at.ndim == 1:
            at = at.unsqueeze(0)
        return at * self.train_to_env            # (1,12) env-side leg action

    def _odom_step(self, p):
        """Advance dead-reckoned base pose from measured body velocity."""
        mvx, mvy = float(p[0, 0]), float(p[0, 1])
        mwz = float(p[0, 5])
        self.ryaw += mwz * STEP_DT
        c, s = math.cos(self.ryaw), math.sin(self.ryaw)
        self.rx += (c * mvx - s * mvy) * STEP_DT
        self.ry += (s * mvx + c * mvy) * STEP_DT

    def _emit_leg(self, leg_env, adim, rate_limit=False):
        if rate_limit and self._prev_leg_cmd is not None:
            leg_env = self._prev_leg_cmd + torch.clamp(
                leg_env - self._prev_leg_cmd, -self.MAX_LEG_DELTA, self.MAX_LEG_DELTA)
        self._prev_leg_cmd = leg_env.detach().clone()
        action = torch.zeros((1, adim), device=self.device, dtype=torch.float32)
        action[:, :12] = leg_env
        return {'action': [action.cpu().numpy()[0].tolist()], 'giveup': False}

    # ============================================================================
    # TEACHER OBS SYNTHESIS (reused AS-IS from solution_teacher_b2piper.py).
    # 1033 = proprio(77) + scan(132) + priv_e(9) + priv_l(45 zeros) + history(10*77).
    # ============================================================================
    def _synth_scan(self) -> torch.Tensor:
        """132-D height scan from hard-coded terrain + odometry pose (UNCHANGED grid)."""
        c, s = math.cos(self.ryaw), math.sin(self.ryaw)
        lx = self.scan_local[:, 0]
        ly = self.scan_local[:, 1]
        wx = self.rx + c * lx - s * ly
        wy = self.ry + s * lx + c * ly
        wx_l = wx.tolist()
        wy_l = wy.tolist()
        heights = torch.tensor(
            [terrain_height(x, y, self.box_cx, self.box_cy) for x, y in zip(wx_l, wy_l)],
            device=self.device, dtype=torch.float32)
        scan = torch.clip(self.rz - heights - SCAN_Z_OFFSET, -1.0, 1.0)
        return scan

    def _build_proprio(self, p, joint_pos_rel, joint_vel, vx_cmd) -> torch.Tensor:
        """Reconstruct the B2Piper teacher's 77-D proprio from env proprio + odometry.

        Layout (num_prop = 3+2+3+4+3*20+5 = 77):
          [0:3]   base_ang_vel_b * 0.25
          [3:5]   imu (roll, pitch)
          [5]     0
          [6]     delta_yaw       = -ryaw  (steer toward +x crossing, target yaw 0)
          [7]     delta_next_yaw  = delta_yaw
          [8:10]  0, 0
          [10]    vx_command
          [11]    env_idx         = 1.0   (non-flat / step terrain)
          [12]    invert_env_idx  = 0.0
          [13:33] joint_pos - default  (20 joints: 12 leg + 8 arm)
          [33:53] joint_vel * 0.05      (20 joints)
          [53:73] teacher's own previous raw 20-dim action
          [73:77] contact_fill (4 feet, unknown at deploy -> 0.0)
        """
        base_ang_vel = p[0, 3:6]                           # raw rad/s
        gravity = p[0, 9:12]                               # projected gravity
        roll = math.atan2(float(gravity[1]), float(gravity[2]) + 1e-6)
        pitch = math.atan2(-float(gravity[0]),
                           math.sqrt(float(gravity[1])**2 + float(gravity[2])**2) + 1e-6)
        delta_yaw = -self.ryaw                             # steer to +x crossing

        out = torch.zeros(self.NUM_PROP, device=self.device, dtype=torch.float32)
        out[0:3] = base_ang_vel * 0.25
        out[3] = roll
        out[4] = pitch
        out[5] = 0.0
        out[6] = delta_yaw
        out[7] = delta_yaw
        out[8] = 0.0
        out[9] = 0.0
        out[10] = vx_cmd
        out[11] = 1.0                                      # env_idx: non-flat (step) terrain
        out[12] = 0.0                                      # invert_env_idx
        # 20-dim joint blocks: legs first 12, arm last 8 ((joint_pos - default)).
        jp_rel = joint_pos_rel[:self.NUM_JOINTS]
        out[13:33] = jp_rel
        out[33:53] = joint_vel[:self.NUM_JOINTS] * 0.05
        out[53:73] = self._teacher_last_action             # teacher's own previous raw 20-dim
        out[73:77] = 0.0                                   # contact_fill (unknown -> 0)
        return out

    def _teacher_action(self, p, joint_pos_rel, joint_vel, base_lin_vel, vx_cmd):
        prop = self._build_proprio(p, joint_pos_rel, joint_vel, vx_cmd)         # (77,)
        scan = self._synth_scan()                                              # (132,)

        # history rolling buffer: proprio with idx6:8 zeroed (matches training).
        prop_hist = prop.clone()
        prop_hist[6:8] = 0.0
        if self.hist is None:
            self.hist = prop_hist.view(1, 1, -1).repeat(1, self.NUM_HIST, 1)
        else:
            self.hist = torch.cat([self.hist[:, 1:], prop_hist.view(1, 1, -1)], dim=1)

        priv_e = torch.zeros(self.NUM_PRIV_E, device=self.device)
        priv_e[0:3] = base_lin_vel[:3] * 2.0
        priv_l = torch.zeros(self.NUM_PRIV_L, device=self.device)              # 45 zeros

        full = torch.cat([prop, scan, priv_e, priv_l,
                          self.hist.reshape(-1)]).view(1, -1)                  # (1, 1033)
        with torch.inference_mode():
            act_raw = self.teacher(full)                 # (1, 20) teacher action space
        if not isinstance(act_raw, torch.Tensor):
            act_raw = torch.as_tensor(act_raw, device=self.device)
        act_raw = act_raw.to(self.device, torch.float32)
        if act_raw.ndim == 1:
            act_raw = act_raw.unsqueeze(0)
        self._teacher_last_action = act_raw[0].clone()   # feed back next step (obs term)
        return act_raw

    def _teacher_to_env_action(self, act_raw, adim):
        """B2Piper teacher 20-D raw output -> leg-first env action (legs + arm) * 0.5."""
        action = torch.zeros((1, adim), device=self.device, dtype=torch.float32)
        n = min(adim, 20)
        action[:, :n] = act_raw[:, :n] * self.TEACHER_TO_ENV_SCALE
        return action

    def _teacher_cross_step(self, p, adim):
        """One TEACHER CROSS step: synthesize the 1033-dim obs from the dead-reckoned pose
        + settled-box terrain, run the teacher, emit teacher_raw*0.5 as the 20-dim action.
        Odometry self.rx/ry/ryaw is advanced from the measured base velocity, and self.rz
        is set from terrain under the robot + ~0.45 stance (for the scan only)."""
        base_lin_vel = p[0, 0:3]
        joint_pos_rel = p[0, 12:12 + adim]
        joint_vel = p[0, 12 + adim:12 + 2 * adim]

        # forward command for the teacher (proprio idx10): steady +x walk-across.
        vx_cmd = self.TEACHER_VX
        act_raw = self._teacher_action(p, joint_pos_rel, joint_vel, base_lin_vel, vx_cmd)
        action = self._teacher_to_env_action(act_raw, adim)

        # advance odometry from the MEASURED base velocity, then update base height (scan).
        self._odom_step(p)
        self.rz = max(0.16, min(0.6,
                                terrain_height(self.rx, self.ry,
                                               self.box_cx, self.box_cy) + 0.45))
        self._t += 1
        return {'action': [action.cpu().numpy()[0].tolist()], 'giveup': False}

    def predicts(self, obs, current_score):
        p = self._proprio(obs)
        P = p.shape[-1]
        adim = (P - 12) // 3

        # ===== TEACHER CROSS: once handed off, the teacher drives every step. =====
        if self._teacher_cross:
            return self._teacher_cross_step(p, adim)

        # ===== START ALIGNMENT: SETTLE -> (REALIGN) -> WARMUP -> RUN =====
        HOLD = os.environ.get("MEM_SETTLE_HOLD", "1") == "1"
        if self._phase == "settle":
            self._phase_t += 1
            self._odom_step(p)
            spd = math.hypot(float(p[0, 0]), float(p[0, 1]))
            wz = abs(float(p[0, 5]))
            if HOLD:
                leg = torch.tensor(self.jprel_start[:12] / 0.5, device=self.device,
                                   dtype=torch.float32).view(1, -1)
            else:
                leg = self._walk_leg(p, adim)        # zero-vel stand
            settled = (self._phase_t >= self.SETTLE_MIN
                       and spd < self.SETTLE_LIN and wz < self.SETTLE_ANG)
            if settled or self._phase_t >= self.SETTLE_MAX:
                self._phase = ("realign" if self.DO_REALIGN else "warmup"); self._phase_t = 0
                if os.environ.get("MEM_VERBOSE"):
                    print(f"[solution] SETTLE done at {self._t} (spd={spd:.3f} wz={wz:.3f}) "
                          f"odom=({self.rx:+.3f},{self.ry:+.3f},{self.ryaw:+.3f})", flush=True)
            self._t += 1
            return self._emit_leg(leg, adim, rate_limit=False)

        if self._phase == "realign":
            self._phase_t += 1
            self._odom_step(p)
            ex = self.REALIGN_TGT_X - self.rx
            ey = self.REALIGN_TGT_Y - self.ry
            eyaw = self.REALIGN_TGT_YAW - self.ryaw
            # normalize yaw error to [-pi,pi]
            eyaw = (eyaw + math.pi) % (2 * math.pi) - math.pi
            dist = math.hypot(ex, ey)
            aligned = (self._phase_t >= self.REALIGN_MIN
                       and dist < self.REALIGN_POS_TOL and abs(eyaw) < self.REALIGN_YAW_TOL)
            if aligned or self._phase_t >= self.REALIGN_MAX:
                self._phase = "warmup"; self._phase_t = 0
                if os.environ.get("MEM_VERBOSE"):
                    print(f"[solution] REALIGN done at {self._t} odom=({self.rx:+.3f},{self.ry:+.3f},"
                          f"{self.ryaw:+.3f}) err(d={dist:.3f},yaw={eyaw:+.3f})", flush=True)
                return self._emit_leg(self._walk_leg(p, adim), adim, rate_limit=False)
            # body-frame velocity toward target + yaw correction
            c, s = math.cos(self.ryaw), math.sin(self.ryaw)
            bex = c * ex + s * ey
            bey = -s * ex + c * ey
            vx = max(-0.6, min(0.6, 1.2 * bex))
            vy = max(-0.6, min(0.6, 1.2 * bey))
            wzc = max(-0.6, min(0.6, 1.5 * eyaw))
            return self._emit_leg(self._walk_leg(p, adim, vx, vy, wzc), adim, rate_limit=False)

        if self._phase == "warmup":
            self._phase_t += 1
            leg = self._walk_leg(p, adim, 0.6, 0.0, 0.0)
            if self._phase_t >= self.WARMUP_N:
                self._phase = "run"; self._phase_t = 0
                self._run_t = 0
                if os.environ.get("MEM_VERBOSE"):
                    print(f"[solution] WARMUP({self.WARMUP_N}) done -> replay begins", flush=True)
            return self._emit_leg(leg, adim, rate_limit=False)

        # ===== RUN: the memorized maneuver (t = trajectory index from replay start) =====
        t = self._run_t
        self._run_t += 1
        self._t += 1

        # PURE REPLAY
        if self.MODE == "replay":
            a = self.seq[t if t < self.seq.shape[0] else self.seq.shape[0] - 1]
            return {'action': [a.tolist()], 'giveup': False}

        # REPLAY + POSITION FEEDBACK
        if self.MODE == "replayfb":
            self._odom_step(p)
            ti = t if t < self.T else self.T - 1
            a_ff = self.seq[ti]                                   # (20,) recorded feed-forward
            ex = float(self.rootpos[ti, 0]) - self.rx
            ey = float(self.rootpos[ti, 1]) - self.ry
            err = math.hypot(ex, ey)
            c, s = math.cos(self.ryaw), math.sin(self.ryaw)
            bex = c * ex + s * ey
            bey = -s * ex + c * ey
            kv = self.FB_KV
            vx = max(-1.0, min(1.0, kv * bex))
            vy = max(-1.0, min(1.0, kv * bey))
            corr_leg = self._walk_leg(p, adim, vx, vy, 0.0)[0].detach().cpu().numpy()  # (12,)
            w_corr = min(self.FB_MAX, self.FB_GAIN * err)
            out = a_ff.copy()
            out[:12] = (1.0 - w_corr) * a_ff[:12] + w_corr * corr_leg
            if os.environ.get("MEM_VERBOSE") and t % 100 == 0:
                print(f"[replayfb] t={t} err={err:.3f} w={w_corr:.2f} odom=({self.rx:+.2f},{self.ry:+.2f})", flush=True)
            return {'action': [out.tolist()], 'giveup': False}

        base_lin_vel = p[:, 0:3]
        base_ang_vel = p[:, 3:6]
        projected_gravity = p[:, 9:12]
        jp = p[:, 12:12 + adim]
        jv = p[:, 12 + adim:12 + 2 * adim]
        last_action = p[:, 12 + 2 * adim:12 + 3 * adim]

        joint_pos_leg = jp[:, self.leg_perm]
        joint_vel_leg = jv[:, self.leg_perm]
        actions_train_leg = last_action[:, :12] * self.env_to_train

        # --- already replaying the memorized cross (hybrid mode)? advance it ---
        if self._cross_i >= 0:
            if self._cross_i < min(self.CROSS_END, self.seq.shape[0]):
                a = self.seq[self._cross_i]; self._cross_i += 1
                return {'action': [a.tolist()], 'giveup': False}
            # cross done -> fall through to tracking for the run-out to the finish line

        # velocity command
        if self.MODE in ("track", "hybrid", "teachercross"):
            mvx = float(base_lin_vel[0, 0]); mvy = float(base_lin_vel[0, 1])
            mwz = float(base_ang_vel[0, 2])
            vx, vy, wz = self._tracking_vel_cmd(mvx, mvy, mwz, t)   # also updates odometry
            vc = torch.tensor([vx, vy, wz], device=self.device, dtype=torch.float32).view(1, 3)
            # POSITION-triggered CROSS handoff: box scored (+14) and robot reached the
            # demo's cross-start x.
            if (self._cross_i < 0 and current_score >= 14.0
                    and self.rx >= self.CROSS_TRIGGER_X):
                if self.MODE == "teachercross":
                    # hand the cross off to the B2Piper parkour teacher (closed-loop).
                    self._teacher_cross = True
                    # fix the settled-box location for the teacher's scandot synthesis.
                    self.box_cx = BOX_PARKED_CX
                    self.box_cy = BOX_PARKED_CY
                    if os.environ.get("MEM_VERBOSE"):
                        print(f"[solution] TEACHER CROSS handoff fired at t={t} "
                              f"odom_rx={self.rx:+.2f} score={current_score} "
                              f"box=({self.box_cx:+.2f},{self.box_cy:+.2f})", flush=True)
                    return self._teacher_cross_step(p, adim)
                elif self.MODE == "hybrid":
                    self._cross_i = self.CROSS_START
                    if os.environ.get("MEM_VERBOSE"):
                        print(f"[solution] CROSS handoff fired at t={t} odom_rx={self.rx:+.2f} "
                              f"score={current_score}", flush=True)
        else:  # "oracle": open-loop recorded command
            tt = t if t < self.T else self.T - 1
            vc = torch.tensor(self.velcmd[tt], device=self.device, dtype=torch.float32).view(1, 3)

        policy_obs = torch.cat([
            base_ang_vel * 0.25,
            projected_gravity,
            vc,
            joint_pos_leg,
            joint_vel_leg * 0.05,
            actions_train_leg,
        ], dim=-1)

        with torch.inference_mode():
            at = self.policy(policy_obs)
        if not isinstance(at, torch.Tensor):
            at = torch.as_tensor(at, device=self.device, dtype=torch.float32)
        at = at.to(self.device, torch.float32)
        if at.ndim == 1:
            at = at.unsqueeze(0)

        leg_env = at * self.train_to_env
        action = torch.zeros((1, adim), device=self.device, dtype=torch.float32)
        action[:, :12] = leg_env
        return {'action': [action.cpu().numpy()[0].tolist()], 'giveup': False}
