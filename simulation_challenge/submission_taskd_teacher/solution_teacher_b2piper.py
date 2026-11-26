"""Task D submission — HARD-CODED box-push + B2Piper PARKOUR-TEACHER 0.4 m walk-across.

EXPERIMENTAL teacher version (adapted from solution_teacher.py.bak, the B2 753-dim
variant). Key differences from the .bak:

  1. TEACHER EMBODIMENT: B2Piper (20 DoF = 12 leg + 8 arm) instead of B2 (12 DoF).
     The teacher obs is 1033-dim (not 753) and the teacher outputs 20 dims (not 12):
       prop(77) + scan(132) + priv_explicit(9) + priv_latent(45 zeros) + history(10*77).
     The teacher's raw 20-dim output maps straight onto the 20-dim leg-first env action
     (legs 0:12, arm 12:20) with a single *0.5 scale (teacher train scale 0.25 vs env
     leg scale 0.5, shared default offset).

  2. STRATEGY: 0.4 m WALK-ACROSS on the -y lane (NOT the +y platform climb the .bak did).
     The box is pushed into the -y lane PIT (the pit exists only for y < +0.5). It sinks
     and settles half-submerged with its TOP at world z = -0.4 (center z = -0.7), filling
     the 1.0 m-deep pit so only a 0.4 m step UP remains from box-top (-0.4) to the far
     flat ground (0.0). The robot crouches low (base z ~0.16-0.37) crossing the box, then
     rises to z ~0.55 on the far ground. It NEVER touches the +y platform.

PHASES (proprio-only deployable):

  PHASE 1 (PUSH): open-loop velocity script driving the walking RL policy (policy.pt,
      12-DoF leg velocity tracker, reused from the working 16-pt submission). Retargeted
      to shove the box into the -y lane pit (settles near world x in [-0.4,+0.4],
      y = -0.45). Approximate for this first version — the priority is correct teacher
      obs synthesis and the walk-across terrain model.

  PHASE 2 (CLIMB / walk-across): the exported B2Piper parkour TEACHER (teacher_b2piper.pt,
      1033 -> 20) drives all 20 joints directly while the robot walks the -y lane over the
      box-as-stepping-stone. The 1033-D obs is SYNTHESIZED from the env proprio + cameras-
      free dead-reckoning odometry (integrate measured base velocity from the known spawn)
      + hard-coded terrain geometry for the height-scan.

CLIMB_MODE:
  "climb"   : (default for this experimental file) attempt the teacher walk-across.
  "recover" : fall back to the proven open-loop +16 recovery from the .bak.
"""

import math
import os

import torch

current_path = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# Hard-coded terrain (world frame). Same tile->world mapping as the .bak:
#   world_x = tile_x - 6.0,  world_y = tile_y - 4.0;  env_origin (-4.2,0,0).
#   size (12,8), border 1.0, pit_width 1.3, platform_height 1.0, difficulty 0.
#   - Robot spawns world (-3.0, 0.0, ~0.5), yaw 0 (facing +x).
#   - Box spawns world (-3.0, +1.6, 0.5): size (0.8 x, 1.0 y, 0.6 z).
#   - Platform (+y side): world x in [-0.65,+0.65], y in [+0.5,+3.5], top z = 1.0.
#     IRRELEVANT to the -y-lane crossing path (the robot stays at y ~ -0.45).
#   - Pit: world x in [-0.65,+0.65] on the -y side (y < +0.5), floor z = -1.0.
#
# STRATEGY CHANGE vs the .bak: the box is pushed into the -y-lane PIT, where it sinks and
# settles half-submerged. Its TOP rests at z = -0.4 (center z = -0.7), filling the pit so
# the crossing is a 0.4 m step UP from box-top (-0.4) to the far flat ground (0.0).
# =============================================================================
PLATFORM_X0, PLATFORM_X1 = -0.65, 0.65
PLATFORM_Y0, PLATFORM_Y1 = 0.5, 3.5
PLATFORM_TOP_Z = 1.0                 # +y platform; not on the crossing path

PIT_X0, PIT_X1 = -0.65, 0.65
PIT_Y_MAX = 0.5                      # pit only on the -y side (y < +0.5)
PIT_FLOOR_Z = -1.0

BOX_HALF_X, BOX_HALF_Y = 0.4, 0.5
# AFTER the push, the box is in the -y-lane pit, half-submerged, TOP at z = -0.4.
BOX_TOP_Z = -0.4
# Box resting place in the -y lane pit (center, world frame). The push retargets here.
BOX_PARKED_CX = 0.0                  # box center x settles near pit center
BOX_PARKED_CY = -0.45                # -y crossing-lane center
# The walk-across lane the robot follows.
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


# =============================================================================
# Scan grid (teacher height-scanner) — UNCHANGED from the .bak / B2 (132 points):
# GridPattern resolution 0.15, size [1.65, 1.5] -> 12 x-cols x 11 y-rows = 132,
# flattened y-outer / x-inner. Sensor at base_link + (0.375, 0, *), rotates with yaw only.
# =============================================================================
SCAN_OFFSET_X = 0.375
_SCAN_XS = [round(-0.825 + 0.15 * i, 4) for i in range(12)]   # -0.825 .. +0.825
_SCAN_YS = [round(-0.75 + 0.15 * j, 4) for j in range(11)]    # -0.75  .. +0.75
_SCAN_LOCAL = [(sx + SCAN_OFFSET_X, sy) for sy in _SCAN_YS for sx in _SCAN_XS]  # 132 (x,y)


class AlgSolution:

    # ---- walking-policy action scaling (same as the working 16-pt submission) ----
    TRAIN_TO_ENV = [0.25, 0.5, 0.5, 0.25, 0.5, 0.5, 0.25, 0.5, 0.5, 0.25, 0.5, 0.5]
    ENV_TO_TRAIN = [4.0, 2.0, 2.0, 4.0, 2.0, 2.0, 4.0, 2.0, 2.0, 4.0, 2.0, 2.0]

    STEP_DT = 0.02                      # decimation 4 * sim.dt 0.005

    # ---- PHASE 1 push velocity script (robot faces +x: vx=world+x, vy=world+y) ----
    SPAWN_X, SPAWN_Y = -3.0, 0.0
    BOX_SPAWN_CY = 1.6                  # box spawns at world y = +1.6
    V_SIDE = 1.0                        # +/-y strafe to reach the box y-line
    V_BACK = -1.2                       # -x back-up to get behind the box first
    V_PUSH = 1.4                        # push speed
    V_APPROACH = 1.6                    # close-up before the walk-across
    V_RETREAT = -1.5                    # -x walk-back in BACKOFF to open run-up room
    BACKOFF_DIST = 1.6

    # ---- mode select ----
    # "climb"   : (default here) after +14, retreat then let the B2Piper teacher attempt
    #             the 0.4 m -y-lane walk-across over the box.
    # "recover" : after +14, run the proven open-loop recovery (strafe -y into the
    #             crossing lane, then walk +x) for the robot-reach +2 -> total 16.
    CLIMB_MODE = "climb"
    # post-+14 recovery (from the working 16-pt submission)
    V_STRAFE = 1.5
    STRAFE_STEPS = 50
    V_FORWARD_POST = 2.0
    PUSH_TIMEOUT_STEPS = 1200

    # ---- B2Piper teacher obs dims (total 1033) ----
    NUM_PROP, NUM_SCAN, NUM_PRIV_E, NUM_PRIV_L, NUM_HIST = 77, 132, 9, 45, 10
    NUM_JOINTS = 20                     # 12 leg + 8 arm
    TEACHER_OBS_DIM = 77 + 132 + 9 + 45 + 10 * 77   # = 1033

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # walking RL policy (velocity command -> 12-D leg action) — reused for the push.
        self.policy = torch.jit.load(current_path + '/policy.pt', map_location=self.device)
        self.policy.eval()
        # B2Piper parkour teacher (1033 obs -> 20 action = leg12 + arm8).
        self.teacher = torch.jit.load(current_path + '/teacher_b2piper.pt',
                                      map_location=self.device)
        self.teacher.eval()

        self.leg_idx = list(range(12))
        self.arm_idx = list(range(12, 20))
        self.train_to_env = torch.tensor(self.TRAIN_TO_ENV, device=self.device).view(1, -1)
        self.env_to_train = torch.tensor(self.ENV_TO_TRAIN, device=self.device).view(1, -1)

        # B2Piper default joint pose in proprio order (legs 12 + arm 8).
        leg_default = [-0.1, 0.8, -1.5,   0.1, 0.8, -1.5,
                       -0.1, 1.0, -1.5,   0.1, 1.0, -1.5]
        self.default_joint_pos = torch.tensor(
            [leg_default + [0.0] * 8], device=self.device, dtype=torch.float32)

        # precompute scan grid local coords as a tensor (132, 2)
        self.scan_local = torch.tensor(_SCAN_LOCAL, device=self.device, dtype=torch.float32)

        self._init_episode_state()

    def _init_episode_state(self):
        # odometry (dead-reckoning of the robot base pose in world)
        self.rx, self.ry, self.ryaw = self.SPAWN_X, self.SPAWN_Y, 0.0
        self.rz = 0.5
        # box starts at its spawn (-3.0, +1.6); push moves it toward (0.0, -0.45).
        self.box_cx = -3.0
        self.box_cy = self.BOX_SPAWN_CY
        self.phase = "PUSH"
        self.step_i = 0
        self.hist = None                   # (1, 10, 77) lazily filled at first teacher step
        self._last_cmd = (0.0, 0.0, 0.0)
        # B2Piper teacher action is 20-dim now.
        self._teacher_last_action = torch.zeros(self.NUM_JOINTS, device=self.device)
        self._backoff_target_x = -3.0
        self._post_step = 0

    def get_action_spec(self):
        return {}

    def reset(self, **kwargs):
        self._init_episode_state()
        return None

    # ------------------------------------------------------------------ odometry
    def _integrate_odometry(self, meas_vx, meas_vy, meas_wz):
        """Advance the dead-reckoned base pose using the MEASURED base velocity
        (obs proprio[0:3] = base_lin_vel in body frame)."""
        self.ryaw += meas_wz * self.STEP_DT
        c, s = math.cos(self.ryaw), math.sin(self.ryaw)
        self.rx += (c * meas_vx - s * meas_vy) * self.STEP_DT
        self.ry += (s * meas_vx + c * meas_vy) * self.STEP_DT
        # Box rides ~0.45 m ahead of the robot's base while pushing, drifting toward the
        # -y-lane pit target (box_cx -> 0.0, box_cy -> -0.45).
        if self.phase == "PUSH":
            speed = math.hypot(meas_vx, meas_vy)
            if speed > 0.1:
                # box follows the robot's heading, capped at the pit target.
                tgt_cx = self.rx + c * 0.45
                tgt_cy = self.ry + s * 0.45
                self.box_cx = max(min(tgt_cx, BOX_PARKED_CX), self.box_cx) \
                    if self.rx < BOX_PARKED_CX else BOX_PARKED_CX
                # ease box_cy toward the lane target as it nears the pit.
                if self.rx > -0.8:
                    self.box_cy = BOX_PARKED_CY
                else:
                    self.box_cy = 0.7 * self.box_cy + 0.3 * tgt_cy

    # ------------------------------------------------------------ scan synthesis
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

    # ------------------------------------------------------- teacher proprio(77)
    def _build_proprio(self, obs, joint_pos_rel, joint_vel) -> torch.Tensor:
        """Reconstruct the B2Piper teacher's 77-D proprio from env proprio + odometry.

        Layout (num_prop = 3+2+3+4+3*20+5 = 77):
          [0:3]   base_ang_vel_b * 0.25
          [3:5]   imu (roll, pitch)
          [5]     0
          [6]     delta_yaw       = -ryaw  (steer toward +x crossing, target yaw 0)
          [7]     delta_next_yaw  = delta_yaw
          [8:10]  0, 0
          [10]    vx_command      = self._last_cmd[0]
          [11]    env_idx         = 1.0   (non-flat / step terrain)
          [12]    invert_env_idx  = 0.0
          [13:33] joint_pos - default  (20 joints: 12 leg + 8 arm)
          [33:53] joint_vel * 0.05      (20 joints)
          [53:73] teacher's own previous raw 20-dim action
          [73:77] contact_fill (4 feet, unknown at deploy -> 0.0)
        """
        proprio = obs["proprio"].to(self.device)[0]
        base_ang_vel = proprio[3:6]                       # raw rad/s
        gravity = proprio[9:12]                           # projected gravity
        roll = math.atan2(float(gravity[1]), float(gravity[2]) + 1e-6)
        pitch = math.atan2(-float(gravity[0]),
                           math.sqrt(float(gravity[1])**2 + float(gravity[2])**2) + 1e-6)
        delta_yaw = -self.ryaw                            # steer to +x crossing
        vx_cmd = self._last_cmd[0]

        p = torch.zeros(self.NUM_PROP, device=self.device, dtype=torch.float32)
        p[0:3] = base_ang_vel * 0.25
        p[3] = roll
        p[4] = pitch
        p[5] = 0.0
        p[6] = delta_yaw
        p[7] = delta_yaw
        p[8] = 0.0
        p[9] = 0.0
        p[10] = vx_cmd
        p[11] = 1.0                                       # env_idx: non-flat (step) terrain
        p[12] = 0.0                                       # invert_env_idx
        # 20-dim joint blocks: legs first 12, arm last 8.
        p[13:33] = joint_pos_rel[:self.NUM_JOINTS]
        p[33:53] = joint_vel[:self.NUM_JOINTS] * 0.05
        p[53:73] = self._teacher_last_action             # teacher's own previous raw 20-dim
        p[73:77] = 0.0                                    # contact_fill (unknown -> 0)
        return p

    def _teacher_action(self, obs, joint_pos_rel, joint_vel, base_lin_vel):
        prop = self._build_proprio(obs, joint_pos_rel, joint_vel)               # (77,)
        scan = self._synth_scan()                                               # (132,)

        # history rolling buffer: proprio with idx6:8 zeroed (matches training).
        prop_hist = prop.clone()
        prop_hist[6:8] = 0.0
        if self.hist is None:
            self.hist = prop_hist.view(1, 1, -1).repeat(1, self.NUM_HIST, 1)
        else:
            self.hist = torch.cat([self.hist[:, 1:], prop_hist.view(1, 1, -1)], dim=1)

        priv_e = torch.zeros(self.NUM_PRIV_E, device=self.device)
        priv_e[0:3] = base_lin_vel[:3] * 2.0
        priv_l = torch.zeros(self.NUM_PRIV_L, device=self.device)               # 45 zeros

        full = torch.cat([prop, scan, priv_e, priv_l,
                          self.hist.reshape(-1)]).view(1, -1)                   # (1, 1033)
        with torch.inference_mode():
            act_raw = self.teacher(full)                 # (1, 20) teacher action space
        if not isinstance(act_raw, torch.Tensor):
            act_raw = torch.as_tensor(act_raw, device=self.device)
        act_raw = act_raw.to(self.device, torch.float32)
        if act_raw.ndim == 1:
            act_raw = act_raw.unsqueeze(0)
        self._teacher_last_action = act_raw[0].clone()   # feed back next step (obs term)
        return act_raw

    # ------------------------------------------------------- walking-policy obs
    def _walking_obs(self, obs, joint_pos_rel, joint_vel, last_action, vel_cmd):
        base_ang_vel = obs["proprio"].to(self.device)[:, 3:6]
        gravity = obs["proprio"].to(self.device)[:, 9:12]
        actions_train_leg = last_action[:12].view(1, -1) * self.env_to_train
        return torch.cat([
            base_ang_vel * 0.25,
            gravity,
            torch.tensor(vel_cmd, device=self.device).view(1, 3),
            joint_pos_rel[:12].view(1, -1),
            (joint_vel[:12] * 0.05).view(1, -1),
            actions_train_leg,
        ], dim=-1)

    def _to_env_action(self, leg_train):
        """Walking-policy (velocity tracker) 12-D output -> 20-D env action."""
        leg_env = leg_train * self.train_to_env
        action = torch.zeros((1, 20), device=self.device, dtype=torch.float32)
        action[:, self.leg_idx] = leg_env
        action[:, self.arm_idx] = 0.0
        return action

    # teacher trained with JointPositionAction scale 0.25 (+use_default_offset); the env
    # leg action uses scale 0.5 (+use_default_offset). Same default offset, so env action
    # = teacher_raw * (0.25/0.5) = teacher_raw * 0.5, applied to ALL 20 dims.
    TEACHER_TO_ENV_SCALE = 0.5

    def _teacher_to_env_action(self, act_raw):
        """B2Piper teacher 20-D raw output -> 20-D leg-first env action (legs + arm)."""
        action = torch.zeros((1, 20), device=self.device, dtype=torch.float32)
        action[:, :20] = act_raw[:, :20] * self.TEACHER_TO_ENV_SCALE
        return action

    # ------------------------------------------------------------------ predicts
    def predicts(self, obs, current_score):
        proprio = obs["proprio"].to(self.device)
        action_dim = (int(proprio.shape[-1]) - 12) // 3          # 20
        base_lin_vel = proprio[0, 0:3]
        joint_pos_rel = proprio[0, 12:12 + action_dim]
        joint_vel = proprio[0, 12 + action_dim:12 + 2 * action_dim]
        last_action = proprio[0, 12 + 2 * action_dim:12 + 3 * action_dim]

        meas_vx, meas_vy = float(base_lin_vel[0]), float(base_lin_vel[1])
        meas_wz = float(proprio[0, 5])                    # base_ang_vel z (proprio idx5)

        # ----- phase transitions -----
        # PUSH: shove the box into the -y-lane pit (+14). Once near the lane, RECOVER or
        #   BACKOFF -> CLIMB (the teacher walk-across).
        reached_lane = self.rx >= -0.8 and self.box_cx >= (BOX_PARKED_CX - 0.3)
        if self.CLIMB_MODE == "recover" and current_score >= 14.0 \
                and self.phase in ("PUSH", "BACKOFF"):
            if self.phase != "RECOVER":
                self.phase = "RECOVER"
                self._post_step = 0
        elif self.phase == "PUSH" and (reached_lane
                                       or self.step_i > self.PUSH_TIMEOUT_STEPS):
            self.phase = "BACKOFF"
            self._backoff_target_x = self.rx - self.BACKOFF_DIST
        if self.phase == "BACKOFF" and (self.rx <= self._backoff_target_x
                                        or self.step_i > self.PUSH_TIMEOUT_STEPS + 400):
            self.phase = "CLIMB"

        if self.phase == "RECOVER":
            # open-loop proven +16 recovery: strafe -y into the crossing lane, then walk +x.
            if self._post_step < self.STRAFE_STEPS:
                vx, vy, wz = 0.0, -self.V_STRAFE, 0.0
            else:
                vx, vy, wz = self.V_FORWARD_POST, 0.0, 0.0
            self._post_step += 1
            self._last_cmd = (vx, vy, wz)
            walking_obs = self._walking_obs(obs, joint_pos_rel, joint_vel,
                                            last_action, (vx, vy, wz))
            with torch.inference_mode():
                leg_train = self.policy(walking_obs)
            if not isinstance(leg_train, torch.Tensor):
                leg_train = torch.as_tensor(leg_train, device=self.device)
            leg_train = leg_train.to(self.device, torch.float32)
            if leg_train.ndim == 1:
                leg_train = leg_train.unsqueeze(0)
            action = self._to_env_action(leg_train)
            self._integrate_odometry(meas_vx, meas_vy, meas_wz)
            self.step_i += 1
            return {'action': action.cpu().numpy().tolist(), 'giveup': False}

        if self.phase in ("PUSH", "BACKOFF"):
            if self.phase == "BACKOFF":
                # walk straight back (-x) on the crossing lane to open run-up room.
                vy = max(-0.5, min(0.5, 1.5 * (CROSS_LANE_Y - self.ry)))
                vx, wz = self.V_RETREAT, 0.0
            else:
                # Open-loop push: (1) get BEHIND the box in x; (2) strafe onto its y-line;
                # (3) push toward the -y-lane pit target (drives box -x->+x and +y->-y).
                # The box spawns at +1.6; we approach from behind (-x) on its y-line, then
                # push it diagonally toward (0.0, -0.45). For a first version we push +x
                # while steering the y-line toward the lane.
                behind_x = self.rx <= (self.box_cx - BOX_HALF_X - 0.45)
                target_y = self.box_cy
                on_y = abs(self.ry - target_y) <= 0.12
                if not behind_x and not on_y:
                    vx, vy, wz = self.V_BACK, 0.0, 0.0
                elif not on_y:
                    vy = self.V_SIDE if self.ry < target_y else -self.V_SIDE
                    if abs(self.ry - target_y) < 0.4:
                        vy *= 0.4
                    vx, wz = 0.0, 0.0
                else:
                    # push +x; steer slightly -y to drift the box toward the -y lane.
                    vx = self.V_PUSH if self.box_cx < BOX_PARKED_CX - 0.05 else self.V_APPROACH
                    vy = max(-0.5, min(0.5, 1.5 * (CROSS_LANE_Y - self.ry)))
                    wz = 0.0
            self._last_cmd = (vx, vy, wz)
            walking_obs = self._walking_obs(obs, joint_pos_rel, joint_vel,
                                            last_action, (vx, vy, wz))
            with torch.inference_mode():
                leg_train = self.policy(walking_obs)
            if not isinstance(leg_train, torch.Tensor):
                leg_train = torch.as_tensor(leg_train, device=self.device)
            leg_train = leg_train.to(self.device, torch.float32)
            if leg_train.ndim == 1:
                leg_train = leg_train.unsqueeze(0)
            action = self._to_env_action(leg_train)

        else:  # CLIMB — the B2Piper parkour teacher drives the legs+arm (walk-across)
            # steady forward vx command (teacher proprio idx10) for the 0.4 m walk-across.
            self._last_cmd = (self.V_APPROACH, 0.0, 0.0)
            act_raw = self._teacher_action(obs, joint_pos_rel, joint_vel, base_lin_vel)
            action = self._teacher_to_env_action(act_raw)
            # crude base-height tracking for the scan: ride the terrain under us + stance.
            # The robot crouches LOW (z 0.16-0.37) over the box (-0.4), rises to ~0.55 on
            # the far flat ground (0.0) -> terrain + ~0.45 stance, clamped to that band.
            self.rz = max(0.16, min(0.6,
                                    terrain_height(self.rx, self.ry,
                                                   self.box_cx, self.box_cy) + 0.45))

        self._integrate_odometry(meas_vx, meas_vy, meas_wz)

        self.step_i += 1
        return {'action': action.cpu().numpy().tolist(), 'giveup': False}
