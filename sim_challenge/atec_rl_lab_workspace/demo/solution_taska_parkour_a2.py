"""Task A (off-road) replay of the A2 Pro + Piper X PARKOUR TEACHER (20-DoF).

Policy: Isaaclab_Parkour run 2026-08-05_01-38-20 / exported_teacher/policy.pt
(A2 teacher, ActorCriticRMA, hist_encoding=True, normalizer baked in). Unlike the
B2 12-DoF teacher (solution_taska_parkour12.py), the A2 teacher is a FULL 20-DoF
policy: it observes and outputs all 20 joints (12 legs + 8 arm) in ARTICULATION
order, so the arm is driven by the policy (not held at default).

Verified obs spec (from the teacher's state_dict: history_encoder in=77,
scan_encoder in=132, actor out=20, priv_encoder in=45):
  proprio 77 = ang_vel*0.25(3) imu[roll,pitch](2) 0(1) delta_yaw(1) delta_next_yaw(1)
               0,0(2) vx_cmd(1) env_idx(1) invert_env_idx(1)
               joint_pos_rel(20) joint_vel*0.05(20) last_action(20) contact_fill(4)
  scandot 132 = clip(base_z - terrain_h - 0.3, -1, 1), 12x11 grid (driver injects live)
  priv_explicit 9 = [base_lin_vel*2 (3), 0*6]
  priv_latent 45 = zeros (ignored under hist_encoding)
  history 770 = 10 x proprio77 (idx6:8 zeroed), init = 10x current
  => full obs = 77 + 132 + 9 + 45 + 770 = 1033

Joint order: the teacher's 20 joints are ARTICULATION (physics DOF) order; the Task A
A2Piper proprio/action are PER-LEG order. The 20-DoF permutation is the SAME one the
A2 lidar student uses (same A2 USD, same joint_names=['.*'] articulation resolve):
  teacher_vec[i]  = taska_vec[STU_FROM_TASKA[i]]   (Task A obs joints -> teacher order)
  taska_act[j]    = teacher_act[TASKA_FROM_STU[j]] (teacher action   -> Task A order)

Scan: the driver (play_taska_parkour.py) raycasts the REAL Task A ground at the 132
grid points and injects self._live_scan.
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

# A2 teacher checkpoint. Use the SELF-CONTAINED export (policy_with_est.pt) whose forward
# runs the estimator internally to fill the 9-dim priv_explicit slot from proprio -- so this
# adapter can leave that slot at zeros. (The plain policy.pt has no estimator and would read
# whatever we put there, which was the main cause of the Task A mis-walking.)
_A2_TEACHER = os.environ.get(
    "TASKA_A2_TEACHER",
    "/hdd0/sohyun/atec_2026_tesk_e/Isaaclab_Parkour/logs/rsl_rl/"
    "unitree_go2_parkour/2026-08-05_01-38-20/exported_teacher/policy_with_est.pt",
)

# --- 20-DoF permutation (verified by the A2 lidar student, same articulation<->per-leg) ---
STU_FROM_TASKA = [3, 0, 9, 6, 4, 1, 10, 7, 12, 5, 2, 11, 8, 13, 14, 15, 16, 17, 18, 19]
TASKA_FROM_STU = [1, 5, 10, 0, 4, 9, 3, 7, 12, 2, 6, 11, 8, 13, 14, 15, 16, 17, 18, 19]


class AlgSolution:

    ACTION_SCALE = 0.5                          # Task A env leg/arm action scale
    NUM_JOINTS = 20                             # A2Piper: 12 legs + 8 arm
    HOLD_ARM = True                             # pin the 8 arm joints at default (action=0)
    ALIGN_ARM_OBS = True                        # also zero arm slots in teacher obs (last_action/jvel)
    # DIAGNOSTIC: left-right mirror the DIRECTIONAL obs (roll, yaw-rate, roll-rate, delta_yaw,
    # gravity_y, scan y-grid). If drift is caused by a y-axis SIGN bug in the obs, mirroring flips
    # the drift to the OTHER side (+y). If it still drifts -y, the cause is NOT obs sign.
    # Enabled by env var MIRROR_LR=1. Does NOT mirror joints (isolates "which way teacher steers").
    MIRROR_LR = bool(int(os.environ.get("MIRROR_LR", "0")))

    NUM_PROP = 3 + 2 + 3 + 4 + 3 * 20 + 5       # 77
    NUM_SCAN = 132
    NUM_PRIV_E = 9
    NUM_PRIV_L = 45                             # verified from the real teacher obs (was 29!)
    NUM_HIST = 10
    PARKOUR_OBS_DIM = NUM_PROP + NUM_SCAN + NUM_PRIV_E \
        + NUM_PRIV_L + NUM_HIST * NUM_PROP      # 1033
    # priv_latent is ignored under hist_encoding=True (the actor's infer_hist_latent reads the
    # history block, not priv_latent), so filling these 45 dims with zeros is exact -- but the
    # SLOT must exist or the whole obs shifts and the history is read from the wrong offset.

    PARKOUR_TRAIN_SCALE = 0.25                  # teacher trained with action scale 0.25
    PARKOUR_OUT_RESCALE = PARKOUR_TRAIN_SCALE / ACTION_SCALE   # 0.5
    CROSS_VX = float(os.environ.get("CROSS_VX", "0.50"))   # forward vel COMMAND (env-overridable).
    # NOTE: this is the command, not the realized speed -- the teacher walks ~1.7x faster than the
    # command (vx_cmd=0.5 -> ~0.85 m/s measured), because reward_tracking_goal_vel doesn't penalize
    # overshoot. To target a REAL speed, divide by ~1.7 (e.g. 0.5 m/s -> CROSS_VX~0.3). Training
    # range was lin_vel_x=(0.3, 0.8), so keep CROSS_VX >= 0.3 to stay in the trained gait regime.
    STEP_DT = 0.02
    FLAT_SCAN_VALUE = 0.12                      # measured flat-ground scandot
    # Line-return steering: the parkour teacher is a GOAL tracker -- it walks toward a target
    # POINT (delta_yaw = bearing to the goal minus yaw), which is how it stays on course in the
    # training env. We don't know the finish line, but spawn point + spawn heading define a
    # reference LINE. We put a look-ahead goal point on that line, a fixed distance ahead of the
    # robot's projected progress, and steer toward it -- so a sideways drift rotates the bearing
    # and geometrically pulls the robot back onto the line (no crab-walk). LOOKAHEAD sets how
    # sharply it returns: smaller = tighter/faster return but more aggressive turning; larger =
    # gentler. LINE_RETURN=0 falls back to a fixed spawn-yaw target (the old drifting behaviour).
    LINE_RETURN = bool(int(os.environ.get("LINE_RETURN", "1")))
    LOOKAHEAD = float(os.environ.get("LOOKAHEAD", "2.0"))           # metres ahead on the line
    ZERO_DYAW = bool(int(os.environ.get("ZERO_DYAW", "0")))         # diag: force delta_yaw=0
    # env_idx / invert_env_idx encoding mode. Training set env_idx = (terrain != 'parkour_flat')
    # PER-STEP, so the teacher runs flat-gait on flat ground and rough-gait on rough ground.
    #   ENV_MODE=auto : classify from the live scan each step (flat if scan is nearly uniform).
    #                   This matches training and handles mixed courses (Task A flat + course rough).
    #   ENV_MODE=flat : force flat  (env_idx=0) -- correct for pure-flat Task A.
    #   ENV_MODE=rough: force rough (env_idx=1) -- the old (wrong-on-flat) behaviour.
    ENV_MODE = os.environ.get("ENV_MODE", "auto")
    ENV_FLAT_STD = float(os.environ.get("ENV_FLAT_STD", "0.03"))    # scan-std below this => flat

    def __init__(self):
        self.device = 'cuda'
        self.parkour = torch.jit.load(_A2_TEACHER, map_location=self.device)
        self.parkour.eval()

        self.scan_local = torch.tensor(_SCAN_LOCAL, device=self.device, dtype=torch.float32)
        self.stu_from_taska = torch.tensor(STU_FROM_TASKA, device=self.device, dtype=torch.long)
        self.taska_from_stu = torch.tensor(TASKA_FROM_STU, device=self.device, dtype=torch.long)
        # Arm joints in TEACHER (articulation) order are INTERLEAVED, not the last 8:
        #   teacher order = FL/FR/RL/RR hip, FL/FR/RL/RR thigh, j1, FL/FR/RL/RR calf, j2..j8
        #   so the 8 arm indices are [8, 13, 14, 15, 16, 17, 18, 19] -- j1 sits at 8, between the
        #   thigh and calf blocks. (The old code used [12:20], which WRONGLY zeroed RR_calf (a LEG,
        #   idx 12) and MISSED j1 (idx 8) -- a persistent lie about the right-rear leg that biased
        #   the gait and made the robot crab-walk to the right.)
        self.arm_teacher_idx = torch.tensor([8, 13, 14, 15, 16, 17, 18, 19],
                                            device=self.device, dtype=torch.long)
        # Arm-hold action (Task A per-leg order joint1..joint8) that drives the arm to the FOLDED
        # pose (all arm joints -> 0) like the colleague's a2piper. target = default + scale*action;
        # default = [0, 0.8, -0.8, 0,0,0,0,0], scale 0.5 -> action = (0 - default)/0.5.
        _arm_default = [0.0, 0.8, -0.8, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.arm_hold_action = torch.tensor([(0.0 - d) / self.ACTION_SCALE for d in _arm_default],
                                            device=self.device, dtype=torch.float32)

        self._live_scan = None
        self._live_yaw = None           # driver injects the robot's TRUE world yaw each step
        self._live_roll = None          # driver injects TRUE euler roll (imu obs slot 3)
        self._live_pitch = None         # driver injects TRUE euler pitch (imu obs slot 4)
        self._live_pos = None           # driver injects the robot's TRUE world (x, y) each step
        self._live_goal_point = None    # driver may inject a look-ahead goal (x,y) on a CURVED path
        self._live_offset = None        # driver may inject the signed lateral offset (curve frame)
        self._init_episode_state()

    def _init_episode_state(self):
        self._ts = 0
        self.ryaw = 0.0
        self._goal_yaw = None           # locked to the spawn yaw on the first step (= "go straight")
        self._spawn_yaw = None          # spawn heading, defines the reference LINE direction
        self._spawn_pos = None          # spawn (x, y), the reference LINE anchor
        self._last_offset = 0.0         # last signed lateral offset from the spawn line (debug)
        self._parkour_hist = None
        self._teacher_last_action = torch.zeros(self.NUM_JOINTS, device=self.device)  # teacher order

    def get_action_spec(self) -> dict[str, dict[str, Any]] | None:
        return {}

    def reset(self, **kwargs) -> None:
        self._init_episode_state()
        return None

    def _integrate_yaw(self, obs):
        # base_ang_vel z lives at proprio[5] in the Task A proprio layout
        wz = float(obs["proprio"].to(self.device)[0][5])
        self.ryaw += wz * self.STEP_DT

    def _scan(self) -> torch.Tensor:
        if self._live_scan is not None:
            s = self._live_scan.to(self.device, torch.float32).view(-1)[:self.NUM_SCAN]
        else:
            s = torch.full((self.NUM_SCAN,), self.FLAT_SCAN_VALUE,
                           device=self.device, dtype=torch.float32)
        if self.MIRROR_LR:
            # grid is 11 y-blocks (outer) x 12 x-points (inner). Mirror y = reverse the y-blocks.
            s = s.view(11, 12).flip(0).reshape(-1)
        return s

    def _build_proprio77(self, obs) -> torch.Tensor:
        proprio = obs["proprio"].to(self.device)[0]
        # Task A proprio group (72) = lin_vel(3) ang_vel(3) cmd(3) grav(3) jpos(20) jvel(20) act(20)
        base_ang_vel = proprio[3:6]
        gravity = proprio[9:12]
        # The teacher's imu obs (slots 3,4) is EXACTLY [wrap_to_pi(roll), wrap_to_pi(pitch)] from
        # euler_xyz_from_quat(root_quat_w) at training time -- an euler decomposition, NOT a
        # projected-gravity reconstruction. Prefer the driver-injected TRUE euler roll/pitch; the
        # gravity approximation below is only a fallback and can carry a sign error that makes the
        # teacher counter-bank and drift sideways.
        if getattr(self, "_live_roll", None) is not None:
            roll = float(self._live_roll)
            pitch = float(self._live_pitch)
        else:
            roll = math.atan2(-float(gravity[1]), -float(gravity[2]))
            pitch = math.atan2(float(gravity[0]),
                               math.sqrt(float(gravity[1])**2 + float(gravity[2])**2) + 1e-6)
        # full 20-joint slices, permute per-leg (Task A) -> articulation (teacher) order
        jpos = proprio[12:12 + self.NUM_JOINTS][self.stu_from_taska].clone()
        jvel = proprio[12 + self.NUM_JOINTS:12 + 2 * self.NUM_JOINTS][self.stu_from_taska].clone()
        # A held arm has ~0 velocity; force the arm jvel slots (teacher-order arm indices, which
        # are INTERLEAVED: [8,13..19]) to exactly 0 so proprio reflects a stationary arm.
        if self.HOLD_ARM and self.ALIGN_ARM_OBS:
            jvel[self.arm_teacher_idx] = 0.0

        # delta_yaw = direction FROM the robot's current position TO a look-ahead GOAL POINT,
        # minus current yaw -- exactly how the training env builds it (target_yaw = atan2 of the
        # vector to the next goal; delta_yaw = target_yaw - yaw). The goal point is a fixed
        # distance ahead ON the spawn line, so if the robot drifts sideways the bearing to that
        # point rotates and produces a heading error that pulls it BACK ONTO the line -- the same
        # position-restoring behaviour the teacher learned. This beats a fixed spawn-yaw target
        # (no position feedback -> parallel drift) and a raw yaw-bias (crab-walk), because here the
        # correction is geometric: the robot walks TOWARD a point on the line, not just faces a
        # biased heading.
        if self._live_yaw is not None:
            cur_yaw = float(self._live_yaw)
            if self._spawn_yaw is None:
                self._spawn_yaw = cur_yaw         # reference LINE direction = spawn heading
                if self._live_pos is not None:
                    self._spawn_pos = (float(self._live_pos[0]), float(self._live_pos[1]))

            if self._live_goal_point is not None and self._live_pos is not None:
                # The DRIVER supplies a look-ahead goal point (world XY) -- used on a CURVED course,
                # where "straight" is the curve tangent, not the spawn line. We just steer toward
                # the injected point, same geometric look-ahead as the straight case.
                px, py = float(self._live_pos[0]), float(self._live_pos[1])
                gx, gy = float(self._live_goal_point[0]), float(self._live_goal_point[1])
                self._last_offset = float(self._live_offset) if self._live_offset is not None else 0.0
                goal_yaw = math.atan2(gy - py, gx - px)
            elif self.LINE_RETURN and self._spawn_pos is not None and self._live_pos is not None:
                px, py = float(self._live_pos[0]), float(self._live_pos[1])
                # progress ALONG the spawn line (projection of current pos onto the line dir)
                cx, cy = self._spawn_pos
                cyaw, syaw = math.cos(self._spawn_yaw), math.sin(self._spawn_yaw)
                s_along = (px - cx) * cyaw + (py - cy) * syaw
                # goal point = a fixed look-ahead distance further along the line
                gx = cx + (s_along + self.LOOKAHEAD) * cyaw
                gy = cy + (s_along + self.LOOKAHEAD) * syaw
                # signed lateral offset (for debug/telemetry only)
                self._last_offset = -syaw * (px - cx) + cyaw * (py - cy)
                # bearing from the robot to the goal point = training-style target_yaw
                goal_yaw = math.atan2(gy - py, gx - px)
            else:
                goal_yaw = self._spawn_yaw

            self._goal_yaw = goal_yaw
            dy = goal_yaw - cur_yaw
            # wrap to (-pi, pi]
            dy = math.atan2(math.sin(dy), math.cos(dy))
            delta_yaw = dy
        else:
            delta_yaw = -self.ryaw                 # fallback if the driver injects no yaw
        # DIAGNOSTIC: force delta_yaw=0 (pure "go straight, no steering") to isolate the teacher's
        # bare flat-ground bias -- whichever way it drifts with ZERO steering IS the bias.
        if self.ZERO_DYAW:
            delta_yaw = 0.0
        p = torch.zeros(self.NUM_PROP, device=self.device, dtype=torch.float32)
        p[0:3] = base_ang_vel * 0.25
        p[3] = roll
        p[4] = pitch
        p[5] = 0.0
        p[6] = delta_yaw
        p[7] = delta_yaw                         # delta_next_yaw ~ delta_yaw
        p[8] = 0.0
        p[9] = 0.0
        p[10] = self.CROSS_VX                     # vx command
        # env_idx / invert_env_idx encode the terrain TYPE (training: env_idx = terrain != flat,
        # per-step). Feeding rough-terrain=1 on flat ground makes the teacher run a rough-terrain
        # gait on flat ground -> asymmetric foot placement -> a persistent turn bias. Classify the
        # terrain from the live scan: a nearly-uniform scan (std < ENV_FLAT_STD) is flat.
        if self.ENV_MODE == "flat":
            is_flat = True
        elif self.ENV_MODE == "rough":
            is_flat = False
        else:  # auto: classify from the current scan
            is_flat = float(self._scan().std()) < self.ENV_FLAT_STD
        self._is_flat = is_flat                  # for ACT_DEBUG
        p[11] = 0.0 if is_flat else 1.0          # env_idx
        p[12] = 1.0 if is_flat else 0.0          # invert_env_idx
        p[13:33] = jpos
        p[33:53] = jvel * 0.05
        p[53:73] = self._teacher_last_action     # 20, teacher order
        p[73:77] = 0.0                           # contact_fill unknown
        if self.MIRROR_LR:
            # Flip the DIRECTIONAL (yaw/roll) components: roll-rate (idx0), yaw-rate (idx2),
            # roll (idx3), delta_yaw (idx6,7). Pitch-rate (idx1) and pitch (idx4) are LR-symmetric.
            p[0] = -p[0]
            p[2] = -p[2]
            p[3] = -p[3]
            p[6] = -p[6]
            p[7] = -p[7]
        return p

    def _parkour_action(self, obs) -> torch.Tensor:
        proprio = obs["proprio"].to(self.device)[0]
        base_lin_vel = proprio[0:3]

        prop = self._build_proprio77(obs)                          # (77,)
        scan = self._scan()                                        # (132,)

        prop_hist = prop.clone()
        prop_hist[6:8] = 0.0
        if self._parkour_hist is None:
            self._parkour_hist = prop_hist.view(1, 1, -1).repeat(1, self.NUM_HIST, 1)
        else:
            self._parkour_hist = torch.cat(
                [self._parkour_hist[:, 1:], prop_hist.view(1, 1, -1)], dim=1)

        # priv_explicit is filled INTERNALLY by policy_with_est.pt's estimator (from proprio),
        # so we leave it at zeros here -- the teacher overwrites this slot before the actor.
        priv_e = torch.zeros(self.NUM_PRIV_E, device=self.device)
        priv_l = torch.zeros(self.NUM_PRIV_L, device=self.device)

        full = torch.cat([prop, scan, priv_e, priv_l,
                          self._parkour_hist.reshape(-1)]).view(1, -1)      # (1, 1033)
        if os.environ.get("ACT_DEBUG") and self._ts == 100:
            print(f"[OBSDIM] full obs dim = {full.shape[-1]} (should be 1033)", flush=True)
            torch.save({"prop": prop.detach().cpu(), "scan": scan.detach().cpu(),
                        "full": full[0].detach().cpu()}, "/hdd0/sohyun/tmp/taska_adapter_obs.pt")
            print("[SAVED_ADAPTER_OBS at t=100]", flush=True)
        with torch.inference_mode():
            act_raw = self.parkour(full)                           # normalizer baked in
        if not isinstance(act_raw, torch.Tensor):
            act_raw = torch.as_tensor(act_raw, device=self.device)
        act_raw = act_raw.to(self.device, torch.float32)
        if act_raw.ndim == 1:
            act_raw = act_raw.unsqueeze(0)
        self._teacher_last_action = act_raw[0].clone()             # 20, teacher order (raw)
        # When we OVERRIDE the arm to a fixed pose (action=0), the command we actually sent to
        # the arm is 0 -- not the teacher's arm output. Feed that truth back into the teacher's
        # last_action obs (arm slots -> 0) so proprio stays self-consistent; otherwise the teacher
        # believes it moved the arm, and that phantom mismatch can perturb the LEG outputs.
        # Arm joints are the INTERLEAVED teacher-order indices [8,13..19] (j1 at 8), NOT [12:20].
        if self.HOLD_ARM and self.ALIGN_ARM_OBS:
            self._teacher_last_action[self.arm_teacher_idx] = 0.0

        # rescale, permute teacher -> Task A per-leg order (all 20 joints incl. arm)
        act_taska = (act_raw[0] * self.PARKOUR_OUT_RESCALE).clamp(-9.6, 9.6)[self.taska_from_stu]
        # Hold the 8 arm joints (Task A per-leg indices 12..19 = joint1..joint8) at the FOLDED
        # pose the colleague uses (arm tucked behind the back): every arm joint at angle 0.
        # env uses use_default_offset=True so target = default + scale*action; to reach target 0
        # we send action = (0 - default)/scale. Our A2 arm default is [0, 0.8, -0.8, 0,0,0,0,0]
        # (joint2/joint3 slightly tucked), so the offsets are [0, -1.6, +1.6, 0,0,0,0,0] at scale
        # 0.5. This makes the arm stand straight/folded like the colleague's a2piper rest pose
        # instead of our forward-tucked default.
        if self.HOLD_ARM:
            act_taska[12:20] = self.arm_hold_action
        return act_taska.view(1, -1)

    def predicts(self, obs, current_score):
        self._integrate_yaw(obs)
        action = self._parkour_action(obs)
        self._ts += 1
        # DIAG: accumulate the time-averaged RAW teacher action (teacher order) to expose any
        # systematic left/right asymmetry. Printed at t=200. teacher order legs:
        #   [0]FL_hip [1]FR_hip [2]RL_hip [3]RR_hip [4]FL_thigh [5]FR_thigh [6]RL_thigh [7]RR_thigh
        #   [9]FL_calf [10]FR_calf [11]RL_calf [12]RR_calf   (arm at 8,13..19)
        if os.environ.get("ACT_AVG"):
            if not hasattr(self, "_act_sum"):
                self._act_sum = torch.zeros(20, device=self.device); self._act_n = 0
            self._act_sum += self._teacher_last_action; self._act_n += 1
            if self._ts == 200:
                m = (self._act_sum / self._act_n).tolist()
                print(f"[ACTAVG] FL_hip={m[0]:+.3f} FR_hip={m[1]:+.3f} (L-R={m[0]-m[1]:+.3f}) | "
                      f"RL_hip={m[2]:+.3f} RR_hip={m[3]:+.3f} (L-R={m[2]-m[3]:+.3f})", flush=True)
                print(f"[ACTAVG] FL_thi={m[4]:+.3f} FR_thi={m[5]:+.3f} (L-R={m[4]-m[5]:+.3f}) | "
                      f"RL_thi={m[6]:+.3f} RR_thi={m[7]:+.3f} (L-R={m[6]-m[7]:+.3f})", flush=True)
                print(f"[ACTAVG] FL_cal={m[9]:+.3f} FR_cal={m[10]:+.3f} (L-R={m[9]-m[10]:+.3f}) | "
                      f"RL_cal={m[11]:+.3f} RR_cal={m[12]:+.3f} (L-R={m[11]-m[12]:+.3f})", flush=True)
        if os.environ.get("ACT_DEBUG") and (self._ts <= 5 or self._ts % 25 == 0):
            _sc = self._scan()
            _a = action.view(-1)
            _dy = (self._goal_yaw - float(self._live_yaw)) if (self._live_yaw is not None and self._goal_yaw is not None) else float('nan')
            _leg = _a[:12]; _arm = _a[12:]
            _raw = self._teacher_last_action  # teacher order (20), raw policy output
            _envidx = 0 if getattr(self, "_is_flat", True) else 1
            _pit = f"{self._live_pitch:+.2f}" if self._live_pitch is not None else "None"
            print(f"[A2P t={self._ts:4d}] pitch={_pit} dyaw={_dy:+.3f} offset={self._last_offset:+.2f} "
                  f"scan_m={float(_sc.mean()):+.2f} scan_std={float(_sc.std()):.3f} "
                  f"env_idx={_envidx}({'flat' if _envidx==0 else 'ROUGH'}) "
                  f"leg[{float(_leg.min()):+.2f},{float(_leg.max()):+.2f}] "
                  f"arm[{float(_arm.min()):+.2f},{float(_arm.max()):+.2f}] "
                  f"rawT[{float(_raw.min()):+.2f},{float(_raw.max()):+.2f}]", flush=True)
        return {'action': action.cpu().numpy().tolist(), 'giveup': False}
