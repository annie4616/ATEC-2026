"""Task A (off-road Hiking) driver for the A2 Pro + Piper X **LiDAR parkour student**.

Policy artifacts (from Isaaclab_Parkour distill run 2026-08-05_09-05-07):
  - exported_deploy/policy.pt        : _ParkourDeployTorchPolicyExporter
        forward(x_obs_1033, scandots_latent_32) -> 20 leg+arm actions.
        estimator fills priv_explicit internally; normalizer baked in.
  - exported_deploy/depth_latest.pt  : _ParkourDeployTorchDepthEncoderExporter
        forward(lidar_5760, proprio_77) -> depth_latent_and_yaw (last 2 = yaw,
        first 32 = scandots latent).

Unlike the old 12-DoF B2 scandot teacher (solution_taska_parkour12.py), this student:
  * is A2Piper 20-DoF, so joint order to the Task A A2Piper env is **identity**
    (both per-leg FR/FL/RR/RL + joint1..8) — NO permutation.
  * consumes the **5760-ray LiDAR directly** (Task A env already emits extero.lidar_scan),
    so NO synthetic height-grid raycast is needed; the scan(132) slice of the policy obs is
    unused because scandots_latent is supplied (actor ignores obs_scan when latent given).
  * holds the 8 arm joints at a fixed **stow-behind-the-back** target, ignoring the policy's
    arm output (arm stowed: joint1/2/3 = 2.6/3.1/-2.9, joint4..8 = 0; validated in
    new_robot_a2pro_piperx/sweep_arm_stow.py, pose "F", robot STANDS, no leg/ground clash).

Student obs (1033) — only proprio(77) + its 10-frame history matter here; scan(132),
priv_latent(45) are zeros (actor uses the latent / hist path), priv_explicit(9) is filled
by policy.pt's estimator:
  [0:77]      proprio77
  [77:209]    scan132        = 0 (scandots_latent supplied separately)
  [209:218]   priv_explicit9 = 0 (policy.pt overwrites via estimator)
  [218:263]   priv_latent45  = 0 (ignored under hist_encoding=True)
  [263:1033]  history 770    = 10 x proprio77 (idx6:8 zeroed), init = 10x current

proprio77 layout (parkour ExtremeParkourObservations, njoints=20):
  ang_vel*0.25(3) imu_roll/pitch(2) 0(1) delta_yaw(1) delta_next_yaw(1) 0,0(2)
  vx_cmd(1) env_idx(1) invert_env_idx(1) joint_pos_rel(20) joint_vel*0.05(20)
  last_action(20) contact_fill(4)  = 3+2+3+4+3*20+5 = 77

Task A env proprio group (72) = [lin_vel(3) ang_vel(3) cmd(3) grav(3) jpos(20) jvel(20) act(20)].
"""

import os
import sys
import math
from typing import Any

import torch

current_path = os.path.dirname(os.path.abspath(__file__))
if current_path not in sys.path:
    sys.path.insert(0, current_path)


class AlgSolution:

    ACTION_SCALE = 0.5                               # Task A env leg/arm action scale
    PARKOUR_TRAIN_SCALE = 0.25                       # student trained action scale
    OUT_RESCALE = PARKOUR_TRAIN_SCALE / ACTION_SCALE # 0.5

    NUM_LEG = 12
    NUM_ARM = 8
    NUM_JOINTS = 20

    # parkour student obs dims (njoints=20)
    NUM_PROP = 3 + 2 + 3 + 4 + 3 * 20 + 5            # 77
    NUM_SCAN = 132
    NUM_PRIV_E = 9
    NUM_PRIV_L = 45
    NUM_HIST = 10
    OBS_DIM = NUM_PROP + NUM_SCAN + NUM_PRIV_E + NUM_PRIV_L + NUM_HIST * NUM_PROP  # 1033

    NUM_LIDAR = 5760
    LIDAR_OFFSET = 0.5                               # matches lidar_features offset in training
    LIDAR_MAXDIST = 10.0                             # LidarPatternCfg max_distance
    CROSS_VX = float(os.environ.get("CROSS_VX", "0.5"))   # forward vel COMMAND (not realized speed:
    # the policy walks faster than the command since reward_tracking_goal_vel doesn't penalize
    # overshoot). Env-overridable so we can dial the REALIZED speed down toward a target m/s.
    # env_idx / invert_env_idx encode the terrain TYPE (training: env_idx=(terrain!=parkour_flat),
    # per-step, on a MIXED flat+rough curriculum -> the student LEARNED to switch gait by this bit).
    # So we must set it dynamically, not force flat. We don't get the sim's terrain name at deploy,
    # so infer it from the LiDAR: rays hitting rough ground scatter in height (large std) vs a flat
    # floor (near-uniform). ENV_MODE: auto=infer from lidar, flat/rough=force.
    ENV_MODE = os.environ.get("ENV_MODE", "auto")
    ENV_ROUGH_STD = float(os.environ.get("ENV_ROUGH_STD", "0.07"))   # lidar height-std -> rough
    # 0.07 (was 0.15): the course's rough/stair sections measure a lidar height-std of ~0.10-0.11,
    # which never exceeded 0.15, so env_idx stayed flat=True on rough terrain (the robot ran the
    # FLAT gait over stairs -> stumbled). 0.07 is above flat noise (~0.03-0.05) but below the
    # measured rough std, so env_idx correctly flips to ROUGH on stairs/slopes. Verified on course.
    # obs[6:8] holds delta_yaw. Training replaces it with the lidar-predicted yaw ONLY when
    # delta_yaw_ok (|delta_yaw| < 0.6); otherwise it keeps the real delta_yaw so the policy turns.
    DELTA_YAW_OK_THRESH = float(os.environ.get("DELTA_YAW_OK", "0.6"))
    # STRAIGHT_MODE: steer toward a FIXED world heading (the course forward axis) every step, so a
    # left/right bias is continually corrected back to straight -- no wobble, no spawn-measurement
    # error. GOAL_YAW is the target world yaw in radians: Task A runs -x -> +x, i.e. +x = yaw 0.
    STRAIGHT_MODE = bool(int(os.environ.get("STRAIGHT_MODE", "1")))
    GOAL_YAW = float(os.environ.get("GOAL_YAW", "0.0"))            # world heading to hold (+x = 0)
    # depth_encoder yaw prediction overwrites our delta_yaw only if enabled (default OFF: the
    # deploy env has no training goals so the prediction misleads; we steer with our own delta_yaw).
    USE_DEPTH_YAW = bool(int(os.environ.get("USE_DEPTH_YAW", "0")))
    STEP_DT = 0.02
    DEPTH_UPDATE_EVERY = 5                           # depth_encoder cadence (matches play.py)

    # WARP_ENCODER: the deployed depth_encoder is the warp-reprojection backbone
    # (WarpScandotsBackbone), which needs an EXTENDED 88-D proprio input (proprio77 + pose11). The
    # adapter appends the synthesized world pose before calling the encoder. Default ON for the
    # warp-matched deploy; set WARP_ENCODER=0 to fall back to a legacy 77-D 1D/conv encoder.
    WARP_ENCODER = bool(int(os.environ.get("WARP_ENCODER", "1")))
    # USE_GT_ODOM: use the course driver's injected GROUND-TRUTH pose (_live_pos/_live_yaw) for the
    # warp pose instead of proprio dead-reckoning. Privileged (A/B baseline only); default OFF so the
    # deploy path is honest (proprio-integrated odometry). GT is available only when the driver
    # injects _live_pos/_live_yaw; falls back to proprio odometry if absent.
    USE_GT_ODOM = bool(int(os.environ.get("USE_GT_ODOM", "0")))

    # Arm target (rad), joint1..8. Hold ALL arm joints at 0 (folded/neutral) to match the
    # teacher the student was distilled from (teacher arm = 0, not the old stow-behind pose).
    ARM_STOW = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # all arm joints -> 0

    # --- 20-DoF joint-order permutation (the crux fix) --------------------------------------
    # The parkour student's obs/action are in ARTICULATION (physics DOF) order because its
    # action term used joint_names=[".*"] with preserve_order=False:
    #   STU = [FL_hip,FR_hip,RL_hip,RR_hip, FL_thigh,FR_thigh,RL_thigh,RR_thigh, joint1,
    #          FL_calf,FR_calf,RL_calf,RR_calf, joint2,joint3,joint4,joint5,joint6,joint7,joint8]
    # The Task A env's proprio joints AND joint_leg(12)+joint_arm(8) action are in PER-LEG order
    # (preserve_order=True on the name lists in a2.py):
    #   TASKA = [FR_hip,FR_thigh,FR_calf, FL_hip,FL_thigh,FL_calf, RR_hip,RR_thigh,RR_calf,
    #            RL_hip,RL_thigh,RL_calf, joint1..joint8]
    # These differ (DOF-grouped-vs-per-leg legs; arm interleaved-vs-contiguous), so feeding the
    # student Task-A-ordered joints (and applying its action as Task-A-ordered) was scrambling
    # the legs — the robot couldn't walk and just sank/twitched in place.
    #   stu_vec[i]    = taska_vec[STU_FROM_TASKA[i]]     (Task A obs joints -> student order)
    #   taska_act[j]  = stu_act[TASKA_FROM_STU[j]]       (student action  -> Task A order)
    STU_FROM_TASKA = [3, 0, 9, 6, 4, 1, 10, 7, 12, 5, 2, 11, 8, 13, 14, 15, 16, 17, 18, 19]
    TASKA_FROM_STU = [1, 5, 10, 0, 4, 9, 3, 7, 12, 2, 6, 11, 8, 13, 14, 15, 16, 17, 18, 19]

    def __init__(self):
        self.device = 'cuda'
        # LIDAR_MODEL_TAG selects the model pair: "" (default) = lo31k (old staircase teacher),
        # "overspeed_" = overspeed-teacher encoder (model_50000, latent_loss 0.032) that fixes lo31k's
        # speed overshoot. Both use the SAME LidarScanBackbone 1D path (LIDAR_2D unset at export).
        _tag = os.environ.get("LIDAR_MODEL_TAG", "")
        self.policy = torch.jit.load(current_path + f'/policy_taska_lidar_{_tag}student.pt',
                                     map_location=self.device)
        self.policy.eval()
        self.depth_encoder = torch.jit.load(current_path + f'/depth_taska_lidar_{_tag}student.pt',
                                            map_location=self.device)
        self.depth_encoder.eval()

        # TEACHER_SCANDOTS mode: run the RAW teacher (no lidar latent). policy_with_est.pt takes obs
        # only (obs[77:209]=true 132-pt scandots) and its own scan_encoder makes the latent + the
        # estimator fills priv_explicit internally. The course driver raycasts the true scandots and
        # injects them via self._live_scandots. This shows the privileged teacher gait in the course.
        self.TEACHER_SCANDOTS = int(os.environ.get("TEACHER_SCANDOTS", "0"))
        self.teacher_policy = None
        if self.TEACHER_SCANDOTS:
            self.teacher_policy = torch.jit.load(
                current_path + '/policy_taska_teacher_scandots.pt', map_location=self.device)
            self.teacher_policy.eval()
        self._live_scandots = None      # (132,) set each step by the course driver

        # arm stow target as an env-action: env applies action*scale about the arm default
        # offset (use_default_offset=True). We instead drive the arm to an ABSOLUTE stow pose,
        # so the arm action we emit must equal (stow - arm_default) / ACTION_SCALE. The arm
        # default (a2.py init) is joint1=0,j2=0.8,j3=-0.8,j4..8=0.
        arm_default = torch.tensor([0.0, 0.8, -0.8, 0.0, 0.0, 0.0, 0.0, 0.0], device=self.device)
        stow = torch.tensor(self.ARM_STOW, device=self.device)
        self.arm_action = ((stow - arm_default) / self.ACTION_SCALE).to(torch.float32)  # (8,)

        self.stu_from_taska = torch.tensor(self.STU_FROM_TASKA, device=self.device, dtype=torch.long)
        self.taska_from_stu = torch.tensor(self.TASKA_FROM_STU, device=self.device, dtype=torch.long)
        # In Task A (per-leg) order the 12 legs are indices 0..11 and the arm joint1..8 are 12..19.
        self.leg_idx = list(range(12))
        self.arm_idx = list(range(12, 20))
        # Arm joints in STUDENT (articulation) order are INTERLEAVED: joint1 sits at index 8,
        # joint2..8 at 13..19 (NOT the last 8). We hold the arm at 0, so the last_action we feed
        # back must report 0 for those slots (we commanded the arm to 0, not the policy's arm out).
        self.arm_stu_idx = torch.tensor([8, 13, 14, 15, 16, 17, 18, 19],
                                        device=self.device, dtype=torch.long)
        self._init_episode_state()

    LOOKAHEAD = 5.0                                  # metres ahead on the spawn line (teacher-tuned)

    def _init_episode_state(self):
        self._ts = 0
        self.ryaw = 0.0
        self._live_roll = None
        self._live_pitch = None
        self._live_yaw = None
        self._live_pos = None
        self._live_goal_point = None           # driver may inject a curve goal (x,y) for courses
        self._final_goal = None                # driver may inject the FIXED finish (x,y): steer straight to it
        self._live_delta_yaw = None            # driver may inject a DIRECT relative steer (LiDAR-only, real-world)
        self._spawn_yaw = None
        self._spawn_pos = None
        self._hist = None                      # (1, NUM_HIST, 77)
        self._last_action = torch.zeros(self.NUM_JOINTS, device=self.device)  # env order (20)
        self._scandots_latent = torch.zeros(1, 32, device=self.device)
        self._yaw = torch.zeros(1, 2, device=self.device)
        self._flat_latent = None               # EMA of flat-ground latent (FLAT_LATENT_CLAMP)
        # start the speed-clamp command LOW so the robot never lunges out of the gate before the
        # integral loop settles (it ramps up if measured speed is under target).
        self._vx_cmd = float(os.environ.get("VX_CMD_INIT", "0.2"))
        self._meas_vx = 0.5                    # measured forward speed (updated each step)
        # --- WARP encoder odometry (proprio-integrated world pose) ------------------------------
        # The warp depth_encoder needs the ABSOLUTE world pose to reproject accumulated lidar into
        # the current frame: pose11 = [sensor_pos3, sensor_quat4(wxyz), root_pos3, root_yaw1]. In
        # training these come from the sim (privileged). At deploy we synthesize them from proprio:
        #   root_yaw = self.ryaw (already integrated from base ang_vel z each step)
        #   root_xy  = integral of base_lin_vel rotated into world by ryaw (dead-reckoning odometry)
        #   root_z   = constant (warp uses only RELATIVE heights, so absolute z cancels)
        #   quat     = from (roll, pitch, ryaw); lidar is base_link-mounted (zero offset) so
        #              sensor_pos == root_pos and sensor_quat == root_quat.
        # The warp module is a rolling world heightmap centered on root_pos, so only RELATIVE
        # motion within its ~60-frame window matters -> small integration drift is tolerable
        # (validated: <0.004 m over the window vs the 0.15 m grid res).
        self._odo_x = 0.0
        self._odo_y = 0.0
        self._odo_z = float(os.environ.get("WARP_ODO_Z", "0.42"))  # nominal base height [m]
        self._warp_reset_pending = True        # reset the encoder's rolling map on first forward

    def get_action_spec(self) -> dict[str, dict[str, Any]] | None:
        return {}

    def reset(self, **kwargs) -> None:
        self._init_episode_state()
        return None

    def _integrate_yaw(self, obs):
        # base_ang_vel z is proprio[5] in the Task A proprio layout
        wz = float(obs["proprio"].to(self.device)[0][5])
        self.ryaw += wz * self.STEP_DT

    def _integrate_odom(self, obs):
        """Dead-reckon the world (x,y) position from the base-frame linear velocity + current yaw.
        proprio[0:3] = base_lin_vel (body frame). Rotate the horizontal part into world by ryaw
        and Euler-integrate. Only relative motion inside the warp window matters, so this modest
        drift is acceptable (validated ~0.004 m/window vs 0.15 m grid res)."""
        p = obs["proprio"].to(self.device)[0]
        vx_b = float(p[0]); vy_b = float(p[1])
        cy = math.cos(self.ryaw); sy = math.sin(self.ryaw)
        vx_w = cy * vx_b - sy * vy_b
        vy_w = sy * vx_b + cy * vy_b
        self._odo_x += vx_w * self.STEP_DT
        self._odo_y += vy_w * self.STEP_DT

    def _synth_pose(self) -> torch.Tensor:
        """Build the warp encoder's 11-D world pose [sensor_pos3, sensor_quat4(wxyz), root_pos3,
        root_yaw1]. Lidar is base_link-mounted (zero offset), so sensor_pos == root_pos and
        sensor_quat == root_quat. Two sources:
          - default: proprio dead-reckoning (self._odo_x/_y, self.ryaw, cached roll/pitch)
          - USE_GT_ODOM=1 and driver injected _live_pos/_live_yaw: privileged GT (A/B baseline)."""
        if self.USE_GT_ODOM and self._live_pos is not None and self._live_yaw is not None:
            rx = float(self._live_pos[0]); ry = float(self._live_pos[1])
            ryaw = float(self._live_yaw)
            roll = float(self._live_roll) if self._live_roll is not None else float(getattr(self, "_cur_roll", 0.0))
            pitch = float(self._live_pitch) if self._live_pitch is not None else float(getattr(self, "_cur_pitch", 0.0))
        else:
            rx = self._odo_x; ry = self._odo_y
            ryaw = self.ryaw
            roll = float(getattr(self, "_cur_roll", 0.0))
            pitch = float(getattr(self, "_cur_pitch", 0.0))
        rz = self._odo_z
        # euler (roll about x, pitch about y, yaw about z) -> quaternion (w,x,y,z), XYZ order
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cyq, syq = math.cos(ryaw * 0.5), math.sin(ryaw * 0.5)
        qw = cr * cp * cyq + sr * sp * syq
        qx = sr * cp * cyq - cr * sp * syq
        qy = cr * sp * cyq + sr * cp * syq
        qz = cr * cp * syq - sr * sp * cyq
        pose = torch.tensor([rx, ry, rz, qw, qx, qy, qz, rx, ry, rz, ryaw],
                            device=self.device, dtype=torch.float32).view(1, -1)  # (1,11)
        return pose

    def _lidar(self, obs) -> torch.Tensor:
        """Match the TRAINING lidar normalization EXACTLY.

        Training (lidar_features._scan, observations.py):
            h = sensor_z - hit_z - offset          # offset=0.5
            return h / clipping_range - 0.5        # clipping_range = max_distance = 10

        Task A's mdp.height_scan ALREADY returns `sensor_z - hit_z - offset` (offset 0.5
        subtracted inside the term). So the env-provided `scan` == training `h`, and the ONLY
        remaining step is `scan / max_distance - 0.5`. The previous code subtracted the offset
        a SECOND time ((scan - 0.5)/10 - 0.5), shifting every ray by 0.05 in latent space and
        making the student read a completely wrong terrain — the robot flailed from step 0.
        """
        # The Task A extero group has a single term (lidar_scan), so the manager returns it
        # as a flat (1, 5760) tensor, NOT a nested {"lidar_scan": ...} dict.
        extero = obs.get("extero")
        if extero is None:
            return torch.zeros(1, self.NUM_LIDAR, device=self.device)
        if isinstance(extero, dict):
            extero = extero.get("lidar_scan")
            if extero is None:
                return torch.zeros(1, self.NUM_LIDAR, device=self.device)
        scan = extero.to(self.device, torch.float32).view(1, -1)
        if scan.shape[1] != self.NUM_LIDAR:
            # pad/trim defensively
            s = torch.zeros(1, self.NUM_LIDAR, device=self.device)
            n = min(scan.shape[1], self.NUM_LIDAR)
            s[:, :n] = scan[:, :n]
            scan = s
        scan = torch.nan_to_num(scan, nan=0.0, posinf=0.0, neginf=0.0)
        return scan / self.LIDAR_MAXDIST - 0.5

    def _lidar_rough_std(self, obs) -> float:
        """Terrain-roughness proxy: std of the LiDAR ground-return heights. The raw extero scan is
        `sensor_z - hit_z - offset` (height above each hit). On flat ground the ground-facing rays
        return a smooth, near-uniform ring -> small std; on stairs/rocks the returns scatter ->
        large std. We take the std over finite, near-ground returns (ignore max-range misses)."""
        extero = obs.get("extero")
        if extero is None:
            return 0.0
        if isinstance(extero, dict):
            extero = extero.get("lidar_scan")
            if extero is None:
                return 0.0
        s = extero.to(self.device, torch.float32).view(-1)[:self.NUM_LIDAR]
        s = torch.nan_to_num(s, nan=self.LIDAR_MAXDIST, posinf=self.LIDAR_MAXDIST,
                             neginf=self.LIDAR_MAXDIST)
        # keep only ground-ish returns (drop max-range misses / far walls that dominate the std)
        m = s.abs() < (self.LIDAR_MAXDIST * 0.5)
        g = s[m]
        if g.numel() < 16:
            return 0.0
        return float(g.std())

    def _build_proprio77(self, obs) -> torch.Tensor:
        proprio = obs["proprio"].to(self.device)[0]              # (72,)
        self._meas_vx = float(proprio[0])                        # base-frame forward speed (m/s)
        base_ang_vel = proprio[3:6]
        gravity = proprio[9:12]
        # roll/pitch: prefer driver-injected TRUE euler (imu obs = wrap_to_pi(roll/pitch) from
        # euler_xyz_from_quat at train time). Fallback gravity form uses atan2(-g_y,-g_z) so an
        # upright robot reads roll=0 -- the old atan2(g_y,g_z) gave roll=pi (looks flipped) and
        # made the policy sink/twitch in place (same bug fixed in the teacher adapter).
        if getattr(self, "_live_roll", None) is not None:
            roll = float(self._live_roll)
            pitch = float(self._live_pitch)
        else:
            roll = math.atan2(-float(gravity[1]), -float(gravity[2]))
            pitch = math.atan2(float(gravity[0]),
                               math.sqrt(float(gravity[1])**2 + float(gravity[2])**2) + 1e-6)
        # cache the current attitude so _synth_pose (warp encoder) can build the sensor quaternion
        # from the SAME roll/pitch the proprio uses (consistency between the two paths).
        self._cur_roll = float(roll)
        self._cur_pitch = float(pitch)
        # Task A proprio joints are in per-leg order; the student expects articulation order.
        jpos = proprio[12:12 + self.NUM_JOINTS][self.stu_from_taska]
        jvel = proprio[12 + self.NUM_JOINTS:12 + 2 * self.NUM_JOINTS][self.stu_from_taska]

        # delta_yaw = heading error to steer by. Two modes:
        #   STRAIGHT (default): just keep the SPAWN heading -- goal_yaw = spawn_yaw. No lateral
        #     position feedback, so no "veer out then correct back" oscillation. Best for the
        #     straight Task A course: hold your initial forward heading and walk.
        #   look-ahead: aim at a point ahead ON the spawn line -> pulls a lateral drift back, but
        #     the position feedback makes it wobble. Use only when staying exactly on a line matters.
        # HIGHEST PRIORITY -- DIRECT delta_yaw injection (real-world guide steering). The driver
        # computes the heading correction PURELY from the onboard LiDAR (rail-relative angle) and
        # hands it here as a relative steer, so NO world position and NO ground-truth yaw are used
        # to decide direction. This is the deployable path: set self._live_delta_yaw each step.
        _inj = getattr(self, "_live_delta_yaw", None)
        if _inj is not None:
            delta_yaw = math.atan2(math.sin(float(_inj)), math.cos(float(_inj)))
        elif self._live_yaw is not None:
            cur_yaw = float(self._live_yaw)
            if self._spawn_yaw is None:
                self._spawn_yaw = cur_yaw
                if self._live_pos is not None:
                    self._spawn_pos = (float(self._live_pos[0]), float(self._live_pos[1]))
            gp = getattr(self, "_live_goal_point", None)
            fg = getattr(self, "_final_goal", None)
            if fg is not None and self._live_pos is not None:
                # FIXED FINAL GOAL (world xy of the course finish). Steer on the straight bearing
                # from the current position to that single fixed point -- no curve tracking. The
                # course curvature is mild enough that a straight line to the goal completes it,
                # and a fixed goal (vs a moving look-ahead) removes the tracking lag that made the
                # robot chronically off-heading and its gait asymmetric. Set via self._final_goal.
                px, py = float(self._live_pos[0]), float(self._live_pos[1])
                goal_yaw = math.atan2(float(fg[1]) - py, float(fg[0]) - px)
            elif gp is not None and self._live_pos is not None:
                # DRIVER-injected look-ahead goal point (world xy) -- CURVED-course pure-pursuit.
                px, py = float(self._live_pos[0]), float(self._live_pos[1])
                goal_yaw = math.atan2(float(gp[1]) - py, float(gp[0]) - px)
            elif self.STRAIGHT_MODE:
                goal_yaw = self.GOAL_YAW                 # FIXED world heading (+x) -> always straight
            elif self._spawn_pos is None or self._live_pos is None:
                goal_yaw = self._spawn_yaw
            else:
                px, py = float(self._live_pos[0]), float(self._live_pos[1])
                cx, cy = self._spawn_pos
                cyaw, syaw = math.cos(self._spawn_yaw), math.sin(self._spawn_yaw)
                s_along = (px - cx) * cyaw + (py - cy) * syaw
                gx = cx + (s_along + self.LOOKAHEAD) * cyaw
                gy = cy + (s_along + self.LOOKAHEAD) * syaw
                goal_yaw = math.atan2(gy - py, gx - px)
            dy = goal_yaw - cur_yaw
            delta_yaw = math.atan2(math.sin(dy), math.cos(dy))
        else:
            delta_yaw = -self.ryaw
        # DIAGNOSTIC: force delta_yaw=0 (no steering, walk straight) to test whether the rough-gait
        # asymmetry is CAUSED by the sustained recovery-steer (if it vanishes -> steering; if it
        # persists -> a policy/obs bias independent of turning).
        if int(os.environ.get("ZERO_DYAW", "0")):
            delta_yaw = 0.0
        self._dbg_dyaw = float(delta_yaw)
        p = torch.zeros(self.NUM_PROP, device=self.device, dtype=torch.float32)
        p[0:3] = base_ang_vel * 0.25
        p[3] = roll
        p[4] = pitch
        p[5] = 0.0
        p[6] = delta_yaw
        p[7] = delta_yaw                    # delta_next_yaw ~ delta_yaw
        p[8] = 0.0
        p[9] = 0.0
        # SPEED CLAMP (symptomatic): the policy walks ~2x its vx command (reward_tracking_goal_vel
        # doesn't penalize overshoot -- true of the teacher too), so a fixed vx=0.5 gives ~1.0 m/s
        # and the robot lunges forward. Close a slow P-loop on the MEASURED forward speed: shrink
        # the vx command when we're over target, grow it (up to CROSS_VX) when under, so realized
        # speed holds near TARGET_SPEED. Toggle with SPEED_CLAMP=0.
        if int(os.environ.get("SPEED_CLAMP", "0")):   # default OFF: just send a constant vx command
            _target = float(os.environ.get("TARGET_SPEED", "0.5"))
            _ki = float(os.environ.get("SPEED_KI", "0.05"))   # integral gain (removes steady bias)
            _kp = float(os.environ.get("SPEED_KP", "0.30"))   # proportional gain (kills fast surges)
            _meas = getattr(self, "_meas_vx", _target)
            _err = _meas - _target
            # integral state: slowly track the command that holds target speed.
            _base = getattr(self, "_vx_cmd", self.CROSS_VX)
            _base = _base - _ki * _err
            _floor = float(os.environ.get("VX_FLOOR", "0.30"))   # teacher trained on cmd 0.3-0.8; don't go below
            _base = max(_floor, min(self.CROSS_VX, _base))
            self._vx_cmd = _base
            # proportional term reacts THIS step to an over-/under-speed surge (e.g. terrain change),
            # so a sudden 1.2 m/s spike is cut immediately instead of waiting for the integral.
            _cmd = _base - _kp * _err
            _cmd = max(_floor, min(self.CROSS_VX, _cmd))       # band: [VX_FLOOR, CROSS_VX]
            p[10] = _cmd
        else:
            p[10] = self.CROSS_VX           # vx command
        self._dbg_p10 = float(p[10])        # the ACTUAL vx command written to the obs slot
        # env_idx: classify terrain type from the LiDAR (training used a dynamic flat/rough bit).
        # Rough ground makes the downward lidar hits scatter in height (large std); a flat floor is
        # near-uniform. This lets the student switch to its rough-terrain gait ON rough ground and
        # its flat gait on flat -- exactly what it was trained to do -- instead of a forced flat bit.
        self._dbg_rough = -1.0
        if self.ENV_MODE == "flat":
            is_flat = True
        elif self.ENV_MODE == "rough":
            is_flat = False
        else:  # auto: infer from lidar height scatter
            self._dbg_rough = self._lidar_rough_std(obs)
            is_flat = (self._dbg_rough < self.ENV_ROUGH_STD)
        self._is_flat = is_flat
        p[11] = 0.0 if is_flat else 1.0     # env_idx
        p[12] = 1.0 if is_flat else 0.0     # invert_env_idx
        p[13:33] = jpos
        p[33:53] = jvel * 0.05
        p[53:73] = self._last_action        # env order (20)
        p[73:77] = 0.0                      # contact_fill unknown
        return p

    def _student_action(self, obs) -> torch.Tensor:
        prop = self._build_proprio77(obs)                        # (77,)

        # maintain the 10-frame history (idx6:8 zeroed for the hist encoder, per training)
        prop_hist = prop.clone()
        prop_hist[6:8] = 0.0
        if self._hist is None:
            self._hist = prop_hist.view(1, 1, -1).repeat(1, self.NUM_HIST, 1)
        else:
            self._hist = torch.cat([self._hist[:, 1:], prop_hist.view(1, 1, -1)], dim=1)

        # refresh the scandots latent + yaw from the lidar every DEPTH_UPDATE_EVERY steps
        if self._ts % self.DEPTH_UPDATE_EVERY == 0:
            lidar = self._lidar(obs)                             # (1, 5760)
            prop_in = prop.view(1, -1).clone()
            prop_in[:, 6:8] = 0.0
            # WARP encoder: append the synthesized 11-D world pose so the encoder's RecurrentDepthBackbone
            # splits it back out and feeds the warp module (proprio77 -> 88-D). Reset the encoder's
            # rolling world heightmap once at episode start (batch=1) so no stale terrain drifts in.
            if self.WARP_ENCODER:
                if self._warp_reset_pending:
                    for _m in ("reset", "reset_warp_state"):
                        _fn = getattr(self.depth_encoder, _m, None)
                        if _fn is not None:
                            try:
                                _fn(1, self.device)
                            except Exception:
                                pass
                    self._warp_reset_pending = False
                prop_in = torch.cat([prop_in, self._synth_pose()], dim=1)   # (1, 88)
            with torch.inference_mode():
                dly = self.depth_encoder(lidar, prop_in)         # (1, 34)
            self._scandots_latent = dly[:, :-2]
            self._yaw = dly[:, -2:]

            # FLAT-LATENT CLAMP: on flat ground the lidar->latent still carries small ray noise
            # that the actor reads as a faint up-slope -> it pitches the body forward (a student-
            # only artifact; the privileged-scan teacher stays level on flat). env_idx already
            # classifies flat vs rough robustly from the lidar height-std. So on FLAT ground we
            # (a) learn a stable "flat latent" as an EMA of the flat-ground latents, and (b) feed
            # that clamped latent to the actor instead of the noisy per-step one. On ROUGH ground
            # we always use the live latent (the terrain detail matters there). Toggle with
            # FLAT_LATENT_CLAMP=0.
            if int(os.environ.get("FLAT_LATENT_CLAMP", "1")):
                is_flat_now = getattr(self, "_is_flat", True)
                if is_flat_now:
                    if self._flat_latent is None:
                        self._flat_latent = self._scandots_latent.clone()
                    else:
                        _a = 0.05   # EMA rate: slowly track the flat-ground latent
                        self._flat_latent = (1 - _a) * self._flat_latent + _a * self._scandots_latent
                    self._scandots_latent = self._flat_latent.clone()

            # LATENT DIAGNOSTIC: compare the student lidar latent against the TEACHER latent
            # (real level 132-pt scandots -> actor.scan_encoder). Splits the crab-walk cause:
            #   large error  -> encoder REPRODUCES the level latent BADLY (encoder problem)
            #   small error  -> latent is faithful but actor still crab-walks (actor/gait problem)
            # Needs the driver to inject self._live_scandots (raycast true grid). LATENT_DIAG=1.
            if int(os.environ.get("LATENT_DIAG", "0")) and self._live_scandots is not None:
                try:
                    _sc = self._live_scandots.to(self.device, torch.float32).view(1, -1)[:, :self.NUM_SCAN]
                    with torch.inference_mode():
                        _teacher_lat = self.policy.actor.scan_encoder(_sc)   # (1,32) level-scandot latent
                    _stu = self._scandots_latent
                    _err = (_stu - _teacher_lat)
                    _mse = float((_err ** 2).mean())
                    _cos = float(torch.nn.functional.cosine_similarity(_stu, _teacher_lat, dim=1))
                    # left/right split: scan_encoder is a per-point MLP over the 132-grid, but the
                    # 32-D latent has no spatial axis; instead report the teacher vs student norms +
                    # the signed mean error (a consistent sign = a systematic bias direction).
                    self._dbg_latmse = _mse
                    self._dbg_latcos = _cos
                    self._dbg_latnorm = (float(_stu.norm()), float(_teacher_lat.norm()))
                    self._dbg_latbias = float(_err.mean())
                    if getattr(self, "_ts", 0) % 25 == 0:
                        print(f"[LATDIAG] mse={_mse:.4f} cos={_cos:+.3f} "
                              f"stu_norm={float(_stu.norm()):.2f} tea_norm={float(_teacher_lat.norm()):.2f} "
                              f"meanbias={float(_err.mean()):+.4f}", flush=True)
                except Exception as _e:
                    if getattr(self, "_ts", 0) % 100 == 0:
                        print(f"[LATDIAG] fail: {type(_e).__name__}: {str(_e)[:120]}", flush=True)

        # assemble the 1033 obs; scan/priv are zeros (latent + hist paths carry info)
        obs_full = torch.zeros(1, self.OBS_DIM, device=self.device, dtype=torch.float32)
        prop2 = prop.view(1, -1).clone()
        # HEADING (crux): training overwrites obs[6:8] with the depth_encoder's yaw prediction ONLY
        # when delta_yaw_ok (|delta_yaw| < 0.6) -- i.e. when the robot is ALREADY roughly aligned,
        # for fine yaw. When NOT aligned, it KEEPS the real delta_yaw so the policy makes a big turn.
        # Blindly writing 1.5*self._yaw every step (the old code) discarded our look-ahead delta_yaw
        # and fed a goal-yaw the deploy env has no goals for -> the student veered ~90deg. Reproduce
        # the gate: keep our delta_yaw (already in prop[6:8]) unless it's small, then use pred yaw.
        # We DRIVE the heading ourselves (fixed +x in STRAIGHT_MODE, or the injected curve goal in
        # look-ahead mode), so by default we NEVER let the depth_encoder's yaw prediction overwrite
        # our delta_yaw: the deploy env has no training goals, so that prediction is unreliable and
        # veers the robot off course. Set USE_DEPTH_YAW=1 to restore the trained delta_yaw_ok gate.
        if self.USE_DEPTH_YAW:
            _dyaw_mag = abs(getattr(self, "_dbg_dyaw", 1.0))
            if _dyaw_mag < self.DELTA_YAW_OK_THRESH:
                prop2[:, 6:8] = 1.5 * self._yaw                  # aligned -> fine yaw from lidar
        obs_full[:, :self.NUM_PROP] = prop2
        hist_flat = self._hist.reshape(1, -1)
        obs_full[:, -self.NUM_HIST * self.NUM_PROP:] = hist_flat

        # DIAGNOSTIC: zero the scandots latent to test whether a LEFT-RIGHT asymmetry in the
        # lidar->latent (the student-only signal; the teacher gets a symmetric privileged scan) is
        # what makes the student veer left. If ZERO_LATENT removes the left drift -> latent is the
        # culprit; if it persists -> the bias is in the policy/proprio path.
        _lat_in = self._scandots_latent
        if int(os.environ.get("ZERO_LATENT", "0")):
            _lat_in = torch.zeros_like(_lat_in)
        with torch.inference_mode():
            if self.TEACHER_SCANDOTS:
                # RAW teacher path: write the TRUE 132-pt scandots into obs[77:209] and let the
                # teacher's own scan_encoder produce the latent (policy_with_est.pt: obs-only forward,
                # estimator fills priv_explicit internally). NO lidar depth_encoder, NO supplied latent.
                if self._live_scandots is not None:
                    _sc = self._live_scandots.to(self.device, torch.float32).view(-1)[:self.NUM_SCAN]
                    obs_full[:, self.NUM_PROP:self.NUM_PROP + self.NUM_SCAN] = _sc.view(1, -1)
                act_raw = self.teacher_policy(obs_full)  # (1, 20)
            else:
                act_raw = self.policy(obs_full, _lat_in)  # (1, 20)
        if not isinstance(act_raw, torch.Tensor):
            act_raw = torch.as_tensor(act_raw, device=self.device)
        act_raw = act_raw.to(self.device, torch.float32).view(-1)   # (20,)

        # Store last_action in the STUDENT (articulation) basis — that is what the student saw
        # of its own prior output at train time, and _build_proprio77 feeds it back unpermuted.
        # We OVERRIDE the arm to 0, so report 0 for the arm slots (interleaved: [8,13..19]) instead
        # of the policy's arm output, keeping proprio self-consistent (as in the teacher adapter).
        self._last_action = act_raw.clone()
        self._last_action[self.arm_stu_idx] = 0.0

        # Permute the student's action (articulation order) into Task A per-leg order, THEN
        # rescale the legs and overwrite the arm with the stow target. Doing the permute first
        # is essential: student index 0..11 is NOT the 12 legs (it's hip*4/thigh*4/joint1/calf*3),
        # so the old identity slice scrambled the legs.
        act_taska = act_raw[self.taska_from_stu]                      # (20,) Task A order
        action_env = torch.zeros(self.NUM_JOINTS, device=self.device, dtype=torch.float32)
        action_env[self.leg_idx] = (act_taska[self.leg_idx] * self.OUT_RESCALE).clamp(-9.6, 9.6)
        action_env[self.arm_idx] = self.arm_action                    # ignore policy arm output
        return action_env.view(1, -1)

    def predicts(self, obs, current_score):
        self._integrate_yaw(obs)
        if self.WARP_ENCODER:
            self._integrate_odom(obs)          # dead-reckon world (x,y) for the warp pose
        action = self._student_action(obs)
        self._ts += 1
        if os.environ.get("ACT_DEBUG") and self._ts % 25 == 0:
            _ly = f"{self._live_yaw:+.2f}" if self._live_yaw is not None else "None"
            _dyaw = getattr(self, "_dbg_dyaw", float('nan'))
            _flat = getattr(self, "_is_flat", None)
            _pit = f"{self._live_pitch:+.2f}" if self._live_pitch is not None else "None"
            # Task A per-leg action order: calf slots FR=2 FL=5 RR=8 RL=11 (calf ~ how high the
            # foot lifts). Diagonal pairs: (FL,RR) vs (FR,RL). Report all four to see asymmetry.
            _a = action.view(-1)
            _feet = (f"FRc={float(_a[2]):+.2f} FLc={float(_a[5]):+.2f} "
                     f"RRc={float(_a[8]):+.2f} RLc={float(_a[11]):+.2f}")
            print(f"[TASKA-LIDAR t={self._ts:4d}] live_yaw={_ly} pitch={_pit} dyaw={_dyaw:+.3f} "
                  f"p10={float(getattr(self,'_dbg_p10',-1)):.2f} meas={float(getattr(self,'_meas_vx',-1)):.2f} "
                  f"flat={_flat} rough={getattr(self,'_dbg_rough',-1):.3f} "
                  f"lat|={float(self._scandots_latent.norm()):.2f} {_feet}", flush=True)
        return {'action': action.cpu().numpy().tolist(), 'giveup': False}
