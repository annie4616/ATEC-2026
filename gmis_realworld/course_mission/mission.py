"""Mission library: perception, locomotion, arm control and the pick-classify-deposit FSM.

Everything the solution scripts need lives here so the three solution variants stay thin
and differ only in policy (which terrain to attempt, how many retries, how many picks).

Checkpoint reuse (all previously trained in this repo, nothing retrained here):
  walk      from_eunha/speed_{1.0,0.75}/model_14999.pt + adaptation.pt  (53-obs RMA)
  crouch    eunha_rl/<sp>/crouch_runs/crouch<depth>_*/model_2999.pt     (53-obs RMA)
  grasp     eunha_rl/detached/champ_eunha3_<sp>_<depth>.txt -> model_*.pt (42-obs, 7-act)
  deposit   eunha_rl/deposit_traj.json      (basket drop, solved offline)
  retrieve  course_mission/retrieve_traj.json (basket -> bin, solved offline)

Frames: "arc" coordinates are (x along the S centerline, w lateral); curve_map converts
to world. Bins sit at arc x = SECTION_CENTERS['bins'], laterals from
terrain.BIN_LATERAL_OF (blue +0.9 / green 0.0 / red -0.9, matching the venue photo).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    import isaaclab.utils.math as math_utils
except ModuleNotFoundError:            # hardware build: see course_mission/mathx.py
    from course_mission import mathx as math_utils
# The IK controller is pure torch and portable, but it lives inside isaaclab; the arm is
# simulator-bound anyway until an analytic Jacobian replaces root_physx_view (DEPLOY WP2),
# so this one stays hard until then.
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg

from course_sim import terrain as ct
from course_mission.catalogue import (
    BIN_OF, OBJ_DIMS, OBJ_RGB, TERRAIN_RANK, TOP_ABOVE_ORIGIN, grasp_hint, grasp_obb,
)
from course_mission.deploy import (  # noqa: F401  (COMPUTE* re-exported for the runner)
    COMPUTE, COMPUTE_LIMITS, GRASP_NOISE, LEG_NOISE, PN_BIAS, PN_DIM_REL, PN_HOLD,
    PN_JITTER, REAL_NOISE, _u,
)
from course_mission.perception import Perception  # noqa: F401  (re-exported)

RW = Path(__file__).resolve().parents[1]

LEG_ORDER = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
ARM_JOINTS = [f"arm_joint{i}" for i in range(1, 7)]
# Opt-in caps on the per-step IK command; see ArmDriver.ik_step. 0 = off (default).
ROT_CAP = float(os.environ.get("MISSION_GRASP_ROT_CAP", "0"))
POS_CAP = float(os.environ.get("MISSION_GRASP_POS_CAP", "0"))
# DEPLOYMENT REALISM, all off by default so the 120-scoring path stays byte-identical.
#   MISSION_ARM_RATE   rad per 20 ms step on every arm joint target. The simulator caps
#     arm speed with velocity_limit_sim=10 rad/s (env_cfg.py:248-250) — an ACTUATOR clamp
#     that exists only inside Isaac. On hardware nothing bounds the command, so the joint
#     target steps far ahead of the hand and the driver protection-faults (env_cfg.py:235).
#     10 rad/s * 0.02 s = 0.2 is the equivalent software limit.
#   MISSION_GRIP_RATE  metres per step on the gripper target (env_cfg.py:254 caps it at
#     1.0 in sim, i.e. 0.02 per step).
#   MISSION_ARM_DELAY  control steps of command->motion latency for arm AND gripper. The
#     leg queue in runner._step models this for legs only, so every grasp number we have
#     was measured with a zero-latency arm.
ARM_RATE = float(os.environ.get("MISSION_ARM_RATE", "0"))
#   MISSION_ARM_RATE_SLEW  1 = rate-limit the COMMAND (a real driver's slew limit),
#     0 = rate-limit the command's distance from the MEASURED position (what the code
#     did by accident, which is a stiffness*ARM_RATE torque ceiling). See _push_arm.
ARM_RATE_SLEW = os.environ.get("MISSION_ARM_RATE_SLEW", "0") == "1"
GRIP_RATE = float(os.environ.get("MISSION_GRIP_RATE", "0"))
#   MISSION_GRIP_VEL_GUARD  finger over-speed protection, m/s (0 = off). env_cfg.py:259
#     rates arm_joint7/8 at 1.0 m/s and PhysX carries that as maxJointVelocity, but the
#     articulation solver only holds it per substep: the audit measured 3.5-6.5 m/s in
#     grasp closes. See write_gripper().
GRIP_VEL_GUARD = float(os.environ.get("MISSION_GRIP_VEL_GUARD", "1.0"))
ARM_DELAY = int(os.environ.get("MISSION_ARM_DELAY", "0"))
#   MISSION_GRASP_FREEZE  hold the grasp target at its first value for the whole attempt
#     (see GraspObsBuilder.build) instead of re-noising the object's live true pose at
#     10 Hz — the rate no real sensor can supply once the object is under the chin.
GRASP_FREEZE = os.environ.get("MISSION_GRASP_FREEZE") == "1"
#   MISSION_IK_FRAME_FIX  correct the TCP-offset frame in ArmDriver._jacobian (see there).
IK_FRAME_FIX = os.environ.get("MISSION_IK_FRAME_FIX") == "1"
#   MISSION_GRASP_HOLD  control steps between target refreshes, overriding deploy.PN_HOLD
#     (5 = the 10 Hz the champions were evaluated at). FREEZE is the limit of this knob;
#     sweeping it answers the question FREEZE can only answer yes/no -- how fresh does a
#     close-range sensor have to be before the grasp works, i.e. how good a wrist camera
#     would have to be to be worth mounting.
GRASP_HOLD = int(os.environ.get("MISSION_GRASP_HOLD", "0")) or PN_HOLD
#   MISSION_GRASP_NOYAW  feed the grasp policy a YAW-FREE base state. The stage-2
#     checkpoints trained with reset_root_state_uniform(pose_range={}) on an identity
#     init rot: every training episode had root_quat_w = (1, 0, 0, 0) + crouch tilt and
#     body_lin_acc_w in a frame aligned with the body's yaw. On the course the robot
#     faces -3..-33 deg along the S-curve and up to +-150 deg when it walks in around a
#     crowd, so the raw world quaternion hands the policy a (w, z) pair it never saw.
#     Strip the yaw (keep roll/pitch) and rotate the acceleration into the yaw-free
#     frame, which is exactly what training produced.
GRASP_NOYAW = os.environ.get("MISSION_GRASP_NOYAW", "0") == "1"
HIST_LEN, FEAT_DIM, LATENT_DIM = 50, 45, 8
ACTION_SCALE = {"hip": 0.125, "other": 0.25}
ENV_LEG_SCALE = 0.5                       # task_b joint_leg scale
TCP_OFFSET = float(os.environ.get("GRASP_TCP_OFFSET", "0.135"))
GRIP_OPEN = float(os.environ.get("GRASP_GRIP_OPEN", "0.05"))
GRIP_DEPTH = float(os.environ.get("GRASP_GRIP_DEPTH", "0.015"))


def arc_of(X: float, Y: float) -> tuple[float, float]:
    """World (X, Y) -> (arc x, lateral w) by 2-stage search along the centerline."""
    xs = np.linspace(ct.COURSE_X0 - 1.0, ct.COURSE_X0 + ct.COURSE_LEN + 1.0, 400)
    for _ in range(2):
        cx, cy = ct.curve_map(xs, 0.0)
        d = (cx - X) ** 2 + (cy - Y) ** 2
        i = int(np.argmin(d))
        lo = xs[max(i - 1, 0)]
        hi = xs[min(i + 1, len(xs) - 1)]
        xs = np.linspace(lo, hi, 200)
    x = float(xs[len(xs) // 2])
    th = float(ct.curve_theta(x))
    # lateral offset: project the residual onto the course normal
    cx, cy = ct.curve_map(x, 0.0)
    w = float(-(X - cx) * np.sin(th) + (Y - cy) * np.cos(th))
    return x, w


def section_of(arc_x: float) -> str:
    i = int((arc_x - ct.COURSE_X0) // ct.SECTION_LEN)
    i = max(0, min(len(ct.COURSE_SEQUENCE) - 1, i))
    return ct.COURSE_SEQUENCE[i]


def bin_targets() -> dict:
    """{'red'|'green'|'blue': (arc_x, lateral)} for the three competition bins."""
    cx = ct.SECTION_CENTERS["bins"]
    # read from terrain, never restated: the two used to disagree the moment the
    # physical row was re-ordered, and the failure is a silent wrong-bin delivery
    return {c: (cx, lat) for c, lat in ct.BIN_LATERAL_OF.items()}


# ======================================================================================
# locomotion: one RMA adaptation net + two frozen actors (walk / crouch)
# ======================================================================================
def _load_actor(path: str, device: str) -> nn.Module:
    net = nn.Sequential(
        nn.Linear(FEAT_DIM + LATENT_DIM, 512), nn.ELU(),
        nn.Linear(512, 256), nn.ELU(),
        nn.Linear(256, 128), nn.ELU(),
        nn.Linear(128, 12),
    )
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
    net.load_state_dict({k.replace("actor.", ""): v for k, v in sd.items() if k.startswith("actor.")})
    return net.to(device).eval().requires_grad_(False)


def _load_adaptation(path: str, device: str) -> nn.Module:
    act = nn.ReLU if os.environ.get("EUNHA_ADAPT_ACT", "elu").lower() == "relu" else nn.ELU

    class _Adapt(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Sequential(nn.Linear(FEAT_DIM, 32), act())
            self.conv = nn.Sequential(
                nn.Conv1d(32, 32, 8, 4), act(),
                nn.Conv1d(32, 32, 5, 1), act(),
                nn.Conv1d(32, 32, 5, 1), act(),
            )
            self.head = nn.Linear(96, LATENT_DIM)

        def forward(self, h):
            z = self.embed(h)
            return self.head(self.conv(z.transpose(1, 2)).flatten(1))

    net = _Adapt()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["adaptation_state_dict"] if isinstance(ck, dict) and "adaptation_state_dict" in ck else ck
    miss, unexp = net.load_state_dict(sd, strict=False)
    assert not miss and not unexp, f"adaptation mismatch {miss} {unexp}"
    return net.to(device).eval().requires_grad_(False)


class HeadCamMount:
    """Owns the head camera's pose: writes it to the sim AND is the source of truth for it.

    Two separate faults had to be fixed here, both measured (2026-08-16):

    1. A camera prim parented at Robot/base_link/head_cam never follows the body. With GPU
       physics the articulation's transforms live in Fabric and are never written back to
       USD, so the child prim keeps its spawn transform. Over one 250-frame walk the
       camera's offset in the base frame went +0.35 m (correct) -> -1.98 -> -39 m: it
       filmed the start of the course while the robot walked away. Hence a root-level prim
       (launch.head_cam_cfg) whose pose we write ourselves.

    2. Writing the pose works, but `Camera.data.pos_w` does not reflect it — probed
       directly, `cam._view.get_world_poses()` returned the new pose while `cam.data.pos_w`
       stayed [0, 0, 0] both immediately and after a step. The rendered image is correct;
       only the sensor's reported pose is dead. Since the detector unprojects with that
       pose, it was unprojecting every image from the world origin.

    So the commanded pose is cached here and handed to Perception directly. That is also
    what the real robot does: camera extrinsics come from the known mount plus the base
    state estimate, not from asking the camera where it is.
    """

    # ROS camera axes expressed in the "world" convention IsaacLab uses for offsets
    # (world: forward +X, left +Y, up +Z | ros: right +X, down +Y, forward +Z)
    _ROS_IN_WORLD = np.array([[0.0, 0.0, 1.0],
                              [-1.0, 0.0, 0.0],
                              [0.0, -1.0, 0.0]])

    def __init__(self, cam, robot, device: str, pos, rot):
        self.cam, self.robot, self.device = cam, robot, device
        self.pos = torch.tensor([pos], dtype=torch.float32, device=device)
        self.rot = torch.tensor([rot], dtype=torch.float32, device=device)
        self.pos_w = np.zeros(3, np.float64)
        self.R_ros = np.eye(3)
        self.sync()

    def sync(self):
        d = self.robot.data
        q = math_utils.quat_mul(d.root_quat_w, self.rot)
        p = d.root_pos_w + math_utils.quat_apply(d.root_quat_w, self.pos)
        self.cam.set_world_poses(p.clone(), q.clone(), convention="world")
        self.pos_w = p[0].detach().cpu().numpy().astype(np.float64)
        R_world = math_utils.matrix_from_quat(q)[0].detach().cpu().numpy().astype(np.float64)
        self.R_ros = R_world @ self._ROS_IN_WORLD


class LegDriver:
    """Drives the 12 leg joints with the walk OR crouch RMA policy (same 53-obs interface).

    Both policies write `q_stand + scale*a`; the env's joint_leg term writes
    `default + 0.5*a_env`, and the A2 default pose IS q_stand, so a per-joint ratio
    converts one to the other (verified in course_sim/scripts/walk_course.py).
    """

    def __init__(self, robot, device, walk_ckpt, crouch_ckpt, adapt_ckpt):
        self.robot, self.device = robot, device
        names = list(robot.joint_names)
        self.leg_ids = torch.tensor([names.index(n) for n in LEG_ORDER], device=device)
        self.walk = _load_actor(walk_ckpt, device)
        self.crouch = _load_actor(crouch_ckpt, device)
        self.adapt = _load_adaptation(adapt_ckpt, device)
        self.ratio = torch.tensor(
            [(ACTION_SCALE["hip"] if "_hip_" in n else ACTION_SCALE["other"]) / ENV_LEG_SCALE
             for n in LEG_ORDER], device=device)
        self.hist = torch.zeros(1, HIST_LEN, FEAT_DIM, device=device)
        self.last_a = torch.zeros(1, 12, device=device)
        self.fresh = True
        self.mode = "walk"
        self._blend = 1.0
        self._prev_a = torch.zeros(1, 12, device=device)

    def set_mode(self, mode: str, blend_steps: int = 60):
        """Switch policy. The history is refilled and the output cross-faded so the
        stand<->crouch transition does not step-change the joint targets."""
        if mode == self.mode:
            return
        self.mode = mode
        self.fresh = True
        self._blend = 0.0
        self._blend_step = 1.0 / max(blend_steps, 1)

    def step(self, cmd: torch.Tensor) -> torch.Tensor:
        """cmd = (vx, vy, wz) when walking, (crouch_flag, 0, 0) when crouching."""
        d = self.robot.data
        q = d.joint_pos[:, self.leg_ids] - d.default_joint_pos[:, self.leg_ids]
        qd_raw = d.joint_vel[:, self.leg_ids]
        av, grav = d.root_ang_vel_b, d.projected_gravity_b
        if REAL_NOISE:
            # IMU gyro, gravity estimate, encoders and their differentiated velocity —
            # the same magnitudes the policy was trained under
            av = av + _u(av.shape, LEG_NOISE["ang_vel"], self.device)
            grav = grav + _u(grav.shape, LEG_NOISE["gravity"], self.device)
            q = q + _u(q.shape, LEG_NOISE["joint_pos"], self.device)
            qd_raw = qd_raw + _u(qd_raw.shape, LEG_NOISE["joint_vel"], self.device)
        obs45 = torch.cat([av * 0.25, grav, cmd, q, qd_raw * 0.05,
                           self.last_a], dim=1)
        if self.fresh:
            self.hist[:] = obs45.unsqueeze(1).expand(-1, HIST_LEN, -1)
            self.fresh = False
        self.hist = torch.roll(self.hist, -1, dims=1)
        self.hist[:, -1] = obs45
        latent = self.adapt(self.hist)
        net = self.walk if self.mode == "walk" else self.crouch
        a = net(torch.cat([obs45, latent], dim=1))
        self.last_a = a
        if self._blend < 1.0:
            a = self._prev_a * (1.0 - self._blend) + a * self._blend
            self._blend = min(1.0, self._blend + self._blend_step)
        self._prev_a = a
        return a * self.ratio


# ======================================================================================
# arm: differential IK (matching the grasp action term) + joint-space trajectory player
# ======================================================================================
class ArmDriver:
    """Writes arm joint targets directly (no action term), in two modes.

    ik_step()   reproduces DifferentialInverseKinematicsAction exactly: relative pose
                command * 0.02, TCP body offset, DLS solve on the base-frame Jacobian.
    hold()/goto() play scripted joint waypoints (deposit / retrieve trajectories).
    """

    def __init__(self, robot, device, scale: float = 0.02):
        self.robot, self.device, self.scale = robot, device, scale
        names = list(robot.joint_names)
        self.joint_ids = [names.index(j) for j in ARM_JOINTS]
        self.grip_ids = [names.index("arm_joint7"), names.index("arm_joint8")]
        self.body_idx = robot.body_names.index("gripper_base")
        self.jacobi_body_idx = self.body_idx           # floating base
        self.jacobi_joint_ids = [i + 6 for i in self.joint_ids]
        # DLS DAMPING. The default lambda_val is 0.01, which in this controller means a
        # damping term of 1e-4 — effectively an undamped pseudo-inverse. Near a singular
        # arm configuration J J^T becomes ill-conditioned and dq = J^T (J J^T + lam I)^-1 dx
        # blows up, so a normal-sized pose command turns into a huge joint step and the arm
        # WHIPS. That is the intermittent fast swing (as opposed to a steady vibration):
        # it fires only when the arm passes near a singularity. Raising lambda damps exactly
        # those configurations and leaves well-conditioned ones almost unchanged.
        # Off by default (keeps the 120-scoring behaviour); MISSION_IK_LAMBDA=0.05 to try.
        _ikcfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True,
                                             ik_method="dls")
        # DEFAULT 0.05. Shipped together with the arm speed ceiling and measured with it:
        # 120/160 against the unconstrained control's 80. Damps the near-singular blow-up
        # that turns a normal pose command into a sudden whip.
        _lam = float(os.environ.get("MISSION_IK_LAMBDA", "0.05"))
        if _lam > 0.0:
            _ikcfg.ik_params = {**(_ikcfg.ik_params or {}), "lambda_val": _lam}
        self.ik = DifferentialIKController(
            cfg=_ikcfg,
            num_envs=1, device=device)
        self.offset_pos = torch.tensor([[0.0, 0.0, TCP_OFFSET]], device=device)
        self.offset_rot = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        self.grip_target = torch.tensor([GRIP_OPEN, -GRIP_OPEN], device=device)
        self._arm_pend: list = []          # MISSION_ARM_DELAY command queues
        self._grip_pend: list = []
        self.jv_clipped = 0                # how often the rate limiter actually bit

    # -- frames -------------------------------------------------------------------
    def frame_pose(self):
        d = self.robot.data
        p, q = math_utils.subtract_frame_transforms(
            d.root_pos_w, d.root_quat_w, d.body_pos_w[:, self.body_idx],
            d.body_quat_w[:, self.body_idx])
        return math_utils.combine_frame_transforms(p, q, self.offset_pos, self.offset_rot)

    def _jacobian(self):
        """Base-frame Jacobian at the TCP, from PhysX.

        FRAME BUG, MEASURED. The offset shift below moves the Jacobian from gripper_base
        to the TCP with  J_lin += -[r]x J_ang.  That identity needs r and J_ang in the
        SAME frame, but J_ang has just been rotated into the BASE frame while
        self.offset_pos is (0, 0, 0.135) in the GRIPPER's own frame. Comparing against an
        analytic URDF Jacobian (course_mission/armkin.py, verified against PhysX FK to
        1.4e-3 mm and against its own finite differences to 4e-7) the angular block and the
        FK match exactly while the linear block differs by median 0.136 / max 0.214 --
        and reproducing this exact convention offline gives median 0.137, while rotating r
        into the base frame first gives 0.000. So the discrepancy IS this line.

        It is left ON by default because every scoring run to date used it and the DLS
        solve is forgiving enough to converge on a slightly wrong descent direction (it is
        also a candidate explanation for the arm tremble). MISSION_IK_FRAME_FIX=1 rotates
        r into the base frame, which is the correct transform; A/B it before shipping.
        """
        j = self.robot.root_physx_view.get_jacobians()[:, self.jacobi_body_idx, :, self.jacobi_joint_ids]
        base_rot = math_utils.matrix_from_quat(math_utils.quat_inv(self.robot.data.root_quat_w))
        j[:, :3, :] = torch.bmm(base_rot, j[:, :3, :])
        j[:, 3:, :] = torch.bmm(base_rot, j[:, 3:, :])
        if IK_FRAME_FIX:
            # r in the base frame: R(base<-ee) @ r_local, with R from the same PhysX pose
            ee_q_b = math_utils.quat_mul(math_utils.quat_inv(self.robot.data.root_quat_w),
                                         self.robot.data.body_quat_w[:, self.body_idx])
            r_b = math_utils.quat_apply(ee_q_b, self.offset_pos)
            j[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(r_b), j[:, 3:, :])
        else:
            j[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(self.offset_pos), j[:, 3:, :])
        j[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(self.offset_rot), j[:, 3:, :])
        return j

    # -- commands -----------------------------------------------------------------
    def ik_step(self, action6: torch.Tensor):
        """One relative-pose IK step from a policy action.

        OPTIONAL COMMAND CAPS, both default OFF so the scoring configuration is untouched.
        The deployed grasp checkpoint commands |a_xyz| 3.7-6.2 and |a_rot| 19-28, which
        through `action6 * scale` (scale 0.02) is 75-124 mm and 22-32 deg of pose delta per
        20 ms control step — against the training config's own stated contract of
        "locked: 2 cm/step @ 50 Hz". The arm cannot follow that, so the joint target sits
        permanently far ahead of the hand and is re-aimed 50 times a second: that is the
        visible tremble, and it is NOT sensor noise (measured at ~2% of the command).
        Capping is a train/deploy divergence, though — training ran uncapped and the policy
        may exploit saturation — so it stays opt-in until a probe shows it helps.
            MISSION_GRASP_ROT_CAP  rad/step on the rotation part (e.g. 0.05 = 3 deg)
            MISSION_GRASP_POS_CAP  metres/step on the translation part (e.g. 0.040)
        """
        p, q = self.frame_pose()
        cmd = action6 * self.scale
        if ROT_CAP > 0.0:
            r = cmd[:, 3:6].norm(dim=1, keepdim=True).clamp(min=1e-9)
            cmd = torch.cat([cmd[:, :3], cmd[:, 3:6] * (r.clamp(max=ROT_CAP) / r)], dim=1)
        if POS_CAP > 0.0:
            t = cmd[:, :3].norm(dim=1, keepdim=True).clamp(min=1e-9)
            cmd = torch.cat([cmd[:, :3] * (t.clamp(max=POS_CAP) / t), cmd[:, 3:6]], dim=1)
        self.ik.set_command(cmd, p, q)
        jp = self.robot.data.joint_pos[:, self.joint_ids]
        des = self.ik.compute(p, q, self._jacobian(), jp)
        self._push_arm(des)

    def ik_step_pos(self, dxyz: torch.Tensor, lam: float = 0.05):
        """One POSITION-ONLY DLS step: the tool orientation is left free.

        ik_step() is a relative *pose* solve with a zero rotation delta, i.e. every
        step also asks for "no rotation", so the arm can only reach where the CURRENT
        tool orientation can be carried to. Inside the basket that leaves a dead zone
        at the front-outboard corner: an object at base x >= 0, y <= -0.22 stalled the
        pick servo at a 0.06-0.07 m residual in 4/4 runs (svG/svJ/svP/svS — 22 tries,
        one object never came out) while the kinematic scan says the point itself is
        reachable with a slightly tilted tool. This solve uses the 3 x 6 linear block
        only, so the wrist may tilt to close the last few centimetres.
        dxyz: base-frame translation for this step (metres, shape (1, 3)).
        """
        j = self._jacobian()[:, 0:3, :]                         # (1, 3, 6)
        jt = j.transpose(1, 2)
        eye = torch.eye(3, device=self.device).unsqueeze(0)
        dq = jt @ torch.linalg.solve(j @ jt + (lam ** 2) * eye, dxyz.reshape(1, 3, 1))
        jp = self.robot.data.joint_pos[:, self.joint_ids]
        self._push_arm(jp + dq.reshape(1, 6))

    def _push_arm(self, des: torch.Tensor):
        """THE single write point for the six arm joint targets.

        Everything that moves the arm funnels through here — ik_step (the RL grasp loop
        and every servo) and set_joints (the scripted trajectories) — so the deployment
        rate limit and the command latency have exactly one place to live.
        """
        des = des.reshape(1, 6)
        if ARM_RATE > 0.0:
            # ANCHOR: measured position (default) or the last COMMAND (slew mode).
            # These are not the same constraint. Anchoring on the measured position
            # caps the PD error at ARM_RATE, and with stiffness 80 that is a hard
            # 80 * 0.2 = 16 Nm torque ceiling on a 100 Nm joint — the joint stalls
            # wherever 16 Nm is not enough and can never build more error, because the
            # clamp re-anchors every step. Measured on probe bagB: play_traj commanded
            # [-0.31, 2.42, -1.46, 0.70, 0.01, 2.78] and 170 steps later the arm sat at
            # [-0.42, 2.37, -0.72, 1.06, 0.67, 2.92], TCP 0.38 m short of a pose armkin
            # confirms is legal, unobstructed and collision-free. Real drivers limit the
            # slew rate of the COMMAND, which is what MISSION_ARM_RATE_SLEW=1 does;
            # default 0 keeps the behaviour every scored run was measured with.
            if ARM_RATE_SLEW:
                base = getattr(self, "_arm_cmd", None)
                if base is None:
                    base = self.robot.data.joint_pos[:, self.joint_ids].clone()
            else:
                base = self.robot.data.joint_pos[:, self.joint_ids]
            step = des - base
            if bool((step.abs() > ARM_RATE).any()):
                self.jv_clipped += 1
            des = base + step.clamp(-ARM_RATE, ARM_RATE)
            if ARM_RATE_SLEW:
                self._arm_cmd = des.clone()
        if ARM_DELAY > 0:
            self._arm_pend.append(des.clone())
            des = (self._arm_pend.pop(0) if len(self._arm_pend) > ARM_DELAY
                   else self.robot.data.joint_pos[:, self.joint_ids].clone())
        self.robot.set_joint_position_target(des, self.joint_ids)

    def set_joints(self, q6: torch.Tensor):
        self._push_arm(q6)

    def gripper_width(self):
        """Live finger opening, q(arm_joint7) - q(arm_joint8), shape (1, 1).

        The same quantity the observation vector carries, exposed so a grasp can be judged
        on whether anything was actually BETWEEN the fingers. It was computed every step
        and discarded, which left every thin-feature argument in this codebase resting on
        arithmetic instead of a measurement.
        """
        d = self.robot.data
        names = self.robot.joint_names
        j7, j8 = names.index("arm_joint7"), names.index("arm_joint8")
        return (d.joint_pos[:, j7] - d.joint_pos[:, j8]).unsqueeze(1)

    def set_gripper(self, open_: bool):
        self.grip_target = torch.tensor(
            [GRIP_OPEN, -GRIP_OPEN] if open_ else [0.0, 0.0], device=self.device)

    def write_gripper(self):
        tgt = self.grip_target.reshape(1, 2)
        if GRIP_RATE > 0.0:
            now = self.robot.data.joint_pos[:, self.grip_ids]
            tgt = now + (tgt - now).clamp(-GRIP_RATE, GRIP_RATE)
        if GRIP_VEL_GUARD > 0.0:
            # HARDWARE-STYLE OVER-SPEED PROTECTION on arm_joint7/8, rated 1.0 m/s.
            # The torque audit caught them at 3.5-6.5 m/s, always inside a grasp
            # close: the position ramp commands the rating itself and a contact
            # release then snaps the fingers past it. A real driver answers an
            # over-speed by parking the target on the joint's CURRENT position,
            # which removes the driving term for that step instead of adding to it;
            # the ramp picks up again as soon as the joint is back inside its rating.
            # Squeezing is unaffected — a finger loaded against an object is slow.
            _v = self.robot.data.joint_vel[:, self.grip_ids]
            _p = self.robot.data.joint_pos[:, self.grip_ids]
            tgt = torch.where(_v.abs() > GRIP_VEL_GUARD, _p, tgt)
        if ARM_DELAY > 0:
            self._grip_pend.append(tgt.clone())
            tgt = (self._grip_pend.pop(0) if len(self._grip_pend) > ARM_DELAY
                   else self.robot.data.joint_pos[:, self.grip_ids].clone())
        self.robot.set_joint_position_target(tgt, self.grip_ids)

    def grip_effort(self):
        """Peak |applied torque| on the two finger joints, or None if unavailable.

        The real grasp verdict cannot read object truth; the portable signals are finger
        WIDTH and finger EFFORT. Width was already exposed (gripper_width); this is the
        other half, and it exists in sim too, so the width/effort decision bands can be
        fitted here before the PiPER-X bench pass.
        """
        t = getattr(self.robot.data, "applied_torque", None)
        if t is None:
            return None
        return float(t[0, self.grip_ids].abs().max())

    def current_q6(self) -> torch.Tensor:
        return self.robot.data.joint_pos[0, self.joint_ids].clone()


# ======================================================================================
# grasp policy: 42-d obs built by hand (the course env's managers cannot supply it)
# ======================================================================================
class GraspPolicy:
    """Champion stage-2 policy: 42 obs -> MLP[256,128,64] ELU -> 7 actions."""

    def __init__(self, ckpt: str, device: str):
        net = nn.Sequential(
            nn.Linear(42, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, 7),
        )
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
        net.load_state_dict({k.replace("actor.", ""): v for k, v in sd.items()
                             if k.startswith("actor.")})
        self.net = net.to(device).eval().requires_grad_(False)
        self.device = device

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class GraspObsBuilder:
    """Rebuilds the stage-2 observation for an arbitrary target object in the course env.

    Layout (see graspenv/mdp/observations.py):
      [0:7] ee_pose_b | [7:10] grasp_vec_ee | [10:16] approach_axes | [16:17] gripper_width
      [17:24] last_action | [24:29] grasp_params | [29:32] obb_dims | [32:42] base_state_real

    The object-derived terms use the rgbl perception model the champions were trained
    with: a world-frame estimate refreshed at 10 Hz with a per-attempt systematic bias
    (seeded from the CAMERA's real localisation error) plus per-update jitter.
    """

    def __init__(self, robot, arm: ArmDriver, device):
        self.robot, self.arm, self.device = robot, arm, device
        self.base_idx = robot.body_names.index("base_link")
        self.reset_target(np.zeros(3), np.array([0.05, 0.05, 0.05]), np.zeros(3))

    def reset_target(self, true_pos_w, dims, cam_error, cls: str | None = None):
        """New grasp attempt: fix the estimate bias from the measured camera error."""
        self.grip_w, self.grip_depth = grasp_hint(cls or "", np.asarray(dims, float))
        d3 = np.asarray(dims, float)
        self.half_lo, self.half_hi = float(min(d3)) / 2, float(max(d3)) / 2
        # Classes whose rigid-body origin is NOT the centre (see catalogue.TOP_ABOVE_ORIGIN)
        self.top_off = TOP_ABOVE_ORIGIN.get(cls or "")
        # what the fingers actually meet, which is not always the whole object
        shown = grasp_obb(cls or "", np.asarray(dims, float))
        b = np.clip(np.asarray(cam_error, float), -np.array(PN_BIAS), np.array(PN_BIAS))
        self.bias = torch.tensor(b, dtype=torch.float32, device=self.device)
        self.dims = torch.tensor(np.asarray(shown, float), dtype=torch.float32,
                                 device=self.device).reshape(1, 3)
        self.dim_err = 1.0 + (torch.rand(1, 3, device=self.device) * 2 - 1) * PN_DIM_REL
        self.est_w = None
        self.hold = 0
        # OBJECT-HONEST target (MISSION_OBJ_HONEST=1): a callable returning the grasp
        # point in the BASE frame, (1, 3). Set by the runner from the camera detection
        # (held in the navigation frame and re-expressed through the pose estimate each
        # step) and the object's simulator pose is then never read here: the grasp
        # vector is est_b - tcp_b with tcp_b from the arm's own FK. Frozen for the whole
        # attempt — the head camera cannot see the grasp workspace (NEAR_CUT, arm
        # occlusion), so no refresh exists on the robot either.
        self.target_b_fn = None
        self.last_action = torch.zeros(1, 7, device=self.device)
        self.params = torch.tensor([[self.grip_w, 0.0, 0.5, 0.5, 0.5]], device=self.device)

    def _grasp_point_w(self, obj, ground_z: float) -> torch.Tensor:
        """Aim just below the object's top — measured, not assumed.

        Using the catalogue height assumes the object is standing. A paper bag is listed
        27 cm tall but comes to rest on its side, and the nominal number then put the
        grasp point 11 cm above the bag, in mid-air: the fingers arrived (TCP within
        0.04-0.06 m), closed on the edge of a 55 mm slab and it slid straight out, for
        281 steps of trying. Any object resting on the ground has its centre one
        half-height up, so the top is 2*z_obj - z_ground whatever its orientation.
        """
        p = obj.data.root_pos_w.clone()
        # z_obj - z_ground gives the half-height for whatever way the object came to
        # rest, but z_ground is a per-section CONSTANT: on the pebble bed an object
        # perched on a stone reads several cm too tall, the grasp point lands above it
        # and the fingers close on nothing but the top edge — 4 of 7 mission failures
        # were "closed on it but it slipped" after this was made terrain-dependent.
        # The true half-height is bounded by the object's own extents either way.
        if self.top_off is not None:
            # Origin is the object's base, not its centre: the top is a known fixed
            # offset up, and the measured-height rule below does not apply.
            half = self.top_off
        else:
            half = (p[:, 2] - ground_z).clamp(min=self.half_lo, max=self.half_hi)
        p[:, 2] = torch.maximum(p[:, 2] + half - self.grip_depth,
                                torch.tensor(ground_z + 0.010, device=self.device))
        return p

    def build(self, obj, ground_z: float) -> torch.Tensor:
        d = self.robot.data
        ee_p = d.body_pos_w[:, self.arm.body_idx]
        ee_q = d.body_quat_w[:, self.arm.body_idx]
        root_p, root_q = d.root_pos_w, d.root_quat_w
        pos_b = math_utils.quat_apply_inverse(root_q, ee_p - root_p)
        quat_b = math_utils.quat_mul(math_utils.quat_inv(root_q), ee_q)

        tcp_w = ee_p + math_utils.quat_apply(
            ee_q, torch.tensor([[0.0, 0.0, TCP_OFFSET]], device=self.device))
        if self.target_b_fn is not None:
            # object-honest: base-frame target from the detection, base-frame TCP from
            # the arm FK; the ee->base rotation is the same quat_b the policy observes
            tcp_b = pos_b + math_utils.quat_apply(
                quat_b, torch.tensor([[0.0, 0.0, TCP_OFFSET]], device=self.device))
            grasp_vec = math_utils.quat_apply_inverse(quat_b, self.target_b_fn() - tcp_b)
            self.hold += 1
        else:
            true_point = self._grasp_point_w(obj, ground_z)
            # THE REFRESH RATE IS A FICTION ON HARDWARE. PN_HOLD=5 re-noises the target
            # off the object's LIVE pose at 10 Hz, but the real detector is capped at 5 Hz
            # and NEAR_CUT=0.40 (perception.py:135) plus arm occlusion put the grasp
            # workspace outside what the head camera can see at all — so the true
            # on-robot refresh rate during the attempt is ZERO. MISSION_GRASP_FREEZE=1
            # seeds est_w once at the start of the attempt and holds it, which is the
            # honest deployment model; the drop in success rate between the two is the
            # price of having no close-range sensor.
            if self.est_w is None or (not GRASP_FREEZE and self.hold % GRASP_HOLD == 0):
                self.est_w = true_point + self.bias + torch.randn(1, 3, device=self.device) * PN_JITTER
            self.hold += 1
            grasp_vec = math_utils.quat_apply_inverse(ee_q, self.est_w - tcp_w)

        tool_w = math_utils.quat_apply(ee_q, torch.tensor([[0.0, 0.0, 1.0], ], device=self.device))
        target_b = math_utils.quat_apply_inverse(
            root_q, torch.tensor([[0.0, 0.0, -1.0]], device=self.device))
        tool_b = math_utils.quat_apply_inverse(root_q, tool_w)

        names = self.robot.joint_names
        j7, j8 = names.index("arm_joint7"), names.index("arm_joint8")
        width = (d.joint_pos[:, j7] - d.joint_pos[:, j8]).unsqueeze(1)

        lin_acc = d.body_lin_acc_w[:, self.base_idx].clamp(-50.0, 50.0) * 0.1
        if GRASP_NOYAW:
            # training frame: yaw 0. q = q_yaw * q_tilt  ->  q_tilt = q_yaw^-1 * q, and
            # the world acceleration seen from the yaw-free frame is R_yaw^-1 * a_w.
            _yaw = math_utils.yaw_quat(root_q)
            root_q = math_utils.quat_mul(math_utils.quat_inv(_yaw), root_q)
            lin_acc = math_utils.quat_apply_inverse(_yaw, lin_acc)
        base_state = torch.cat([root_q, d.root_ang_vel_b * 0.25, lin_acc], dim=1)

        if REAL_NOISE:
            # encoder/FK error, finger encoder, IMU — the magnitudes the champion was
            # fine-tuned under (GRASP_REAL_NOISE=1 in grasp_env_cfg.py)
            pos_b = pos_b + _u(pos_b.shape, GRASP_NOISE["ee_pose"], self.device)
            quat_b = quat_b + _u(quat_b.shape, GRASP_NOISE["ee_pose"], self.device)
            width = width + _u(width.shape, GRASP_NOISE["width"], self.device)
            target_b = target_b + _u(target_b.shape, GRASP_NOISE["axes"], self.device)
            tool_b = tool_b + _u(tool_b.shape, GRASP_NOISE["axes"], self.device)
            base_state = base_state + _u(base_state.shape, GRASP_NOISE["base_state"],
                                         self.device)
        return torch.cat([
            pos_b, quat_b, grasp_vec, target_b, tool_b, width, self.last_action,
            self.params, self.dims * self.dim_err, base_state,
        ], dim=1)

    def build_virtual(self, pos_b: torch.Tensor, quat_b: torch.Tensor, obj,
                      ground_z: float, width: float = 2.0 * GRIP_OPEN) -> torch.Tensor:
        """The step-0 observation the policy WOULD see from N hypothetical gripper_base
        poses (base frame, [N,3] + [N,4] wxyz), for scoring pregrasp-bank start
        configurations BEFORE the arm is moved there. Same arithmetic as build(), with
        the pose supplied instead of read from the sim; the grasp point is seeded here
        so the scored and the real first observation share it (reset_target must have
        been called), the body state is the robot's live one, the fingers are open and
        last_action is zero — a fresh attempt.

        Why it exists: training (graspenv reset_pregrasp_rsi) drops the object under the
        gripper base of a random bank config, so the actor has only ever seen step-0
        grasp vectors of ~0.15 m or less. The mission's "far" pick starts 0.15-0.34 m
        away, outside that, and the actor then drives AWAY from the object into the
        wrist joint limits (the flail). Scoring every reachable bank config through the
        actor and starting from the one whose first action points at the object keeps
        the start inside the distribution the policy was trained on.
        """
        d = self.robot.data
        n = int(pos_b.shape[0])
        root_p, root_q = d.root_pos_w, d.root_quat_w
        root_p_n, root_q_n = root_p.expand(n, 3), root_q.expand(n, 4)
        if self.target_b_fn is not None:
            est_b = self.target_b_fn().expand(n, 3)
        else:
            true_point = self._grasp_point_w(obj, ground_z)
            if self.est_w is None:
                self.est_w = true_point + self.bias + torch.randn(1, 3, device=self.device) * PN_JITTER
            est_b = math_utils.quat_apply_inverse(root_q_n, self.est_w.expand(n, 3) - root_p_n)
        tcp_b = pos_b + math_utils.quat_apply(
            quat_b, torch.tensor([[0.0, 0.0, TCP_OFFSET]], device=self.device).expand(n, 3))
        grasp_vec = math_utils.quat_apply_inverse(quat_b, est_b - tcp_b)
        tool_b = math_utils.quat_apply(
            quat_b, torch.tensor([[0.0, 0.0, 1.0]], device=self.device).expand(n, 3))
        target_b = math_utils.quat_apply_inverse(
            root_q_n, torch.tensor([[0.0, 0.0, -1.0]], device=self.device).expand(n, 3))
        width_t = torch.full((n, 1), float(width), device=self.device)
        lin_acc = d.body_lin_acc_w[:, self.base_idx].clamp(-50.0, 50.0) * 0.1
        if GRASP_NOYAW:
            _yaw = math_utils.yaw_quat(root_q)
            root_q = math_utils.quat_mul(math_utils.quat_inv(_yaw), root_q)
            lin_acc = math_utils.quat_apply_inverse(_yaw, lin_acc)
        base_state = torch.cat([root_q, d.root_ang_vel_b * 0.25, lin_acc], dim=1).expand(n, 10)
        return torch.cat([
            pos_b, quat_b, grasp_vec, target_b, tool_b, width_t,
            torch.zeros(n, 7, device=self.device), self.params.expand(n, 5),
            (self.dims * self.dim_err).expand(n, 3), base_state,
        ], dim=1)


# ======================================================================================
# perception: head-camera colour detection + depth unprojection
# ======================================================================================
