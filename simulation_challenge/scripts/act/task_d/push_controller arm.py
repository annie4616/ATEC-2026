"""Arm-assisted box placement for Task D: push the box into the pit, then use the
ARM to fine-position it so it MAXIMALLY spans the pit's x-extent — minimizing the
open fore/aft (x) gap the robot could fall into while crossing.

Geometry (world frame):
  - Pit: world x in [-0.4, +0.4] (width ~0.8 m in x), long in y, DEPTH 1.0 m.
  - Box: 0.8 (x) x 1.0 (y) x 0.6 (z), 8 kg.
  - Laid flat with its 0.8 m side along x and CENTERED at x=0, the box footprint
    exactly bridges the pit width [-0.4, +0.4]. The failure mode in earlier runs was
    the box settling OFF-CENTER or YAWED, leaving an open x-gap on one side where the
    robot's feet dropped in. So the arm's job is the LAST-MILE fix: nudge the box so
    its center sits at x ~ 0 and it is square (yaw ~ 0), giving full x-coverage.
  - The arm cannot LIFT the 8 kg box (effort ~100 N·m), but it can PUSH it
    horizontally (only friction to overcome) — so the arm presses the box's near
    x-face toward the pit center.

Phases:
  BACK_UP / ALIGN_Y / PUSH_Y / ALIGN_X / PUSH_X : velocity-driven coarse push that
      gets the box roughly into the pit (same as push_controller.py).
  ARM_ADJUST : robot holds just behind the pit; the ARM reaches down onto the box and
      pushes it toward pit-center x=0 / squares it, until x-coverage is maximized.
  DONE       : box is centered & square in the pit; hand back to the crosser.

Interface:
  ctrl = ArmPlaceController(robot=..., device=..., num_envs=1)
  ctrl.reset()
  vel_cmd, arm_jpos_des = ctrl.compute(robot_pos_w, box_pos_w, robot_quat_w, box_quat_w)
    - vel_cmd: (vx, vy, wz) body-frame velocity command for the walking policy.
    - arm_jpos_des: (1, n_arm) desired arm joint positions, or None to hold the arm
      at its default pose. The caller maps this onto the full action vector.
"""

from __future__ import annotations

import math

import torch

from atec_rl_lab.utils import CartesianController


class ArmPlaceController:
    # --- pit / box geometry (world frame) ---
    PIT_X_CENTER = 0.0
    PIT_X_MIN = -0.4
    PIT_X_MAX = 0.4
    BOX_X_HALF = 0.4
    BOX_Y_HALF = 0.5

    # Coarse-push targets (same intent as push_controller.py): clear the platform in
    # -y, then push +x into the pit so the box lands near the pit center.
    BOX_Y_TARGET = -0.3
    BOX_PUSH_DONE_X = 0.0          # push until the box center reaches the pit center

    # Clearances / tolerances for the coarse push.
    BACK_CLEARANCE = 0.4
    SIDE_CLEARANCE = 0.25
    ALIGN_Y_OVERSHOOT_MARGIN = 0.15
    ALIGN_TOL = 0.10

    # Coarse-push speeds (robot faces +x, so vx=world+x, vy=world+y).
    V_FORWARD = 2.5
    V_PUSH = 1.4
    V_PUSH_INTO_PIT = 0.8
    PUSH_INTO_PIT_X = -0.6
    V_BACK = -2.0
    V_SIDE = 1.0

    HEADING_TOL = 0.15
    V_TURN = 1.5

    # --- ARM_ADJUST phase ---
    # How precisely the box must be centered/squared in x before we call it done.
    BOX_X_CENTER_TOL = 0.06        # |box_x - 0| under this -> centered enough
    BOX_YAW_TOL = 0.10             # |box yaw| under this -> square enough (rad)
    # Where the robot stands during ARM_ADJUST: just behind the near pit rim so the
    # box is within arm reach in front of it. (world x; robot faces +x.)
    ARM_STAND_X = -1.05
    ARM_STAND_TOL = 0.12
    # Gripper contact point on the box (base-frame, relative to the robot root):
    # reach FORWARD (+x) and DOWN (-z) to touch the box's top/near face in the pit.
    # The box top in the pit sits ~ -0.4 in world; the robot root is ~ +0.55, so the
    # gripper must drop ~0.9 m and reach forward to the box's near x-face. These are a
    # first guess — tune on video (see ARM_PUSH_STEP for how it advances the target).
    ARM_REACH_Z_B = -0.65          # base-frame z of the gripper contact (down)
    ARM_REACH_X_NEAR_B = 0.55      # base-frame x where the gripper first contacts
    ARM_PUSH_STEP = 0.015          # per-step +x advance of the gripper target (push)
    ARM_PUSH_X_MAX_B = 1.10        # don't drive the gripper target past this (reach lim)

    def __init__(self, robot, device: str = "cuda:0", num_envs: int = 1,
                 ee_body_name: str = "gripper_base",
                 arm_joint_names: list[str] | None = None):
        self.robot = robot
        self.device = device
        if arm_joint_names is None:
            # 6 controllable arm joints (exclude the 2 gripper fingers) for IK.
            arm_joint_names = ["arm_joint1", "arm_joint2", "arm_joint3",
                               "arm_joint4", "arm_joint5", "arm_joint6"]
        # Position-only IK: we only care where the gripper touches the box, not its
        # orientation, and the arm starts far from any particular orientation.
        self.ik = CartesianController(
            robot=robot,
            ee_body_name=ee_body_name,
            arm_joint_names=arm_joint_names,
            num_envs=num_envs,
            device=device,
            command_type="position",
        )
        self.state = "BACK_UP"
        # current gripper push target x (base-frame), advanced during ARM_ADJUST
        self._arm_push_x_b = self.ARM_REACH_X_NEAR_B

    def reset(self):
        self.state = "BACK_UP"
        self._arm_push_x_b = self.ARM_REACH_X_NEAR_B
        self.ik.reset()

    @staticmethod
    def _yaw_from_quat(q) -> float:
        w, x, y, z = q[0].item(), q[1].item(), q[2].item(), q[3].item()
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _face_x_wz(self, ryaw):
        if ryaw is None or abs(ryaw) <= self.HEADING_TOL:
            return 0.0
        return -self.V_TURN if ryaw > 0 else self.V_TURN

    def _arm_target(self):
        """Drive the gripper to the current base-frame push target and return arm
        joint positions (1, n_arm). The caller overlays these on the action vector."""
        ee_pos_b = torch.tensor(
            [[self._arm_push_x_b, 0.0, self.ARM_REACH_Z_B]],
            device=self.device, dtype=torch.float32,
        )
        return self.ik.compute_base(ee_pos_b, ee_quat_b=None)

    def compute(
        self,
        robot_pos: torch.Tensor,
        box_pos: torch.Tensor,
        robot_quat: torch.Tensor | None = None,
        box_quat: torch.Tensor | None = None,
    ):
        """Return (vel_cmd=(vx,vy,wz), arm_jpos_des or None)."""
        rx, ry = robot_pos[0].item(), robot_pos[1].item()
        bx, by, bz = box_pos[0].item(), box_pos[1].item(), box_pos[2].item()
        ryaw = self._yaw_from_quat(robot_quat) if robot_quat is not None else None
        byaw = self._yaw_from_quat(box_quat) if box_quat is not None else 0.0
        wz = self._face_x_wz(ryaw)

        # --- predicates (coarse push) ---
        behind_x = rx <= (bx - self.BOX_X_HALF - self.BACK_CLEARANCE)
        on_plus_y = ry >= (by + self.BOX_Y_HALF + self.SIDE_CLEARANCE)
        x_centered = abs(rx - bx) <= self.ALIGN_TOL
        y_centered = abs(ry - by) <= self.ALIGN_TOL
        box_y_done = by <= self.BOX_Y_TARGET

        # box placement quality (for ARM_ADJUST -> DONE)
        box_x_centered = abs(bx - self.PIT_X_CENTER) <= self.BOX_X_CENTER_TOL
        box_square = abs(byaw) <= self.BOX_YAW_TOL
        robot_at_stand = abs(rx - self.ARM_STAND_X) <= self.ARM_STAND_TOL

        # --- transitions ---
        if self.state == "BACK_UP":
            if behind_x:
                self.state = "ALIGN_Y"
        elif self.state == "ALIGN_Y":
            if on_plus_y and x_centered:
                self.state = "PUSH_Y"
        elif self.state == "PUSH_Y":
            if box_y_done:
                self.state = "ALIGN_X"
        elif self.state == "ALIGN_X":
            if behind_x and y_centered:
                self.state = "PUSH_X"
        elif self.state == "PUSH_X":
            if by > (self.BOX_Y_TARGET + 0.5):
                self.state = "ALIGN_Y"
            elif bx >= self.BOX_PUSH_DONE_X:
                # box is roughly in the pit -> hand off to the arm for fine placement
                self.state = "ARM_ADJUST"
                self._arm_push_x_b = self.ARM_REACH_X_NEAR_B
        elif self.state == "ARM_ADJUST":
            if box_x_centered and box_square:
                self.state = "DONE"

        # =============================== commands ===============================
        if self.state == "BACK_UP":
            return (self.V_BACK, 0.0, wz), None

        if self.state == "ALIGN_Y":
            y_target = by + self.BOX_Y_HALF + self.SIDE_CLEARANCE
            y_cap = y_target + self.ALIGN_Y_OVERSHOOT_MARGIN
            if not on_plus_y:
                vx = self.V_BACK if not behind_x else 0.0
                vy = 0.0 if ry >= y_cap else self.V_SIDE
                return (vx, vy, wz), None
            vx = 0.0
            if abs(rx - bx) > self.ALIGN_TOL:
                vx = self.V_PUSH if (bx - rx) > 0 else -self.V_PUSH
            vy = -self.V_SIDE if ry > y_cap else 0.0
            return (vx, vy, wz), None

        if self.state == "PUSH_Y":
            xdiff = bx - rx
            vx = max(-self.V_PUSH, min(self.V_PUSH, 3.0 * xdiff))
            vy = 0.0 if abs(xdiff) > 0.30 else -self.V_SIDE
            return (vx, vy, wz), None

        if self.state == "ALIGN_X":
            vx = self.V_BACK if not behind_x else 0.0
            vy = 0.0
            if abs(ry - by) > self.ALIGN_TOL:
                vy = self.V_SIDE if (by - ry) > 0 else -self.V_SIDE
            return (vx, vy, wz), None

        if self.state == "PUSH_X":
            ydiff = by - ry
            vy = max(-self.V_SIDE, min(self.V_SIDE, 3.0 * ydiff))
            gap = (bx - self.BOX_X_HALF) - rx
            if abs(ydiff) > 0.30 or rx > bx:
                vx = 0.0
            elif bx >= self.PUSH_INTO_PIT_X:
                vx = self.V_PUSH_INTO_PIT
            elif gap > 0.3:
                vx = self.V_FORWARD
            else:
                vx = self.V_PUSH
            return (vx, vy, wz), None

        if self.state == "ARM_ADJUST":
            # 1) Park the robot just behind the near pit rim (don't walk into the pit),
            #    keeping it square to +x and centered on the box's y so the arm can
            #    reach the box dead ahead.
            vx = 0.0
            if not robot_at_stand:
                vx = self.V_PUSH if rx < self.ARM_STAND_X else self.V_BACK
            ydiff = by - ry
            vy = max(-self.V_SIDE, min(self.V_SIDE, 2.0 * ydiff))

            # 2) ARM push: if the box still sits BEHIND the pit center (bx < 0), advance
            #    the gripper target forward (+x) to push the box's near face toward
            #    center. If it OVERSHOT (bx > 0), back the gripper target off so we stop
            #    pushing (we can't pull it back with a position-only press, so easing off
            #    lets the coarse dynamics / next pass settle it).
            if bx < self.PIT_X_CENTER - self.BOX_X_CENTER_TOL:
                self._arm_push_x_b = min(self.ARM_PUSH_X_MAX_B,
                                         self._arm_push_x_b + self.ARM_PUSH_STEP)
            elif bx > self.PIT_X_CENTER + self.BOX_X_CENTER_TOL:
                self._arm_push_x_b = max(self.ARM_REACH_X_NEAR_B,
                                         self._arm_push_x_b - self.ARM_PUSH_STEP)
            arm_jpos = self._arm_target()
            return (vx, vy, wz), arm_jpos

        # DONE: box centered & square in the pit. Hold position, arm back to default.
        return (0.0, 0.0, wz), None
