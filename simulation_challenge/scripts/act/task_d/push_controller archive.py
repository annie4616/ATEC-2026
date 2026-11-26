"""Velocity-command state machine for Task D: push the box into the pit, then cross.

Scene geometry (world frame, from task_d/env_cfg.py + terrain.py):
  - Robot spawns near x = -3, y = 0, walking toward +x.
  - The pit spans roughly x in [-0.7, 0.7] across the full y width; the reward
    `box_in_target_x` fires when the box center x enters [-0.7, 0.7].
  - The box spawns at (-3, 1.6, 0.5), size 0.8 (x) x 1.0 (y) x 0.6 (z): same x as
    the robot, offset +1.6 m in y. So box x-half = 0.4, y-half = 0.5.
  - Success: robot root x > 3.5.

The robot and box start at the SAME x, with the box 1.6 m to the robot's +y.
If the robot strafes toward the box in +y while still level in x, it clips the
box's -y face and shoves the box sideways (+y) forever. So we approach in three
strictly ordered phases:

  BACK_UP : reverse in -x (y held fixed) until the robot is well behind the box's
            -x face, with enough clearance that a later +y strafe cannot touch it.
  ALIGN_Y : x held (no forward creep), strafe in +y until level with the box's y.
  TURN    : at the push position (still behind the box, not in contact), rotate in
            place to PUSH_TARGET_YAW before driving forward.
  PUSH    : drive +x to push the box forward until its center x reaches the pit.
  CROSS   : once the box is in the pit, drive +x over the threshold x > 3.5.

Heuristic oracle; the collector still filters on the real success condition
(robot x > 3.5).
"""

from __future__ import annotations

import math

import torch


class PushIntoPitController:
    PIT_X_MIN = -0.7
    PIT_X_MAX = 0.7
    # Box geometry margins.
    BOX_X_HALF = 0.4
    BOX_Y_HALF = 0.5
    # Clearance behind the box's -x face before it's safe to strafe in y.
    # = box x-half + box y-half (worst-case corner) + robot body margin.
    BACK_CLEARANCE = 0.4
    Y_ALIGN_TOL = 0.18
    # If the gap from the robot front to the box -x face exceeds this, sprint to
    # close it; otherwise push steadily. (No reverse: the flat policy can't.)
    PUSH_GAP_MAX = 0.30
    V_FORWARD = 2.5 # 속도 push 때랑 cross 할 때 다르게 해야 할 수도
    V_PUSH = 0.4           # slower while pushing, so we don't outrun the box
    V_BACK = -3.0
    V_SIDE = 1.3
    # Diagonal push: while driving the box forward (+x), also nudge it toward the
    # pit on the -y side. Set V_PUSH_Y = 0 for a pure +x push.
    V_PUSH_Y = -0.40
    # Lateral offset of the robot's push-contact point from the box CENTER in y.
    # The robot lines up at (by - PUSH_Y_OFFSET) instead of the box center, so a
    # positive value makes it start pushing from the -y side of the box.
    PUSH_Y_OFFSET = -0.1515
    # While in contact during PUSH, rotate (yaw) the robot to this target heading
    # (radians, world frame; 0 = facing +x). Rotation stops once within tolerance.
    PUSH_TARGET_YAW = -0.65        # turn to face down-and-left (toward the -y pit)
    PUSH_YAW_TOL = 0.01            # rad; stop rotating once this close to target
    V_TURN = 1.5                 # yaw rate (rad/s) while turning toward target

    def __init__(self):
        self.state = "BACK_UP"

    def reset(self):
        self.state = "BACK_UP"

    def compute_velocity_command(
        self,
        robot_pos: torch.Tensor,    # (3,) world
        box_pos: torch.Tensor,      # (3,) world
        robot_quat: torch.Tensor | None = None,  # (4,) world wxyz; for yaw turn
    ) -> tuple[float, float, float]:
        rx, ry = robot_pos[0].item(), robot_pos[1].item()
        bx, by = box_pos[0].item(), box_pos[1].item()

        # Current robot yaw (heading) from the world-frame quaternion (wxyz).
        ryaw = None
        if robot_quat is not None:
            w, x, y, z = (robot_quat[0].item(), robot_quat[1].item(),
                          robot_quat[2].item(), robot_quat[3].item())
            ryaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        # Target y the robot lines up to before pushing: offset to the -y side of
        # the box center so it starts pushing left-of-center (from behind).
        push_y = by - self.PUSH_Y_OFFSET

        # Behind the box's -x face by a safe clearance (so +y strafing is clean).
        behind_clear = rx <= (bx - self.BOX_X_HALF - self.BACK_CLEARANCE)
        y_aligned = abs(ry - push_y) <= self.Y_ALIGN_TOL
        box_in_pit = self.PIT_X_MIN <= bx <= self.PIT_X_MAX
        # Gap between the robot's front and the box's -x face.
        gap = (bx - self.BOX_X_HALF) - rx

        # Yaw error toward the target heading (only meaningful if we have a quat).
        yaw_err = None
        if ryaw is not None:
            yaw_err = self._wrap_angle(self.PUSH_TARGET_YAW - ryaw)
        yaw_aligned = (yaw_err is None) or (abs(yaw_err) <= self.PUSH_YAW_TOL)

        # ---- transitions ----
        if self.state == "BACK_UP":
            if behind_clear:
                self.state = "ALIGN_Y"
        elif self.state == "ALIGN_Y":
            if not behind_clear:
                self.state = "BACK_UP"          # drifted forward; reverse again
            elif y_aligned:
                self.state = "TURN"             # at push position -> rotate first
        elif self.state == "TURN":
            if yaw_aligned:
                self.state = "PUSH"             # facing target -> now push
        elif self.state == "PUSH":
            if box_in_pit:
                self.state = "CROSS"

        # ---- command generation ----
        if self.state == "BACK_UP":
            # Reverse only; never move in y here, or we clip the box.
            return self.V_BACK, 0.0, 0.0

        if self.state == "ALIGN_Y":
            # Strafe toward the offset push_y (left-of-center); hold x (no forward
            # creep) so we stay safely behind the box while lining up.
            vy = self.V_SIDE if (push_y - ry) > 0 else -self.V_SIDE
            return 0.0, vy, 0.0

        if self.state == "TURN":
            # Reached the push position (still behind the box, NOT in contact yet).
            # Rotate in place toward PUSH_TARGET_YAW before driving forward; hold
            # x and y so the heading change doesn't disturb our alignment.
            wz = 0.0
            if yaw_err is not None and abs(yaw_err) > self.PUSH_YAW_TOL:
                wz = self.V_TURN if yaw_err > 0 else -self.V_TURN
            return 0.0, 0.0, wz

        if self.state == "PUSH":
            # This flat policy reverses poorly, so we NEVER back up here. Contact
            # with the box's -x face means gap ~ 0, and that is exactly when we
            # must keep driving to move the box. Only stop if we actually overran
            # the box center (rx > bx), letting the box settle ahead of us.
            if rx > bx:
                vx = 0.0                         # overran box center -> wait
            elif gap > self.PUSH_GAP_MAX:
                vx = self.V_FORWARD              # gap opened -> close it fast
            else:
                vx = self.V_PUSH                 # in contact -> push steadily

            # Diagonal push: a constant -y component drives the box toward the pit.
            # Plus a weak safety correction so the robot doesn't drift off its
            # intended offset line (push_y) faster than it pushes.
            vy = -self.V_PUSH_Y
            if (push_y - ry) > 0.3:              # robot fell too far -y of push_y
                vy += 0.15                       # ease back toward push_y
            elif (push_y - ry) < -0.3:           # robot drifted +y past push_y
                vy -= 0.15

            # Heading was already set in the TURN state; just keep driving/pushing.
            return vx, vy, 0.0

        # CROSS: drive straight to the goal.
        return self.V_FORWARD, 0.0, 0.0

    @staticmethod
    def _wrap_angle(a: float) -> float:
        """Wrap an angle to [-pi, pi] so the turn takes the shortest direction."""
        return (a + math.pi) % (2.0 * math.pi) - math.pi
