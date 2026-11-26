"""Velocity-command state machine for Task D: push the box into the pit, then cross.

Scene geometry (world frame, from task_d/env_cfg.py + terrain.py):
  - Robot spawns at x = -3, y = 0; box spawns at x = -3, y = +1.6.
  - The box (0.8 x 1.0 x 0.6) starts inside the PLATFORM y-band (world y +0.5..+3.5,
    height ~1.1 m). The pit/traverse lane is the -y side (world y < +0.5), and the
    pit/platform sit at world x ~ 0.5..1.9.
  - Pushing the box straight +x jams it into the 1.1 m platform wall. So we must
    FIRST move the box in -y out of the platform band, THEN push it +x into the pit.
  - Reward `box_in_target_x` fires on box center x in [-1.4, 0.7] (y-agnostic), but
    physically the box only reaches that x once it has cleared the platform in y.
  - Success: robot root x > 3.5.

Two-stage push (no in-place rotation; the locomotion policy can strafe, lin_vel_y
was trained on +-1.0):

  BACK_UP   : reverse in -x until well behind the box's -x face (clearance for y moves).
  ALIGN_PY  : strafe to the box's +y side (ry > by) so we can shove the box in -y.
  PUSH_Y    : drive -y, pushing the box's +y face, until box y <= BOX_Y_TARGET
              (box has cleared the platform band).
  ALIGN_MX  : strafe/reverse to line up behind the box's -x face at the box's y.
  PUSH_X    : drive +x to push the box forward until its center x enters the pit.
  CROSS     : once the box is in the pit, drive +x over the threshold x > 3.5.

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
    # Push the box in -y only until its center reaches y = 0 (its +y face at +0.5,
    # just clearing the platform band). Don't over-push: if it ends up far -y (e.g.
    # y < -0.4) the stepping stone sits off to the side of the robot's +x cross path
    # and the robot drops into the empty pit beside it. We slow the push near the
    # target (see PUSH_Y) so momentum doesn't carry it past ~ -0.1.
    BOX_Y_TARGET = 0.0

    # Clearances / tolerances.
    BACK_CLEARANCE = 0.4         # behind the box -x face before lateral moves
    SIDE_CLEARANCE = 0.4         # on the box +y side before the -y push
    Y_ALIGN_TOL = 0.18
    X_ALIGN_TOL = 0.18
    # Before the +x push, center the robot on the box's -x face MUCH more tightly:
    # if it pushes off-center (by even ~0.18 m) it applies a torque, the box yaws,
    # the contact slips to a corner and the box rotates back toward +y. Push only
    # when within this tight band, and require the box to have stopped sliding.
    PUSH_X_CENTER_TOL = 0.15     # how close to the box center (y) before pushing +x
    BOX_STILL_TOL = 0.02         # box must move less than this per call to be "still"
    PUSH_GAP_MAX = 0.30          # contact gap threshold (sprint vs steady push)

    # Speeds (vx = robot forward, vy = robot left, both in the robot/base frame;
    # the robot keeps its default +x heading so base frame ~ world frame).
    V_FORWARD = 2.5
    V_PUSH = 1.2                 # +x push speed while in contact (move the 8 kg box)
    V_BACK = -3.0
    V_SIDE = 1.3                 # lateral repositioning speed
    V_PUSH_Y = -0.8             # -y push speed while shoving the box sideways
    # Before crossing, wait until the box has fully settled to the pit floor so it
    # forms a stable stepping stone. The box settles to z ~ -0.6 in the 1.0 m pit;
    # treat it as settled once its center z drops below this.
    BOX_SETTLED_Z = -0.45
    # During PUSH_X / CROSS the robot must face +x (yaw 0) so its body-frame +x
    # command actually drives world +x. After the -y strafing in PUSH_Y the body
    # can be turned; correct it back toward +x with this yaw rate / tolerance.
    HEADING_TOL = 0.15           # rad; how close to +x before we stop yawing
    V_TURN = 1.0                 # yaw rate (rad/s) to face +x

    def __init__(self):
        self.state = "BACK_UP"
        self._prev_by = None     # box y on the previous call (to detect sliding)

    def reset(self):
        self.state = "BACK_UP"
        self._prev_by = None

    def compute_velocity_command(
        self,
        robot_pos: torch.Tensor,    # (3,) world
        box_pos: torch.Tensor,      # (3,) world
        robot_quat: torch.Tensor | None = None,  # unused (kept for API compatibility)
    ) -> tuple[float, float, float]:
        rx, ry = robot_pos[0].item(), robot_pos[1].item()
        bx, by, bz = box_pos[0].item(), box_pos[1].item(), box_pos[2].item()

        # Robot heading (yaw) from the world-frame quaternion (wxyz). 0 = facing +x.
        ryaw = None
        if robot_quat is not None:
            w, x, y, z = (robot_quat[0].item(), robot_quat[1].item(),
                          robot_quat[2].item(), robot_quat[3].item())
            ryaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        def face_x_wz():
            """Yaw command to turn the robot back toward +x (0 if already facing +x
            or yaw unknown)."""
            if ryaw is None or abs(ryaw) <= self.HEADING_TOL:
                return 0.0
            return -self.V_TURN if ryaw > 0 else self.V_TURN

        # --- geometric predicates ---
        # Behind the box's -x face (so a y move doesn't clip the box).
        behind_x_clear = rx <= (bx - self.BOX_X_HALF - self.BACK_CLEARANCE)
        # On the box's +y side, far enough to push it in -y cleanly.
        on_plus_y_side = ry >= (by + self.BOX_Y_HALF + self.SIDE_CLEARANCE)
        # To push the box in -y the robot must be aligned with the box in X (level
        # with the box's body, not behind/ahead of it), AND on the +y side.
        x_at_box = abs(rx - bx) <= self.X_ALIGN_TOL
        ready_push_y = on_plus_y_side and x_at_box
        # Box has cleared the platform band in y.
        box_y_done = by <= self.BOX_Y_TARGET
        # Lined up behind the box's -x face at the box's y (ready to push +x).
        x_behind = rx <= (bx - self.BOX_X_HALF)
        y_at_box = abs(ry - by) <= self.Y_ALIGN_TOL
        box_in_pit = self.PIT_X_MIN <= bx <= self.PIT_X_MAX
        # Box has stopped sliding in y (PUSH_Y momentum has died out). Until it is
        # still, aligning to its y is futile (it keeps moving out from under us).
        box_still = (self._prev_by is None) or (abs(by - self._prev_by) <= self.BOX_STILL_TOL)
        self._prev_by = by
        # Tightly centered on the box's -x face in y (so the +x push is on-center
        # and doesn't yaw the box into a corner).
        y_centered = abs(ry - by) <= self.PUSH_X_CENTER_TOL

        # --- transitions ---
        if self.state == "BACK_UP":
            if behind_x_clear:
                self.state = "ALIGN_PY"
        elif self.state == "ALIGN_PY":
            if ready_push_y:
                self.state = "PUSH_Y"
        elif self.state == "PUSH_Y":
            if box_y_done:
                self.state = "ALIGN_MX"
        elif self.state == "ALIGN_MX":
            # Enter PUSH_X only when the box has stopped sliding AND we are tightly
            # centered behind it -- otherwise we push a corner and it yaws away.
            if x_behind and box_still and y_centered:
                self.state = "PUSH_X"
        elif self.state == "PUSH_X":
            if by > (self.BOX_Y_TARGET + 0.4):
                # The +x push shoved the box back toward the platform band in y;
                # re-clear it in -y before continuing.
                self.state = "ALIGN_PY"
            elif box_in_pit:
                self.state = "CROSS"

        # --- command generation ---
        if self.state == "BACK_UP":
            # Reverse only; never move in y here, or we clip the box.
            return self.V_BACK, 0.0, 0.0

        if self.state == "ALIGN_PY":
            # Reach (box x, box +y side) in TWO sub-steps to avoid clipping the box:
            #  1) while still behind the box's -x face, strafe +y until past its +y
            #     side (so the later +x move can't hit it);
            #  2) once clear in y, close the x gap to level with the box body.
            y_goal = by + self.BOX_Y_HALF + self.SIDE_CLEARANCE
            if not on_plus_y_side:
                # not yet past the box in y -> stay behind in x, strafe +y only
                return 0.0, self.V_SIDE, 0.0
            # clear of the box in y now -> align x (level with box body)
            vx = 0.0
            if abs(rx - bx) > self.X_ALIGN_TOL:
                vx = self.V_PUSH if (bx - rx) > 0 else -self.V_PUSH
            vy = self.V_SIDE if (y_goal - ry) > 0 else 0.0
            return vx, vy, 0.0

        if self.state == "PUSH_Y":
            # Drive -y to push the box's +y face toward the pit lane. Slow down as
            # the box nears BOX_Y_TARGET so momentum doesn't carry it far past it
            # (over-pushed box lands off the cross path -> robot misses it).
            dy = by - self.BOX_Y_TARGET          # how far the box still has to go
            if dy < 0.15:
                vy = -0.5                        # very close: ease off (less overshoot)
            else:
                vy = self.V_PUSH_Y               # otherwise: full push
            vx = 0.0
            if rx < (bx - self.BOX_X_HALF - 0.1):
                vx = self.V_PUSH                # ease forward to keep contact in x
            return vx, vy, 0.0

        if self.state == "ALIGN_MX":
            # Approach the box's -x face HEAD-ON (not diagonally), so the +x push
            # later hits the face center, not a corner. Do it in strict order:
            #   1) get centered on the box's y while staying behind it in x;
            #   2) only once centered in y, close in to just behind the -x face.
            # Mixing the y-move and x-move would clip a corner and yaw the box.
            if abs(ry - by) > self.PUSH_X_CENTER_TOL:
                # center in y first; hold x behind the box (back up only if too close)
                vx = self.V_BACK if not x_behind else 0.0
                vy = self.V_SIDE if (by - ry) > 0 else -self.V_SIDE
                return vx, vy, 0.0
            # centered in y -> close the x gap straight in (no vy, so we stay on the
            # face center line). Stop closing once just behind the -x face.
            gap = (bx - self.BOX_X_HALF) - rx
            vx = self.V_FORWARD if gap > 0.0 else 0.0
            return vx, 0.0, 0.0

        if self.state == "PUSH_X":
            # Push the box +x into the pit. The robot may be turned from the -y
            # strafing, so keep correcting its heading back to +x; a body that
            # faces +x makes its +x command actually drive world +x (and pushes
            # the box straight). Track the box's y firmly: if we drift off in y the
            # contact slips to a corner and the box stalls (seen at box x ~ -2.6).
            wz = face_x_wz()
            # If the body is still badly turned (from the -y strafing), turning and
            # driving at once just spins in place: face +x FIRST, then push.
            if ryaw is not None and abs(ryaw) > 0.4:
                return 0.0, 0.0, wz
            ydiff = by - ry
            # Keep the contact ON the face center: if we slip off the box's y center,
            # STOP pushing and re-center before driving +x again (off-center push
            # hits a corner and yaws the box away).
            if abs(ydiff) > self.PUSH_X_CENTER_TOL:
                vy = self.V_SIDE if ydiff > 0 else -self.V_SIDE
                return 0.0, vy, wz
            gap = (bx - self.BOX_X_HALF) - rx
            if rx > bx:
                vx = 0.0                          # overran box center -> let it settle
            elif gap > self.PUSH_GAP_MAX:
                vx = self.V_FORWARD               # not in contact -> sprint to close
            else:
                vx = self.V_PUSH                  # in contact -> push hard
            # Gentle y tracking ONLY: a large vy would shove the box sideways (+y,
            # back toward the platform) instead of pushing it straight +x. Keep it
            # small and capped so we just stay roughly centered on the box's -x face.
            vy = max(-0.3, min(0.3, 0.5 * ydiff))
            return vx, vy, wz

        # CROSS: the box is in the pit. Keep correcting heading to +x throughout
        # (after the -y strafing the body is turned; it must face +x to cross).
        wz = face_x_wz()
        # First WAIT for the box to fully settle to the pit floor (z below
        # BOX_SETTLED_Z) so it is a stable stepping stone -- crossing while it is
        # still tipping/falling makes the robot trip. Use the wait to also turn the
        # robot to face +x. Hold position (no -x) so we stay lined up behind it.
        if bz > self.BOX_SETTLED_Z:
            return 0.0, 0.0, wz
        # Box settled. Line up with its y, then drive straight +x over it.
        ydiff = by - ry
        if abs(ydiff) > self.Y_ALIGN_TOL:
            # Not centered on the box in y yet -> strafe to its y before crossing
            # (don't drive +x off to the side and miss the stepping stone).
            vy = self.V_SIDE if ydiff > 0 else -self.V_SIDE
            return 0.0, vy, wz
        # Aligned and facing +x: drive straight over the box.
        return self.V_FORWARD, 0.3 * ydiff, wz
