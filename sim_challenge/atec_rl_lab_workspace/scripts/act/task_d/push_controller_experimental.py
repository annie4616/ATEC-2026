"""Velocity-command state machine for Task D: push the box into the pit, then cross.

Scene geometry (world frame):
  - Robot spawns at x = -3, y = 0; box spawns at x = -3, y = +1.6.
  - The box (0.8 x 1.0 x 0.6) starts inside the PLATFORM y-band (world y +0.5..+3.5,
    height ~1.1 m). The pit/traverse lane is the -y side (world y < +0.5); the
    pit/platform sit at world x ~ 0.5..1.9. Pushing +x jams the box into the
    platform wall, so it must first be moved -y out of the platform band.
  - Success: robot root x > 3.5.

Two pushes, each a SINGLE direction so the sign is unambiguous. The robot keeps
facing +x the whole time (yaw 0), so its body-frame command maps directly to the
world: vx = world +x, vy = world +y. It pushes the box's +y face in -y to clear
the platform, then pushes its -x face in +x into the pit. Pushing hard ("확") in
one axis at a time keeps the narrow box square instead of letting it yaw away.

  BACK_UP : reverse in -x behind the box.
  ALIGN_Y : get onto the box's +y side, centered on its x.
  PUSH_Y  : push the box's +y face hard in -y until it clears the platform.
  ALIGN_X : get behind the box's -x face, centered on its y.
  PUSH_X  : push the box's -x face in +x into the pit.
  CROSS   : step over the settled box (robot x > 3.5).

Heuristic oracle; collector filters on the real success (robot x > 3.5).
"""

from __future__ import annotations

import math

import torch


class PushIntoPitController:
    PIT_X_MIN = -0.7
    PIT_X_MAX = 0.7
    BOX_X_HALF = 0.4
    BOX_Y_HALF = 0.5

    # Push the box in -y until its center reaches this (clear of the platform band).
    BOX_Y_TARGET = -0.3
    # Push the box in +x until its center passes this (past the pit center ~0, so
    # the settled stepping stone sits on the robot's cross path, like the working
    # case where the box ended at x ~ +0.2).
    BOX_PUSH_DONE_X = 0.2

    # Clearances / tolerances.
    BACK_CLEARANCE = 0.4
    SIDE_CLEARANCE = 0.4
    ALIGN_TOL = 0.10

    # Speeds (vx = robot forward = world +x; vy = robot left = world +y; the robot
    # always faces +x so body == world).
    V_FORWARD = 2.5
    V_PUSH = 1.4                  # hard push speed (move the 8 kg box fast)
    V_BACK = -2.0
    V_SIDE = 1.0                  # repositioning / -y push speed (slow enough to
                                 # not overshoot the +y side or topple the robot)

    # Keep the robot facing +x (yaw 0) so vx/vy map straight to world.
    HEADING_TOL = 0.15
    V_TURN = 1.5

    BOX_SETTLED_Z = -0.45
    # Crossing: line up tightly on the box's y, get right behind it, then step over
    # it at a moderate speed (too fast trips on the step; too slow stalls).
    CROSS_Y_TOL = 0.08           # tight y alignment onto the stepping stone
    CROSS_APPROACH_GAP = 0.55    # close to within this of the box before mounting
    V_CROSS = 1.3                # forward speed while stepping over the box

    def __init__(self):
        self.state = "BACK_UP"

    def reset(self):
        self.state = "BACK_UP"

    @staticmethod
    def _yaw_from_quat(q) -> float:
        w, x, y, z = q[0].item(), q[1].item(), q[2].item(), q[3].item()
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _face_x_wz(self, ryaw):
        if ryaw is None or abs(ryaw) <= self.HEADING_TOL:
            return 0.0
        return -self.V_TURN if ryaw > 0 else self.V_TURN

    def compute_velocity_command(
        self,
        robot_pos: torch.Tensor,
        box_pos: torch.Tensor,
        robot_quat: torch.Tensor | None = None,
    ) -> tuple[float, float, float]:
        rx, ry = robot_pos[0].item(), robot_pos[1].item()
        bx, by, bz = box_pos[0].item(), box_pos[1].item(), box_pos[2].item()
        ryaw = self._yaw_from_quat(robot_quat) if robot_quat is not None else None
        wz = self._face_x_wz(ryaw)

        # --- predicates ---
        behind_x = rx <= (bx - self.BOX_X_HALF - self.BACK_CLEARANCE)
        on_plus_y = ry >= (by + self.BOX_Y_HALF + self.SIDE_CLEARANCE)
        x_centered = abs(rx - bx) <= self.ALIGN_TOL
        y_centered = abs(ry - by) <= self.ALIGN_TOL
        box_y_done = by <= self.BOX_Y_TARGET

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
                self.state = "ALIGN_Y"        # box drifted back toward platform
            elif bx >= self.BOX_PUSH_DONE_X:
                # Push the box PAST the pit center so the stepping stone lands on
                # the robot's cross path (not stuck at the near pit edge).
                self.state = "CROSS"

        # --- commands (robot faces +x, so vx=world+x, vy=world+y) ---
        if self.state == "BACK_UP":
            return self.V_BACK, 0.0, wz

        if self.state == "ALIGN_Y":
            # Reach the box's +y side in TWO ordered sub-steps so we never clip the
            # box while repositioning:
            #   1) while still behind the box in x, strafe +y until past its +y side;
            #   2) only once clear in y, close the x gap to center on the box.
            if not on_plus_y:
                # stay behind in x (back up if we crept forward), strafe +y
                vx = self.V_BACK if not behind_x else 0.0
                return vx, self.V_SIDE, wz
            vx = 0.0
            if abs(rx - bx) > self.ALIGN_TOL:
                vx = self.V_PUSH if (bx - rx) > 0 else -self.V_PUSH
            return vx, 0.0, wz

        if self.state == "PUSH_Y":
            # Push the box's +y face in -y, staying centered on the box's x. Drive
            # -y continuously, but firmly re-center in x (a firm +x/-x correction);
            # stop the -y push only when BADLY off-center in x (would yaw the box).
            xdiff = bx - rx
            vx = max(-self.V_PUSH, min(self.V_PUSH, 3.0 * xdiff))  # firm x re-center
            vy = 0.0 if abs(xdiff) > 0.30 else -self.V_SIDE
            return vx, vy, wz

        if self.state == "ALIGN_X":
            # Get behind the box's -x face, centered on its y; keep facing +x.
            vx = self.V_BACK if not behind_x else 0.0
            vy = 0.0
            if abs(ry - by) > self.ALIGN_TOL:
                vy = self.V_SIDE if (by - ry) > 0 else -self.V_SIDE
            return vx, vy, wz

        if self.state == "PUSH_X":
            # Push the box's -x face in +x, staying centered on the box's y. Drive
            # +x continuously (so we never stall), but proportionally slow the push
            # the more off-center we are, and strafe firmly back to center. Stop the
            # +x push only when BADLY off-center (would shove a corner).
            ydiff = by - ry
            vy = max(-self.V_SIDE, min(self.V_SIDE, 3.0 * ydiff))  # firm y re-center
            gap = (bx - self.BOX_X_HALF) - rx
            if abs(ydiff) > 0.30 or rx > bx:
                vx = 0.0                          # badly off-center / overran -> hold x
            elif gap > 0.3:
                vx = self.V_FORWARD
            else:
                vx = self.V_PUSH
            return vx, vy, wz

        # CROSS: step over the box (now a stepping stone in the pit) to the goal.
        ydiff = by - ry
        # 1) Wait for the box to settle to the pit floor (stable footing).
        if bz > self.BOX_SETTLED_Z:
            # use the wait to line up tightly on the box's y
            if abs(ydiff) > self.CROSS_Y_TOL:
                return 0.0, (self.V_SIDE if ydiff > 0 else -self.V_SIDE), wz
            return 0.0, 0.0, wz
        # 2) Get TIGHTLY centered on the box's y before mounting it (so the robot
        #    lands on the stone, not the empty pit beside it).
        if abs(ydiff) > self.CROSS_Y_TOL:
            return 0.0, (self.V_SIDE if ydiff > 0 else -self.V_SIDE), wz
        # 3) Step over the box at a moderate speed, holding the tight y track.
        #    Once past the box (robot ahead of it) sprint to the goal.
        if rx > bx + self.BOX_X_HALF:
            return self.V_FORWARD, 0.5 * ydiff, wz
        return self.V_CROSS, 0.5 * ydiff, wz
