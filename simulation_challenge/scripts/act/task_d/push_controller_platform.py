"""Velocity-command state machine for Task D *platform* strategy (+x climb).

IDEA: don't fill the pit. Push the box up against the PLATFORM's -x front face and
use it as a single step to CLIMB +x onto the platform top, then walk +x across the
platform and off the far side toward x > 3.5.

Why +x (not +y): try1 showed that pulling the box -y (robot pushing from the side)
is slow and unreliable — the walking policy shuffles laterally and the box stalls.
The box STARTS at world (-3, +1.6), which is already inside the platform's y-band
(+0.5..+3.5), so we don't need to move it in y at all: just push it +x straight to
the platform front and climb +x over it. This reuses the reliable forward push.

Scene geometry (world frame, env_origin = (-4.2,0,0); assets spawn WITHOUT the
env-origin offset, so init_state.pos == world pos; confirmed by probe):
  - Robot spawns world (-3.04, 0.00, 0.52); box spawns world (-3.00, +1.60, 0.30
    after settling -> box is 0.6 tall, so its TOP is at z ~ +0.60).
  - Platform: raised block, world x in [+0.4, +2.0], y in [+0.5, +3.5], TOP z ~ +1.4.
    Its -x face is a vertical wall at world x ~ +0.4 (the front the robot hits first).
  - Pit: same x-band on the -y side (y < +0.5), depth 1.0.

Plan (robot keeps facing +x, yaw 0, so vx = world +x, vy = world +y):
  GO_TO_BOX_Y : strafe from spawn (y~0) up to the box's y-line (~+1.6) so the robot
                is in the platform y-band and lined up to push the box +x.
  ALIGN_X     : get behind the box's -x face, centered on its y.
  PUSH_X      : push the box +x until its center reaches park_x (box +x face at the
                platform front ~+0.4).
  APPROACH    : close right up behind the box (still in -x), centered on its y.
  CLIMB       : drive +x HARD to step ground -> box-top(0.6) -> platform-top(1.4).
  CROSS       : up on the platform -> drive +x off the far side toward x > 3.5.

NOTE: platform top ~1.4 m, box top ~0.6 m -> a ~0.8 m remaining step. One box may
not be enough for the walking policy to mount in one move; this is an EXPERIMENT to
see how far the box-as-step gets us (judge from the video).

Key coordinates are constructor args so they can be corrected after the video.
"""

from __future__ import annotations

import math

import torch


class PushToPlatformController:
    BOX_X_HALF = 0.4
    BOX_Y_HALF = 0.5

    def __init__(
        self,
        park_x: float = -1.0,        # box center x once parked against the wall (MEASURED:
                                     # box stalls at x~-1.06 wedged on the platform front)
        climb_y: float = 1.6,        # y-line to push/climb along (box start y, inside platform band)
        # platform TOP z. TaskDEnvCfg.platform_height_range=(1.0,1.2), difficulty=0
        # -> platform height ~1.0 m. Box top ~0.6 m, so the residual step from the
        # box to the platform is ~0.4 m (the climbable target the RL was tuned for).
        platform_top_z: float = 1.0,
        platform_front_x: float = -0.66,  # MEASURED: platform -x face ~ box_stall_x + box_half
    ):
        self.park_x = park_x
        self.climb_y = climb_y
        self.platform_top_z = platform_top_z
        self.platform_front_x = platform_front_x
        self.state = "GO_TO_BOX_Y"
        # box-stall detection: if the box stops advancing in +x while we push, it is
        # wedged against the platform wall -> treat it as parked and start climbing.
        self._last_bx = None
        self._stall_steps = 0

    def reset(self):
        self.state = "GO_TO_BOX_Y"
        self._last_bx = None
        self._stall_steps = 0

    # --- speeds (robot faces +x: vx = world +x, vy = world +y) ----------------
    V_FORWARD = 2.0
    V_PUSH = 1.4
    V_BACK = -1.8
    V_SIDE = 1.0
    V_CLIMB = 1.8                 # +x drive to mount the box->platform step
    V_TURN = 1.5

    BACK_CLEARANCE = 0.45
    APPROACH_GAP = 0.15          # how close behind the box before climbing
    ALIGN_TOL = 0.12
    Y_TOL = 0.15
    HEADING_TOL = 0.15
    PARK_X_TOL = 0.20

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
        rx, ry, rz = robot_pos[0].item(), robot_pos[1].item(), robot_pos[2].item()
        bx, by, bz = box_pos[0].item(), box_pos[1].item(), box_pos[2].item()
        ryaw = self._yaw_from_quat(robot_quat) if robot_quat is not None else None
        wz = self._face_x_wz(ryaw)

        # --- predicates ---
        behind_x = rx <= (bx - self.BOX_X_HALF - self.BACK_CLEARANCE)
        y_on_box = abs(ry - by) <= self.Y_TOL
        x_centered_on_box = abs(rx - bx) <= self.ALIGN_TOL
        close_behind = (bx - self.BOX_X_HALF - rx) <= self.APPROACH_GAP

        # box-wedged detector: once the box reaches the platform wall (~x=-1.06) it
        # stops moving and the robot can't close the last ~0.8 m gap (it just shuffles
        # in place pushing an immovable, wall-backed box — try3/try5). Detect this by
        # the BOX being near the wall (bx > -1.4) and not advancing for a while; the
        # robot's own x-stall is implied. Don't require close_behind (unreachable: the
        # box body blocks the robot from getting within 0.15 m of the box center).
        if self.state == "PUSH_X" and bx > -1.4:
            if self._last_bx is not None and abs(bx - self._last_bx) < 0.003:
                self._stall_steps += 1
            else:
                self._stall_steps = 0
            self._last_bx = bx
        else:
            self._last_bx = bx
            self._stall_steps = 0
        box_parked = (bx >= self.park_x - self.PARK_X_TOL) or (self._stall_steps >= 50)

        # --- transitions ---
        if self.state == "GO_TO_BOX_Y":
            if abs(ry - self.climb_y) <= self.Y_TOL:
                self.state = "ALIGN_X"
        elif self.state == "ALIGN_X":
            if behind_x and y_on_box:
                self.state = "PUSH_X"
        elif self.state == "PUSH_X":
            # box wedged on the wall -> go straight to CLIMB (skip APPROACH; the box
            # body blocks the robot from ever getting "close_behind", so the climb
            # itself must close the last gap by mounting the box).
            if box_parked:
                self.state = "CLIMB"
        elif self.state == "CLIMB":
            # mounted: root clearly UP near the platform top (not just standing on
            # the ground at z~0.59) AND past the front face in x. Robot standing root
            # is ~0.55 on the ground; on the platform top (~1.0) it sits ~1.0+0.55.
            # Require rz above ground-standing + a good fraction of the step.
            if rz >= 0.85 and rx >= self.platform_front_x:
                self.state = "CROSS"

        # --- commands ---
        if self.state == "GO_TO_BOX_Y":
            # Reach the box's y-line WITHOUT touching the box or overrunning the +y
            # platform edge (try2 failure: robot strafed +y, bumped the box to y=3.0
            # and walked off the +y void). Two ordered sub-steps:
            #   1) first back up in -x so we're well behind the box (rx << bx);
            #   2) only then strafe +y, with a HARD cap at climb_y so lateral
            #      momentum can't carry us past the box's y onto the platform edge.
            if not behind_x:
                # get behind the box in x first; hold y
                return self.V_BACK, 0.0, wz
            # behind in x: strafe +y but never past climb_y (cap kills overshoot)
            if ry >= self.climb_y - self.Y_TOL:
                vy = 0.0
            elif ry >= self.climb_y - 0.4:
                vy = 0.5            # ease in near the target
            else:
                vy = self.V_SIDE
            return 0.0, vy, wz

        if self.state == "ALIGN_X":
            # back up behind the box's -x face, trim y onto the box center
            vx = self.V_BACK if not behind_x else 0.0
            vy = 0.0
            if abs(ry - by) > self.ALIGN_TOL:
                vy = self.V_SIDE if (by - ry) > 0 else -self.V_SIDE
            return vx, vy, wz

        if self.state == "PUSH_X":
            # push the box +x, staying centered on its y; slow / hold if badly off y
            ydiff = by - ry
            vy = max(-self.V_SIDE, min(self.V_SIDE, 3.0 * ydiff))
            gap = (bx - self.BOX_X_HALF) - rx
            if abs(ydiff) > 0.30 or rx > bx:
                vx = 0.0
            elif gap > 0.3:
                vx = self.V_FORWARD
            else:
                vx = self.V_PUSH
            return vx, vy, wz

        if self.state == "APPROACH":
            # close the last gap behind the box, tight y; keep box from sliding by
            # pushing gently +x while we snug up
            ydiff = by - ry
            vy = max(-self.V_SIDE, min(self.V_SIDE, 3.0 * ydiff))
            vx = self.V_PUSH if not close_behind else 0.0
            return vx, vy, wz

        if self.state == "CLIMB":
            # drive +x HARD to mount the box then the platform. Track the BOX's y
            # (not climb_y) so the robot stays centered on the box as a stepping stone
            # (the box drifted to y~1.16 during the push). Keep y-trim gentle so it
            # doesn't strafe off the step.
            ydiff = by - ry
            vy = max(-0.5, min(0.5, 2.0 * ydiff))
            return self.V_CLIMB, vy, wz

        # CROSS: up on the platform -> drive +x off the far side toward x > 3.5
        return self.V_FORWARD, 0.0, wz
