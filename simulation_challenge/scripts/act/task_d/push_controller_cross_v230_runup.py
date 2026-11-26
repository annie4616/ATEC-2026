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


class PushIntoPitCrossController:
    """Experimental variant of PushIntoPitController that, after settling the box as
    a stepping stone, drives the robot ACROSS the pit at a configurable speed. The
    goal is to find the highest CROSS speed at which the robot rides the box over the
    0.4 m step and reaches x > 3.5 WITHOUT toppling. V_CROSS / V_FORWARD_CROSS /
    CROSS_APPROACH_GAP are constructor-tunable so a sweep can probe the fall limit.
    """
    PIT_X_MIN = -0.7
    PIT_X_MAX = 0.7
    BOX_X_HALF = 0.4
    BOX_Y_HALF = 0.5

    # Push the box in -y until its center reaches this (clear of the platform band).
    BOX_Y_TARGET = -0.3
    # Push the box in +x until its center passes this (past the pit center ~0, so
    # the settled stepping stone sits on the robot's cross path, like the working
    # case where the box ended at x ~ +0.2).
    BOX_PUSH_DONE_X = 0.5         # push the box further +x so it settles near the FAR pit
                                 # wall, minimizing the step the robot must climb after
                                 # mounting it (it was settling near the entry at x~-0.2,
                                 # leaving a long 0.4 m climb the robot slipped back off).

    # --- box 90-deg rotation (ROTATE_BOX state) ---
    # Target |yaw| at which the long side spans x. pi/2 = 90 deg.
    BOX_YAW_TARGET = math.pi / 2
    BOX_YAW_TOL = 0.20            # ~11 deg: close enough; stop torquing
    # To spin the box we push one END of its +y face (offset from center in x) so the
    # off-center force yaws it. This is how far in +x from box center the robot lines
    # up while rotating (push the box's +x half -> yaws CCW toward +pi/2).
    ROT_PUSH_X_OFFSET = 0.30
    V_ROT = 1.0                  # +x push speed into the offset back corner (slow so the
                                 # box has time to yaw instead of being shoved forward)
    # While rotating, the +x push also advances the box; cap how far +x it may go before
    # the 90 deg turn completes, so it doesn't reach the pit / platform edge mid-rotation
    # and fall (observed: box ran to x ~ -1.0 at 54 deg and dropped). Hold +x once past
    # this until the yaw target is reached.
    ROT_BOX_X_MAX = -1.8

    # Clearances / tolerances.
    BACK_CLEARANCE = 0.4
    # Reduced from 0.4: the box's +y side sits at world y ~ +2.1 and the platform
    # ENDS (drops to the void) at world y ~ +3.5, leaving only ~1.4 m of standing
    # room on the box's +y side. A large side clearance + strafe overshoot drove the
    # robot off that far edge (it fell at y ~ +3.4). Keep it tight so the robot stops
    # just behind the box's +y face, well clear of the platform edge.
    SIDE_CLEARANCE = 0.25
    # Extra margin past the on_plus_y target beyond which we hard-stop +y strafe, so
    # momentum can't carry the robot off the platform's far edge while it re-centers x.
    ALIGN_Y_OVERSHOOT_MARGIN = 0.15
    ALIGN_TOL = 0.10

    # Speeds (vx = robot forward = world +x; vy = robot left = world +y; the robot
    # always faces +x so body == world).
    V_FORWARD = 3.0
    V_PUSH = 2.5                  # hard push speed (move the 8 kg box fast)
    # Faster shove for the final stretch of PUSH_X, to fling the box decisively over the
    # pit rim so it drops deep instead of teetering on the edge.
    V_PUSH_INTO_PIT = 3.0
    # Box x beyond which PUSH_X switches to the fast into-pit shove (near the rim ~0).
    PUSH_INTO_PIT_X = -0.6
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
    # --- run-up + charge crossing ---
    # Stepping over the settled box from a standstill fails (robot pitches into the pit
    # beside the box). Instead back up to build a run-up, then charge +x at full speed so
    # momentum carries the robot over the box/step. RUN_UP_X = world x to retreat to
    # before charging (well behind the pit); CHARGE drives v_forward_cross.
    RUN_UP_X = -2.2              # retreat to this x to gather a run-up before charging
    RUN_UP_TOL = 0.15
    # Robot x at which we STOP pushing the box and bail to RUN_UP: if the robot follows
    # the box all the way to the pit rim it pitches in WITH the box (observed: robot fell
    # at x~-0.5 still in PUSH_X). Stop while still safely on the near platform; the box
    # carries the rest of the way under its own momentum.
    PUSH_X_BAIL_RX = -1.7        # bail to RUN_UP well before the rim: at -1.3 the robot
                                 # followed the box to the pit edge and fell in during
                                 # RUN_UP. The box still carries to the far wall on its own
                                 # momentum (BOX_PUSH_DONE_X handles the target), so the
                                 # robot can bail early and still get a far-wall box.

    def __init__(self, v_cross=1.3, v_forward_cross=2.5, cross_approach_gap=0.55,
                 enable_rotate=True):
        self.state = "BACK_UP"
        # Tunable crossing speeds for the sweep.
        self.v_cross = v_cross                    # speed while stepping onto/over the box
        self.v_forward_cross = v_forward_cross    # sprint speed once clearly past the box
        self.cross_approach_gap = cross_approach_gap
        # When False, skip the 90-deg box rotation and push the box in its spawn
        # orientation (short side spanning the pit). Run-up + charge crossing still applies.
        self.enable_rotate = enable_rotate
        # Rotation direction, fixed on first ROTATE_BOX entry so we don't flip-flop
        # (and undo the rotation) as box_yaw crosses 0. +1 drives yaw -> -pi/2 (CW).
        self.rot_dir = None

    def reset(self):
        self.state = "BACK_UP"
        self.rot_dir = None

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
        box_quat: torch.Tensor | None = None,
    ) -> tuple[float, float, float]:
        rx, ry = robot_pos[0].item(), robot_pos[1].item()
        bx, by, bz = box_pos[0].item(), box_pos[1].item(), box_pos[2].item()
        ryaw = self._yaw_from_quat(robot_quat) if robot_quat is not None else None
        wz = self._face_x_wz(ryaw)
        # Box yaw, used to rotate the box 90 deg so its long (1.0 m) side spans the
        # pit width (x) instead of the short (0.8 m) side — shrinking the leftover gap
        # from 0.85 m to 0.65 m. yaw ~ 0 at spawn; target is +/-pi/2.
        byaw = self._yaw_from_quat(box_quat) if box_quat is not None else None

        # --- predicates ---
        behind_x = rx <= (bx - self.BOX_X_HALF - self.BACK_CLEARANCE)
        on_plus_y = ry >= (by + self.BOX_Y_HALF + self.SIDE_CLEARANCE)
        x_centered = abs(rx - bx) <= self.ALIGN_TOL
        y_centered = abs(ry - by) <= self.ALIGN_TOL
        box_y_done = by <= self.BOX_Y_TARGET

        # Robot lined up behind the box's +y back corner (in BOTH x and y). The box must
        # ultimately go -y (off the +y platform, into the traverse lane at y<0.5), so the
        # rotating push must come from the +y side and drive the box -y — NOT the -y side
        # (that shoved the box +y toward the platform edge and off it). Push the box's +y
        # back corner with +x: that corner advances, the box yaws AND drifts -y.
        rot_corner_y = by + self.BOX_Y_HALF                    # box +y back corner (yaw~0)
        at_rot_start = behind_x and abs(ry - rot_corner_y) <= self.ALIGN_TOL * 2

        # --- transitions ---
        if self.state == "BACK_UP":
            if self.enable_rotate and byaw is not None:
                # Behind the box AND aligned on its +y corner -> rotate by pushing it +x.
                if at_rot_start:
                    self.state = "ROTATE_BOX"
            elif behind_x:
                # Rotation disabled (or no yaw feedback): go straight to the -y push,
                # pushing the box in its spawn orientation.
                self.state = "ALIGN_Y"
        elif self.state == "ROTATE_BOX":
            if byaw is not None and abs(abs(byaw) - self.BOX_YAW_TARGET) <= self.BOX_YAW_TOL:
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
            elif bx >= self.BOX_PUSH_DONE_X or rx >= self.PUSH_X_BAIL_RX:
                # Box seated in the pit OR robot nearing the rim -> stop pushing and back
                # up for a run-up before charging across (don't follow the box into the pit).
                self.state = "RUN_UP"
        elif self.state == "RUN_UP":
            # Retreat behind the pit and align TIGHTLY on the box's y (within 0.05) before
            # charging, so the robot launches straight down the box's center line (0 deg
            # in y), not at a slight angle that drifts it off the box mid-charge.
            if rx <= self.RUN_UP_X + self.RUN_UP_TOL and abs(ry - by) <= 0.05:
                self.state = "CHARGE"

        # --- commands (robot faces +x, so vx=world+x, vy=world+y) ---
        if self.state == "BACK_UP":
            vx = self.V_BACK if not behind_x else 0.0
            # In BOTH modes, pre-align in y toward the box's +y back corner while backing
            # up, so ALIGN_Y doesn't start a long +y strafe from y~0 that builds momentum
            # and overruns the platform edge (the no-rotate fall). Cap at the +y face so
            # we never push past it here.
            y_cap_backup = by + self.BOX_Y_HALF + self.SIDE_CLEARANCE
            target_y = min(rot_corner_y, y_cap_backup)
            vy = 0.0 if ry >= y_cap_backup else max(-self.V_SIDE, min(self.V_SIDE, 1.5 * (target_y - ry)))
            return vx, vy, wz

        if self.state == "ALIGN_Y":
            # Reach the box's +y side in TWO ordered sub-steps so we never clip the
            # box while repositioning:
            #   1) while still behind the box in x, strafe +y until past its +y side;
            #   2) only once clear in y, close the x gap to center on the box.
            # A hard +y cap stops momentum from carrying the robot off the platform's
            # far edge (it previously overran to y ~ +3.4 and fell).
            # Aim slightly PAST the on_plus_y threshold so the proportional strafe doesn't
            # asymptotically stall just short of it and get stuck in sub-step 1 forever
            # (on_plus_y uses by+HALF+CLEARANCE; settling exactly there is a deadlock).
            y_target = by + self.BOX_Y_HALF + self.SIDE_CLEARANCE + 0.10
            # PROPORTIONAL approach to y_target: a constant V_SIDE strafe builds momentum
            # the walking policy can't stop in time, overrunning y_cap and falling off the
            # platform edge (robot reached y~3.8). Slow down as we near y_target so we
            # settle there instead of sailing past.
            vy_to_target = max(-self.V_SIDE, min(self.V_SIDE, 1.5 * (y_target - ry)))
            if not on_plus_y:
                # stay behind in x (back up if we crept forward), ease toward y_target.
                vx = self.V_BACK if not behind_x else 0.0
                return vx, vy_to_target, wz
            # On the +y side: close the x gap while holding y at y_target (ease back -y
            # if momentum still carried us past it).
            vx = 0.0
            if abs(rx - bx) > self.ALIGN_TOL:
                vx = self.V_PUSH if (bx - rx) > 0 else -self.V_PUSH
            return vx, vy_to_target, wz

        if self.state == "PUSH_Y":
            # Push the box's +y face in -y, staying centered on the box's x. Drive
            # -y continuously, but firmly re-center in x (a firm +x/-x correction);
            # stop the -y push only when BADLY off-center in x (would yaw the box).
            xdiff = bx - rx
            vx = max(-self.V_PUSH, min(self.V_PUSH, 3.0 * xdiff))  # firm x re-center
            vy = 0.0 if abs(xdiff) > 0.30 else -self.V_SIDE
            return vx, vy, wz

        if self.state == "ROTATE_BOX":
            # Rotate by pushing the box's -x FACE (its back) with +x, but OFFSET in -y
            # so the off-center forward push yaws the box instead of just translating it.
            # Physics: robot behind the box (-x side), aimed at the box's -y corner,
            # driving +x -> the -y corner advances while the +y corner lags -> box turns
            # CCW -> yaw INCREASES toward +pi/2. This reuses the well-behaved "push +x
            # from behind" motion (PUSH_X) that the robot does reliably, just shifted
            # off-center; far more stable than the earlier -y side-push that only slid
            # the box. Once rotated, PUSH_Y/PUSH_X take over to seat it in the pit.
            if byaw is None:
                return 0.0, 0.0, wz
            if self.rot_dir is None:
                self.rot_dir = +1.0          # push the +y back corner with +x
            # Track the box-local +y back corner (-x, +y) as it swings with byaw, so the
            # contact point follows the turning box instead of sliding to face center.
            # Local corner offset (box frame): (-BOX_X_HALF, +BOX_Y_HALF).
            lx, ly = -self.BOX_X_HALF, +self.BOX_Y_HALF
            c, s = math.cos(byaw), math.sin(byaw)
            corner_x = bx + (lx * c - ly * s)
            corner_y = by + (lx * s + ly * c)
            # Stand just BEHIND that corner in x and aligned in y, then drive +x.
            target_rx = corner_x - self.BACK_CLEARANCE
            target_ry = corner_y
            vy = max(-self.V_SIDE, min(self.V_SIDE, 2.0 * (target_ry - ry)))
            if rx > corner_x - 0.05:
                # overran the corner in x -> back off so we keep pushing the face, not riding over
                return self.V_BACK, vy, wz
            # Stop advancing the box once it has moved far enough +x: keep re-aligning on
            # the corner (vy) but don't push +x, so the box finishes yawing in place
            # instead of running off the platform / into the pit before reaching 90 deg.
            if bx >= self.ROT_BOX_X_MAX:
                return 0.0, vy, wz
            # Drive +x into the offset corner to apply yaw torque.
            vx = self.V_ROT
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
            elif bx >= self.PUSH_INTO_PIT_X:
                # near the rim -> fast shove so the box drops deep into the pit
                vx = self.V_PUSH_INTO_PIT
            elif gap > 0.3:
                vx = self.V_FORWARD
            else:
                vx = self.V_PUSH
            return vx, vy, wz

        if self.state == "RUN_UP":
            # Back up (-x) to RUN_UP_X while lining up on the box's y, to build a run-up.
            ydiff = by - ry
            vy = max(-self.V_SIDE, min(self.V_SIDE, 2.0 * ydiff))
            vx = self.V_BACK if rx > self.RUN_UP_X else 0.0
            return vx, vy, wz

        # CHARGE: sprint +x at full speed so momentum carries the robot over the box /
        # 0.4 m step, instead of pitching into the pit beside it from a standstill.
        # TRACK THE BOX'S Y AGGRESSIVELY: the robot sprints +x at v_forward_cross (~3),
        # but the box can roll in -y as it drops into the pit; with only a V_SIDE(=1) y
        # correction the robot can't keep up and drifts to the box's +y side, missing it
        # and falling into the pit beside it. Use a strong y gain and let the y-correction
        # speed scale with the forward speed so the robot stays centered on the box's y.
        ydiff = by - ry
        vy_max = max(self.V_SIDE, self.v_forward_cross)   # allow y to keep up with +x
        vy = max(-vy_max, min(vy_max, 5.0 * ydiff))
        # Stay on the box's y line within a TIGHT 0.05 band: if off by more, slow the
        # forward sprint hard and re-center first, so the robot rides straight over the
        # box center instead of clipping its edge and falling in beside it.
        vx = self.v_forward_cross if abs(ydiff) <= 0.05 else self.v_forward_cross * 0.25
        return vx, vy, wz
