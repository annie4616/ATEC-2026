"""Getting the object into the bin.

Four of the five objects the simulator ever grasped under honest conditions were lost here,
so this is not the easy half.  Three rules carry it, and all three are refusals:

1. **A bin memory is fresh only if it is young in BOTH time and distance.**  `BinMemory.rel`
   returning None is the single staleness test; never age alone.
2. **Refuse to re-centre on a stale sighting.**  The same 40-90 mm correction rescued one
   drop and ruined another -- expected value zero, variance positive.  Do not take the bet.
3. **Refuse to re-centre when the two beliefs disagree by more than the detector is worth.**
   The bin fix is good to 50-60 mm, so a demand for more than 140 mm of shift is not a
   correction, it is a contradiction.

And one positive rule: release DEEP.  Objects let go at rim height bounce off the rim and
roll out to the 5-point zone; 0.36 m down puts the free fall below bounce-out energy.
"""
from __future__ import annotations

import numpy as np

from course_mission.catalogue import BIN_OF


class Delivery:
    def __init__(self, cfg, binmem, arm_controller) -> None:
        d = cfg["deliver"]
        self.cfg = d
        self.mem = binmem
        self.arm = arm_controller
        self.mode = str(d.get("mode", "sighted"))
        self.recentre_max = float(d["recentre_max_m"])
        self.recentre_cap = float(d["recentre_step_cap_m"])
        self.recentre_min = float(d["recentre_min_m"])
        self.refuse_stale = bool(d["recentre_refuse_when_stale"])
        self.half_along = float(d["believe_half_along_m"])
        self.half_across = float(d["believe_half_across_m"])
        self.apron_along = float(d["apron_half_along_m"])
        self.apron_across = float(d["apron_half_across_m"])
        self.stand_near = float(d["stand_off_near_m"])
        self.stand_far = float(d["stand_off_far_m"])
        self.stop_lat = float(d["stop_lat_m"])
        self.stop_lat_relaxed = float(d["stop_lat_relaxed_m"])

    @staticmethod
    def bin_for(cls: str) -> str | None:
        return BIN_OF.get(cls)

    # ---- approach ---------------------------------------------------------------------
    def approach_command(self, colour: str, odo, step: int, elapsed_frac: float = 0.0):
        """-> (vx, vy, wz), arrived, reason.

        `None` for the bin position is a real answer: it means walk on and look, not guess.
        """
        rel = self.mem.rel(colour, odo, step)
        if rel is None:
            return np.zeros(3), False, "no fresh sighting"
        dx = float(rel[0]) - 0.5 * (self.stand_near + self.stand_far)
        dy = float(rel[1])
        tol = self.stop_lat_relaxed if elapsed_frac > 0.6 else self.stop_lat
        if self.stand_near < float(rel[0]) < self.stand_far and abs(dy) < tol:
            return np.zeros(3), True, "arrived"
        vx = float(np.clip(0.6 * dx, -0.32, 0.5))
        vy = float(np.clip(0.8 * dy, -0.25, 0.25))
        wz = float(np.clip(1.2 * np.arctan2(dy, max(rel[0], 0.2)), -0.6, 0.6))
        return np.array([vx, vy, wz]), False, "approaching"

    # ---- re-centring over the opening --------------------------------------------------
    def recentre(self, colour: str, tcp_b, odo, step: int) -> tuple[np.ndarray, str]:
        """-> (shift to apply in the base frame, refusal reason or "").

        Under honest operation the thing centred is the TCP, not the load: where the object
        hangs inside the jaw is not observable, so centre what is.
        """
        rel = self.mem.rel(colour, odo, step)
        if rel is None:
            return np.zeros(2), "stale" if self.refuse_stale else ""
        want = np.asarray(rel, float)[:2] - np.asarray(tcp_b, float)[:2]
        if float(np.max(np.abs(want))) > self.recentre_max:
            return np.zeros(2), "disagree"
        shift = np.clip(want, -self.recentre_cap, self.recentre_cap)
        if float(np.linalg.norm(shift)) < self.recentre_min:
            return np.zeros(2), ""
        return shift, ""

    # ---- the self-verdict ----------------------------------------------------------------
    def believed(self, colour: str, tcp_b, odo, step: int) -> bool:
        """Did the jaw open with the TCP inside the sighted bin's half-opening?

        This is the deployable stand-in for the judge's landing classification.  When there
        is no fresh sighting it returns True -- not out of optimism, but because the robot
        has no evidence either way and must not spend the object's quota slot on a guess
        that it failed.
        """
        rel = self.mem.rel(colour, odo, step)
        if rel is None:
            return True
        d = np.asarray(rel, float)[:2] - np.asarray(tcp_b, float)[:2]
        return bool(abs(d[0]) < self.half_along and abs(d[1]) < self.half_across)

    # ---- the 25 % fallback -----------------------------------------------------------------
    def in_apron(self, rel_to_bin_row) -> bool:
        """Walk there FIRST, then test, then release.  A failed bin approach tends to leave
        the robot exactly outside the zone, and anything done out of bounds is void."""
        p = np.asarray(rel_to_bin_row, float)
        return bool(abs(p[0]) < self.apron_along and abs(p[1]) < self.apron_across)
