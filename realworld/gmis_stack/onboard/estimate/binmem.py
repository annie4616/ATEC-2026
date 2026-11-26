"""Where the bins are, remembered in the BODY frame.

Four of the five objects the simulator ever managed to grasp under honest conditions were
lost at delivery -- released, but not into a bin.  This module is the piece that addresses
that, and its design is one measurement:

    over 20 scored drops, every drop whose bin sighting was younger than ~2,600 steps landed
    in the bin (10/10) and none of the older ones did (0/7).  The drift ramps about 7 mm per
    100 steps, and the robot stands at a bin for 1,500-3,000 steps per unload -- so the
    SECOND drop of a stop is always the one that goes wrong.

Hence the sighting is stored **body-relative, carried on raw odometry**, and expires on
BOTH age and distance travelled:

  * age alone is not enough.  One run's 1,093-step-old sighting passed the age gate, had
    already failed the distance gate, and the "fresh" correction it authorised pushed the
    drop 0.26 m out.
  * distance alone is not enough either.  A legitimate walk-in spends about 4 m (the sighting
    window is 2.4-3.5 m out, the drop stand-off is 0.55 m, and stepping between bins backs
    off ~1.1 m), so a 3 m gate cut every drop of one trip.

`rel(colour) is None` is therefore the ONLY staleness test any caller should use.  Age on its
own is available for logging and for nothing else.

Pure numpy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _rot2(a: float, v) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    v = np.asarray(v, float)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


@dataclass
class Sighting:
    step: int
    odo: np.ndarray          # raw odometry pose (x, y, yaw) at the moment of the sighting
    rel: np.ndarray          # bin centre in the body frame at that moment
    rng: float               # slant range of the sighting


class BinMemory:
    """One sighting per colour, dead-reckoned forward through uncorrected odometry.

    Uncorrected on purpose: a belt fix arrives as a jump in the corrected estimate, and a
    jump between the sighting and the drop is exactly the error we are trying not to make.
    The S point of THIS sighting minus the S pose of THIS instant is exact; the odometry
    carries it without any corridor fit in between.
    """

    def __init__(self, cfg) -> None:
        d = cfg["deliver"]
        self.max_age = int(d["bin_max_age_steps"])
        self.max_move = float(d["bin_max_move_m"])
        self.near_range = float(d["bin_near_range_m"])
        self._obs: dict[str, Sighting] = {}
        self.n_fix = 0
        self.n_far = 0

    # ---- writing ----------------------------------------------------------------------
    def observe(self, bins: dict, odo, step: int) -> int:
        """`detect_bins` output + the current raw odometry pose -> stored sightings.

        A FAR sighting is a tangent-line projection, not a position: a 7 m ray across a
        curving course lands 1.0-1.2 m off the centreline.  It is kept only where nothing
        nearer is remembered, and it is never allowed to correct anything.
        """
        if not bins:
            return 0
        odo = np.asarray(odo, float)
        n = 0
        for colour, p in (bins.get("colors") or {}).items():
            rng = float((bins.get("ranges") or {}).get(colour, 9.9))
            far = rng > self.near_range
            if far:
                self.n_far += 1
                if colour in self._obs:
                    continue           # never overwrite a near memory with a far one
            # `p` is already in the frame the camera pose was given in; express it body-side.
            rel = _rot2(-float(odo[2]), np.asarray(p, float)[:2] - odo[:2])
            self._obs[colour] = Sighting(step=step, odo=odo.copy(), rel=rel, rng=rng)
            self.n_fix += 1
            n += 1
        return n

    # ---- reading ----------------------------------------------------------------------
    def rel(self, colour: str, odo, step: int) -> np.ndarray | None:
        """Bin centre in the CURRENT body frame, or None if the memory is stale.

        None is the answer callers must branch on.  Returning a stale position "with low
        confidence" is what produced the coin-flip corrections.
        """
        s = self._obs.get(colour)
        if s is None:
            return None
        if (step - s.step) > self.max_age:
            return None
        odo = np.asarray(odo, float)
        if float(np.linalg.norm(odo[:2] - s.odo[:2])) > self.max_move:
            return None
        # Carry the body-frame vector forward: rotate into world at the sighting, subtract
        # the travel, rotate into the body frame now.
        p_w = s.odo[:2] + _rot2(float(s.odo[2]), s.rel)
        return _rot2(-float(odo[2]), p_w - odo[:2])

    def age(self, colour: str, step: int) -> int:
        s = self._obs.get(colour)
        return 10 ** 9 if s is None else int(step - s.step)

    def moved(self, colour: str, odo) -> float:
        s = self._obs.get(colour)
        if s is None:
            return float("inf")
        return float(np.linalg.norm(np.asarray(odo, float)[:2] - s.odo[:2]))

    def seen(self) -> list[str]:
        return sorted(self._obs)

    def report(self, odo=None, step: int = 0) -> dict:
        out = {"fixes": self.n_fix, "far_rejected": self.n_far, "colours": {}}
        for c, s in self._obs.items():
            out["colours"][c] = {"age": int(step - s.step), "rng": round(s.rng, 2),
                                 "moved": (None if odo is None
                                           else round(self.moved(c, odo), 2)),
                                 "fresh": odo is not None and self.rel(c, odo, step) is not None}
        return out
