"""The attempt ledger: what happened, and where it died.

Half of this work package is this file, for a reason worth stating.  The simulator's honest
runs failed 258 of 265 grasp attempts, and it was possible to find out WHY only because the
runs logged enough to reconstruct it afterwards -- the aiming error against ground truth, the
TCP residual at closure, the gripper width and effort at the end.  At a venue there is no
ground truth and there are no second runs.

So every quantity that was used to diagnose those runs has a deployable substitute here, and
the one that carries the most is `reobs_delta`: the distance between the estimate the robot
approached on and the estimate it re-measures from the crouch, immediately before committing
the arm.  Ground truth is unavailable but *disagreement between two independent looks* is
not, and the recorded failure mode -- a third of attempts aimed at a different article -- is
exactly the kind of error that shows up as disagreement.

The ledger is written continuously, so a run that ends in a crash or a judge's whistle still
leaves its evidence.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Attempt:
    """One walk-up-and-try.  An object that needs retries leaves several of these."""
    t: float
    step: int
    track_id: str
    cls: str

    # --- how much the robot should have trusted its target -----------------------------
    n_views: int = 0
    best_range_m: float = 0.0
    view_spread_m: float = 0.0
    reobs_delta_m: float | None = None     # approach estimate vs crouch re-observation
    bank_drift_m: float | None = None

    # --- what the arm did ----------------------------------------------------------------
    path: str = ""                          # rl | scripted_ik
    grasp_steps: int = 0
    tcp_residual_m: float | None = None     # closest the TCP came to its target
    abort: str = ""                         # stall | diverge | hover | probes | timeout
    probes_used: int = 0

    # --- what the fingers felt -----------------------------------------------------------
    gw_cmd_m: float | None = None
    gw_end_m: float | None = None
    effort_end: float | None = None
    lift_m: float | None = None
    loaded_steps: int = 0

    # --- delivery --------------------------------------------------------------------------
    bin_colour: str = ""
    bin_obs_age: int | None = None
    bin_moved_m: float | None = None
    recentre_m: float | None = None
    recentre_refused: str = ""              # "" | stale | disagree
    released_believed: bool = False

    # --- outcome ---------------------------------------------------------------------------
    stage: str = "started"
    note: str = ""

    def close(self, stage: str, note: str = "") -> "Attempt":
        self.stage = stage
        if note:
            self.note = note
        return self


# The stages an attempt can die at, in pipeline order.  Having this list explicit is what
# lets the summary say "12 attempts, 9 of them never got a usable target" instead of "12
# attempts failed".
STAGES = (
    "no_target",          # nothing engageable in range
    "young_track",        # seen once; not trusted
    "ghost",              # went to look twice, found nothing -- written off
    "approach_timeout",   # never reached the aim point
    "unstable_crouch",    # terrain would not hold the squat
    "out_of_bank",        # arrived, but outside the pre-grasp bank: approach never converged
    "arrived_empty",      # closed on nothing -- the aim was wrong
    "slipped",            # closed on it and lost it
    "carry_lost",         # held it, dropped it on the walk
    "bin_unreachable",    # could not get to the bin
    "apron",              # placed in the 25 % zone
    "released_believed",  # opened with the TCP inside the sighted bin
)


class Ledger:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.attempts: list[Attempt] = []
        self._open: Attempt | None = None
        self.t0 = time.monotonic()

    def start(self, track, step: int) -> Attempt:
        a = Attempt(t=round(time.monotonic() - self.t0, 2), step=step,
                    track_id=track.tid, cls=track.cls,
                    n_views=track.n_views, best_range_m=round(track.best_r, 3),
                    view_spread_m=round(track.spread, 3))
        self.attempts.append(a)
        self._open = a
        return a

    @property
    def current(self) -> Attempt | None:
        return self._open

    def close(self, stage: str, note: str = "") -> None:
        if self._open is not None:
            self._open.close(stage, note)
            self._open = None
            self.flush()

    def flush(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.report(), indent=2, default=float) + "\n")

    # ---- the summary that gets read at the venue -----------------------------------------
    def summary(self) -> dict:
        by_stage: dict[str, int] = {}
        by_cls: dict[str, dict[str, int]] = {}
        for a in self.attempts:
            by_stage[a.stage] = by_stage.get(a.stage, 0) + 1
            d = by_cls.setdefault(a.cls, {"attempts": 0, "held": 0, "delivered": 0})
            d["attempts"] += 1
            d["held"] += int(a.lift_m is not None and a.lift_m > 0.02)
            d["delivered"] += int(a.released_believed)
        reobs = [a.reobs_delta_m for a in self.attempts if a.reobs_delta_m is not None]
        return {
            "attempts": len(self.attempts),
            "by_stage": dict(sorted(by_stage.items(), key=lambda kv: -kv[1])),
            "by_class": by_cls,
            "believed_delivered": sum(1 for a in self.attempts if a.released_believed),
            # The venue-side stand-in for "how often were we aimed at the wrong article".
            "reobs_delta_median_m": (round(sorted(reobs)[len(reobs) // 2], 3)
                                     if reobs else None),
            "reobs_over_0p3m": sum(1 for r in reobs if r > 0.30),
        }

    def report(self) -> dict:
        return {"summary": self.summary(),
                "stages": list(STAGES),
                "attempts": [asdict(a) for a in self.attempts]}

    def render(self) -> str:
        s = self.summary()
        out = [f"  attempts {s['attempts']}   believed delivered {s['believed_delivered']}"]
        for k, v in s["by_stage"].items():
            out.append(f"    {k:20s} {v}")
        for c, d in sorted(s["by_class"].items()):
            out.append(f"    {c:10s} tried {d['attempts']:2d}  held {d['held']:2d}  "
                       f"delivered {d['delivered']:2d}")
        if s["reobs_delta_median_m"] is not None:
            out.append(f"    re-observation disagreement: median "
                       f"{s['reobs_delta_median_m']*1000:.0f} mm, "
                       f"{s['reobs_over_0p3m']} attempts over 300 mm")
        return "\n".join(out)
