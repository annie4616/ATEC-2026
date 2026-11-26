"""Object identity from the detector's own history.

The simulator answered "which article is this blob?" by comparing against the object's TRUE
world position and its TRUE class, and threw away any detection that failed to match.  Two
things were wrong with that, and both were measured:

  * It was an ORACLE on the default path.  At step 68,500 of the 120-point run, 144 of the
    476 detections that reached the gate -- 30.3 % -- were killed by it.  And because the
    test included true-class equality, a misclassification could never be engaged at all,
    while scoring is BY CLASS.
  * It does not exist at a venue.

So identity has to come from the robot's own history: every detection lands somewhere, on
the nearest same-class track inside a gate or on a new track of its own.

Three details carry almost all of the quality, and each replaced something simpler that was
measured to be worse:

1. **The gate grows with range.**  A flat gate cannot be right at both 0.5 m and 3 m.
2. **The position is the median of the three NEAREST-RANGE views**, not of all of them.
   A flat median-of-views track sat a median 0.96 m from the article it belonged to and only
   39 % landed within 0.30 m -- which is why the grasp "arrived on its target and closed on
   nothing" while the servo residual read 7-11 mm.  A 0.6 m view is worth more than twenty
   taken from 3 m.
3. **A track seen once is not a track.**  With the simulator's false-positive filter gone, a
   texture artefact becomes a track like any other; requiring two views is what replaces it.

And a track the robot cannot find twice is written off as a ghost, because "keep the old
estimate" sends it back to an empty patch of deck for ever -- one run wandered from lat
+1.79 to -1.79 chasing one, took two out-of-bounds penalties and left the course.

numpy only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Track:
    tid: str
    cls: str
    views: list[tuple[np.ndarray, float]] = field(default_factory=list)
    best_r: float = 9.9
    last_seen: int = 0
    first_seen: int = 0
    lost_looks: int = 0
    attempts: int = 0
    ghost: bool = False

    # ---- position ---------------------------------------------------------------------
    @property
    def p(self) -> np.ndarray:
        """Median of the three nearest-range views.  See the module docstring: this is the
        difference between a track that points at the article and one that points 0.96 m
        away from it."""
        near = sorted(self.views, key=lambda v: v[1])[:3]
        return np.median(np.stack([v[0] for v in near]), axis=0)

    @property
    def n_views(self) -> int:
        return len(self.views)

    @property
    def spread(self) -> float:
        """How far apart the kept views are.  Large spread = the track is a mixture of two
        articles, or of an article and an artefact.  This is a confidence signal available
        WITHOUT ground truth, which is the only kind that helps at a venue."""
        if len(self.views) < 2:
            return 0.0
        pts = np.stack([v[0][:2] for v in self.views])
        return float(np.max(np.linalg.norm(pts - pts.mean(axis=0), axis=1)))

    def add(self, pos: np.ndarray, rng: float, step: int) -> None:
        self.views.append((np.asarray(pos, float).copy(), float(rng)))
        # Keep the ten nearest-range views; range is the quality axis, not recency.
        self.views = sorted(self.views, key=lambda v: v[1])[:10]
        self.best_r = min(self.best_r, float(rng))
        self.last_seen = step


class ObjectTracker:
    def __init__(self, cfg) -> None:
        t = cfg["track"]
        self.gate0 = float(t["gate_m"])
        self.gate_per_m = float(t["gate_per_m"])
        self.min_views = int(t["min_views"])
        self.lost_looks_max = int(t["lost_looks"])
        self.blacklist_radius = float(t["blacklist_radius_m"])
        self.tracks: dict[str, Track] = {}
        self.blacklist: list[np.ndarray] = []
        self.perma_blacklist: list[np.ndarray] = []
        self._n = 0
        self.n_assoc = 0
        self.n_new = 0

    # ---- association ------------------------------------------------------------------
    def observe(self, det: dict, step: int) -> Track:
        """One detection -> the track it belongs to.  Never returns None: an unmatched
        detection becomes its own track rather than being discarded."""
        cls = det["cls"]
        pos = np.asarray(det["pos_w"], float)
        rng = float(det.get("dist", 0.0) or 0.0)

        best, best_d = None, 1e9
        for tr in self.tracks.values():
            if tr.cls != cls or tr.ghost:
                continue
            d = float(np.linalg.norm(tr.p[:2] - pos[:2]))
            gate = self.gate0 + self.gate_per_m * max(rng, tr.best_r)
            if d < gate and d < best_d:
                best, best_d = tr, d

        if best is None:
            self._n += 1
            best = Track(tid=f"trk{self._n:03d}", cls=cls, first_seen=step)
            self.tracks[best.tid] = best
            self.n_new += 1
        else:
            self.n_assoc += 1
        best.add(pos, rng, step)
        best.lost_looks = 0
        return best

    def observe_all(self, dets: list[dict], step: int) -> list[Track]:
        return [self.observe(d, step) for d in dets or []]

    # ---- selection --------------------------------------------------------------------
    def engageable(self, tr: Track) -> bool:
        """A track the mission is allowed to walk to.

        `min_views` is the whole false-positive defence now that the simulator's identity
        table is gone.  One run's first sweep produced three lunchbox tracks, two of them
        1.10 and 1.62 m from any real lunchbox, on a single frame each.
        """
        return (not tr.ghost
                and tr.n_views >= self.min_views
                and not self.blacklisted(tr.p))

    def blacklisted(self, pos) -> bool:
        p = np.asarray(pos, float)[:2]
        for b in (*self.blacklist, *self.perma_blacklist):
            if float(np.linalg.norm(b - p)) < self.blacklist_radius:
                return True
        return False

    def blacklist_spot(self, pos, perma: bool = False) -> None:
        p = np.asarray(pos, float)[:2].copy()
        (self.perma_blacklist if perma else self.blacklist).append(p)

    # ---- ghosts -----------------------------------------------------------------------
    def lost_look(self, tr: Track) -> bool:
        """The robot went to look and found nothing of that class.  -> True once written off.

        Two failed looks, not one: a single miss is a bad viewing angle, two is an empty
        patch of deck.  The write-off also blacklists the spot, so the search does not
        rediscover the same artefact from a different stance ten seconds later.
        """
        tr.lost_looks += 1
        if tr.lost_looks >= self.lost_looks_max:
            tr.ghost = True
            self.blacklist_spot(tr.p, perma=True)
            return True
        return False

    def forget(self, tr: Track, perma: bool = False) -> None:
        tr.ghost = True
        self.blacklist_spot(tr.p, perma=perma)

    # ---- reporting --------------------------------------------------------------------
    def report(self) -> dict:
        live = [t for t in self.tracks.values() if not t.ghost]
        return {
            "tracks": len(self.tracks),
            "live": len(live),
            "ghosts": sum(1 for t in self.tracks.values() if t.ghost),
            "engageable": sum(1 for t in live if self.engageable(t)),
            "associations": self.n_assoc,
            "new_tracks": self.n_new,
            "blacklist": len(self.blacklist) + len(self.perma_blacklist),
            "by_class": {c: sum(1 for t in live if t.cls == c)
                         for c in sorted({t.cls for t in live})},
        }


def distance_key(tr: Track, base_xy, cls_rank: dict, picked: dict, terrain_rank: int = 0):
    """The mission's target ordering, ported verbatim from the simulator.

    Lexicographic (class rank + already-picked-of-this-class, terrain rank, distance).  The
    class term means a class that has already been collected loses priority to one that has
    not, which is what stops the robot spending an entire run on the easiest article.
    """
    d = float(np.linalg.norm(tr.p[:2] - np.asarray(base_xy, float)[:2]))
    return (cls_rank.get(tr.cls, 0) + picked.get(tr.cls, 0), terrain_rank, d)
