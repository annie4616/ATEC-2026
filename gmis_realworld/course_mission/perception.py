"""Head-camera detector — numpy only, no torch and no Isaac.

Lives apart from mission.py so scripts/tune_detector.py can score a change against
recorded frames in about a second instead of spending a GPU slot and ten minutes on a
simulator run. mission.py re-exports Perception, so importers do not need to care.
"""

from __future__ import annotations

import os

import numpy as np

from course_mission.catalogue import OBJ_DIMS, OBJ_RGB
from course_mission.deploy import CAM_DEPTH_ABS, CAM_DEPTH_REL, CAM_RGB_SIGMA, REAL_NOISE


class Perception:
    """Head-camera detector: nearest-prototype colour classification with the TERRAIN
    included as background classes, then blob geometry to reject what is left.

    Three measured facts drove this design (see scripts/calibrate_colors.py,
    scripts/debug_detect.py and the offline frame analysis in logs/frame.npz):
      1. rendering desaturates everything, so absolute colour bands miss objects at range
         (a can is S 0.52 up close but blends into the tan mud, which sits at H37 S0.31);
      2. the unprojected HEIGHT is only good to about +-8 cm at 0.4-3 m, so a
         "stands proud of the ground" test cannot separate a 3 cm rind from the ground —
         an early version of this class used one and detected nothing at all;
      3. the horizontal position and extent ARE accurate, so span and lateral offset are
         reliable discriminators — which is what separates the red boundary strips (long,
         at |lateral| ~ 1.8 m) from the identically-coloured red paper bag.

    So: classify every pixel against object AND terrain prototypes in (hue, saturation),
    keep only object-labelled pixels, cluster, and reject clusters wider than a
    competition object. The caller additionally drops anything outside the course
    corridor.
    """

    # Measured on rendered head-camera frames: (hue deg, saturation).
    #
    # TWO fitted sets, because there are two worlds. MISSION_VENUE_PHOTO=1 dresses the
    # course in photographs of the real props and ground (course_sim/image/); without it
    # everything is a flat vertex colour. A prototype fitted to one is wrong for the
    # other, and not by a little: shipping the photo world against the flat prototypes
    # measured 45 % precision / 43 % recall, against 57 / 71 for the flat world it
    # replaced. Both sets come from scripts/fit_prototypes*.py, which samples object
    # pixels by PROJECTING labelled ground-truth positions rather than by colour
    # threshold -- otherwise the fit assumes the answer it is measuring.
    #
    #   photo world, dataset_v5 (643 frames, deployed noise): 59 % precision / 76 % recall
    #   flat  world, dataset_rind (300 frames, same noise):    57 % / 71 %
    #
    # so dressing the course costs nothing measurable once the prototypes follow it.
    VENUE_PHOTO = os.environ.get("MISSION_VENUE_PHOTO") == "1"

    if VENUE_PHOTO:
        # Objects are single-material and keep ONE prototype each -- except the rind,
        # which is physically two-faced (striped green skin / red flesh) and needs one per
        # face. Fitting objects as k-means clusters was tried and is worse: the can's
        # shaded second cluster (48.1, 0.51) sits in bright grass and took can precision
        # from 71 % to 30 %.
        OBJ_PROTO = {"can": (48.0, 0.85), "rind": (116.0, 0.77),
                     "rind_flesh": (352.0, 0.79), "redbag": (342.0, 0.38),
                     "purplebag": (242.0, 0.51), "bottle": (52.0, 0.59)}
        # TERRAIN is the opposite case: a photograph is a distribution, not a colour, and
        # collapsing it to its median throws away exactly the tail that causes false
        # positives. Real grass reaches S 0.7, which is nearer the watermelon skin than
        # the grass median -- that alone produced 306 rind reports of which 22 were real.
        # Four k-means clusters per surface (scripts/fit_prototypes_km.py) cut that to 56
        # reports at 39 % good. "sky" is sampled where depth is infinite; taking the top
        # image rows instead sampled the green recycling bins.
        BG_PROTO = {
            "grass_0": (75.1, 0.34), "grass_1": (79.3, 0.49),
            "grass_2": (218.9, 0.02), "grass_3": (354.6, 0.12),
            "mixed_0": (79.2, 0.58), "mixed_1": (86.2, 0.11),
            "mixed_2": (240.1, 0.02), "mixed_3": (193.0, 0.04),
            "mud_0": (39.4, 0.11), "mud_1": (37.4, 0.26),
            "mud_2": (218.2, 0.12), "mud_3": (338.1, 0.34),
            "pebble_0": (35.4, 0.06), "pebble_1": (78.7, 0.37),
            "pebble_2": (35.2, 0.22), "pebble_3": (222.3, 0.07),
            "sky_0": (212.7, 0.14), "sky_1": (208.0, 0.21),
            "sky_2": (209.1, 0.30), "sky_3": (212.4, 0.46),
            # The recycling BINS are scenery, not trash, and the camera stares at them
            # from 0.4 m on every unload. The green one measures (140, 0.73) -- 24 degrees
            # from the photo-fitted watermelon skin (116, 0.77) at the same saturation --
            # so with no prototype of its own it went to the nearest object class: 171 of
            # 179 rind reports in dataset_v5 were the green bin. The flat-colour rind
            # prototype (137, 0.34) was nearly the same hue but half the saturation, which
            # is why this never showed up before.
            # Refreshed 2026-08-26 for the REAL bin colour (photo-measured green bin:
            # hue 137.5, sat 0.60 — the old sim green was hue 133/sat 0.82 and these
            # entries were fitted to renders of it). Two entries keep the fitted spread
            # around the measured centre; the (151, 0.41) shading entry survives scaled.
            "bin_0": (137.5, 0.60), "bin_1": (148.0, 0.38),
            "bin_2": (135.0, 0.66), "bin_3": (140.5, 0.50),
            # red/blue bins at their REAL colours. The red bin (2.5, 0.73) is otherwise
            # nearest to the redbag prototype (0.0, ~0.5) — the exact green-bin-as-rind
            # failure mode, one bin over. The blue bin is added for symmetry/margin.
            "bin_red": (2.5, 0.73), "bin_blue": (217.5, 0.68),
        }
    else:
        OBJ_PROTO = {"can": (58.0, 0.52), "rind": (137.0, 0.34),
                     # FLESH-UP RINDS (0908, dataset_0908): the venue rind prop lands
                     # flesh-side-up after the drop-spawn, and its visible face
                     # measures H 349-350 / S 0.59-0.76 — nowhere near the green skin
                     # prototype. Without this entry real rinds were INVISIBLE (75
                     # reports, 0 within 0.45 m; every mission "rind" track was a
                     # terrain false positive — the ghost-track plague). With it:
                     # rind 67% good, overall recall 69% -> 88% on the same frames.
                     "rind_flesh": (350.0, 0.70), "redbag": (0.0, 0.50),
                     "purplebag": (293.0, 0.35), "bottle": (205.0, 0.15)}
        BG_PROTO = {"mud": (37.0, 0.31), "grass": (105.0, 0.20),
                    "sky_pebble": (207.0, 0.22),
                    # real bin colours (2026-08-26): without these, a green-bin pixel
                    # (137.5, 0.60) lands on the flat rind prototype (137.0, 0.34) and a
                    # red-bin pixel (2.5, 0.73) on redbag (0.0, 0.50) — the measured
                    # 179-report bin-as-rind incident, reintroduced by realistic colours.
                    "bin_green": (137.5, 0.60), "bin_red": (2.5, 0.73),
                    "bin_blue": (217.5, 0.68)}

    # A prototype is a COLOUR CLUSTER, not a class: any extra key above is reported under
    # its aliased class, so downstream code (BIN_OF, OBJ_DIMS, grasping) never sees it.
    PROTO_ALIAS = {"rind_flesh": "rind"}
    H_SCALE, S_SCALE = 20.0, 0.15
    # the lunchbox is achromatic; so are the pebble bed and the stair treads, which are
    # brighter (V 0.87+) than the box (V 0.67-0.75).
    # The photographed bagasse clamshell measures S 0.120 / V 0.776, so the original 0.10
    # cut half its pixels loose to the nearest chromatic prototype -- purplebag, 30 degrees
    # away in a hue that means nothing at this saturation. 0.16 clears the lunchbox and
    # still excludes mud (S 0.176); the V window keeps out stair treads (0.918), grass
    # (0.843), mud (0.831) and sky (0.831). Swept 0.10/0.12/0.14/0.16 and the V ceiling
    # 0.78/0.82/0.86: overall precision moved 56-59 % across the whole sweep, so this is a
    # plateau, not a peak.
    LB_SAT, LB_VAL = (0.16, (0.52, 0.82)) if VENUE_PHOTO else (0.10, (0.52, 0.82))
    LB_MAX_SPAN = 0.25
    MIN_PIX = 25
    # --- rind-specific gates -----------------------------------------------------------
    # The rind prototype (137.0, 0.34) sits in the widest empty hue arc of the whole map
    # (grass 105 -> sky_pebble 207), so it inherits every pixel that 4-LSB RGB noise
    # randomises off near-achromatic terrain — pebble measures H 240 / S 0.005. None of
    # the shared thresholds could see this, because scripts/tune_detector.py scored the
    # detector with add_noise=False while the mission has always run it with noise ON.
    CELL_FILL = {"rind": 20}          # of cell*cell = 144 px, before a cell may merge
    # Per-class blob span caps. The rind cap was 0.15 for a 90 mm rind; the venue
    # prop is 200 mm (guide 289) whose p10-p90 back-projected extent lands at ~0.16,
    # AT or PAST that cap -- i.e. the gate that exists to reject terrain would have
    # rejected every real rind instead, silently, before it was ever reported.
    # 0.24 = 1.2x the true length, the same ratio 0.15 had against 90 mm... which it
    # did not: 0.15/0.09 = 1.67. Keeping that ratio would need 0.33 and would swallow
    # MAX_SPAN entirely, so 0.24 is the deliberate compromise and the number to
    # revisit if rind recall is short.
    CLS_MAX_SPAN = {"lunchbox": 0.25,
                    "rind": 0.15 if os.environ.get("COURSE_VENUE", "hk") == "legacy"
                    else 0.24}
    # true-rind blob median sat: photo world 0.286-0.380, flesh-up default world
    # 0.59-0.76; terrain 0.08. The default world can afford 0.34 (kills 79% of the
    # terrain-as-rind reports, dataset_0908) — the photo world cannot (would cut its
    # own rinds at 0.286-0.34), so the gate is branch-aware.
    MIN_BLOB_SAT = {"rind": 0.24 if VENUE_PHOTO else 0.34}
    # per-class minimum measured height above ground (flat-paint veto, see the emit
    # site): redbag vs the red start/finish lines. Applied only when h_est is finite.
    CLS_MIN_H = {"redbag": 0.05}
    # Rind floor RETUNED 0826 on the probe_photo_range frames: the photo-world rind
    # subtends only 14-45% of the expected face area (true blobs of 1866/549/195/90 px
    # at 0.9/1.3/1.7/2.1 m all sat DEAD ON the truth position and were all killed by
    # the old 0.5 floor -> rind 0/16). 0.12 recovers 16/16 across the ladder at a cost
    # of +4 rind false reports over 64 frame-draws — noise the registry match and the
    # class quota already absorb.
    AREA_LO = {"rind": 0.12}
    NEAR_CUT = 0.40      # closer than this is the robot's own arm/gripper
    # MEASURED on 200 labelled frames (scripts/tune_detector.py) once the camera pose
    # was fixed: the detector is accurate from 1.0-2.5 m (median position error
    # 0.06-0.10 m) and hopeless beyond 2.5 m (median 2.6 m). Capping the range at
    # 2.2 m keeps recall at 88% while halving the median error (1.47 -> 0.69 m).
    FAR_CUT = 2.2
    # Span is measured p10-p90, NOT min-max: pixels on an object's silhouette interpolate
    # depth between the object and the ground behind it, and those few "flying pixels"
    # inflate a min-max extent to 0.6 m for a 0.13 m bag. Measured on real frames the
    # p10-p90 extent is 0.04-0.12 m for the competition objects and 0.42-0.53 m for
    # terrain / boundary-strip blobs, so 0.30 separates them cleanly.
    MAX_SPAN = 0.30
    SPAN_PCT = (10, 90)

    def __init__(self, cam, device, mount=None):
        # `mount` (HeadCamMount) is the source of truth for the camera pose: the sensor's
        # own cam.data.pos_w never updates for a root-level camera we drive ourselves.
        self.cam, self.device, self.mount = cam, device, mount

    @staticmethod
    def _rgb_to_hsv(img: np.ndarray):
        a = img.astype(np.float32) / 255.0
        mx = a.max(axis=-1)
        mn = a.min(axis=-1)
        df = mx - mn + 1e-8
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
        h = np.zeros_like(mx)
        m = mx == r
        h[m] = (60 * ((g - b) / df))[m]
        m = mx == g
        h[m] = (60 * ((b - r) / df) + 120)[m]
        m = mx == b
        h[m] = (60 * ((r - g) / df) + 240)[m]
        h = (h + 360) % 360
        s = df / (mx + 1e-8)
        return h, s, mx

    @classmethod
    def label_map(cls, h, s, v):
        """Per-pixel nearest prototype; background labels come back as None.

        Running-min over the ~22 prototypes: keeps only (H,W) work buffers instead of the
        (P,H,W) distance stack the argmin version allocated -- that stack was ~22x the frame
        in float64 and, on the N5095's weak memory bandwidth, the largest single cost in
        detect_arrays (label_map + its reduce/argmin were ~50% of the profile).  Each
        prototype's squared distance is compared to the running best with a STRICT `<`, so
        the first (lowest-index) prototype wins a tie -- exactly the sequential version's
        order (OBJ prototypes first, so an object label beats a background one on a tie).
        All buffers are float32.
        """
        keys = list(cls.OBJ_PROTO)
        h = np.asarray(h, np.float32)
        s = np.asarray(s, np.float32)
        hscl = np.float32(cls.H_SCALE)
        sscl = np.float32(cls.S_SCALE)

        best = np.full(h.shape, np.inf, np.float32)      # running min distance
        out = np.zeros(h.shape, np.int8)                 # 0 = background
        # OBJ prototypes -> label i+1, then BG prototypes -> label 0, in this order so the
        # strict `<` reproduces the original tie-break (OBJ wins over BG at equal distance).
        protos = ([(cls.OBJ_PROTO[k][0], cls.OBJ_PROTO[k][1], i + 1)
                   for i, k in enumerate(keys)]
                  + [(hh, ss, 0) for hh, ss in cls.BG_PROTO.values()])
        for hh, ss, lab in protos:
            dh = (h - np.float32(hh) + 180.0) % 360.0 - 180.0
            dh /= hscl
            dh *= dh                                     # (h-dist)^2, in place
            dsq = (s - np.float32(ss)) / sscl
            dsq *= dsq
            dh += dsq                                    # total squared distance, reuse dh
            m = dh < best
            best[m] = dh[m]
            out[m] = lab

        lb = (s < cls.LB_SAT) & (v > cls.LB_VAL[0]) & (v < cls.LB_VAL[1])
        out[lb] = len(keys) + 1
        return out, keys + ["lunchbox"]


    # ---- geometry-first detector -----------------------------------------------------
    # The colour-first design above was built on frames from a camera that had frozen at
    # the spawn point (see launch.head_cam_cfg), so its central claim — "the unprojected
    # height is only good to +-8 cm, a stands-proud-of-the-ground test cannot work" — was
    # measured on images of terrain tens of metres away. With the camera actually on the
    # robot the depth is good enough to segment on height, which is far more robust than
    # colour: it does not care how the renderer desaturates a can at 2 m.
    GROUND_CELL = 0.15      # m, world-xy cell for the local ground estimate
    GROUND_PCT = 10         # percentile of z within a cell taken as "ground"
    GROUND_NBR = 2          # cells; the neighbourhood must contain some bare ground
    MIN_HEIGHT = 0.030      # m above local ground to count as an object pixel
    MAX_HEIGHT = 0.40       # taller than any competition object -> terrain or scenery
    MAX_EXTENT = 0.34       # m, p10-p90 horizontal extent cap

    @classmethod
    def detect_height(cls, rgb, depth, K, cam_p, cam_rot, ground_z: float,
                      add_noise: bool | None = None):
        """Segment by height above the LOCAL ground, then classify the cluster by colour
        and size. Same return shape as detect_arrays()."""
        # the camera hands back (H, W, 1); recorded frames are already (H, W)
        depth = np.asarray(depth)
        depth = depth.reshape(depth.shape[0], depth.shape[1])
        if REAL_NOISE if add_noise is None else add_noise:
            rgb = np.clip(rgb.astype(np.float32)
                          + np.random.normal(0, CAM_RGB_SIGMA, rgb.shape), 0, 255)
            sig = CAM_DEPTH_ABS + CAM_DEPTH_REL * np.nan_to_num(depth, nan=0.0, posinf=0.0)
            depth = depth + np.random.normal(0, 1.0, depth.shape).astype(np.float32) * sig

        H, W = depth.shape
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        R = _as_rot(cam_rot)
        vv, uu = np.mgrid[0:H, 0:W]
        valid = np.isfinite(depth) & (depth > cls.NEAR_CUT) & (depth < cls.FAR_CUT)
        if valid.sum() < cls.MIN_PIX * 4:
            return []
        d = depth[valid]
        pts = np.stack([(uu[valid] - cx) / fx * d, (vv[valid] - cy) / fy * d, d], axis=1)
        pw = cam_p[None, :] + pts @ R.T

        # local ground: low percentile of z per world-xy cell, minimised over a
        # neighbourhood so a cell fully covered by an object still sees bare ground
        gi = np.floor(pw[:, 0] / cls.GROUND_CELL).astype(np.int64)
        gj = np.floor(pw[:, 1] / cls.GROUND_CELL).astype(np.int64)
        gi -= gi.min()
        gj -= gj.min()
        ni, nj = int(gi.max()) + 1, int(gj.max()) + 1
        if ni * nj > 4_000_000:
            return []
        flat = gi * nj + gj
        order = np.argsort(flat, kind="stable")
        fs, zs = flat[order], pw[order, 2]
        edges = np.flatnonzero(np.diff(fs)) + 1
        cell_z = np.full(ni * nj, np.inf, np.float32)
        for a, b in zip(np.r_[0, edges], np.r_[edges, len(fs)]):
            cell_z[fs[a]] = np.percentile(zs[a:b], cls.GROUND_PCT)
        grid = cell_z.reshape(ni, nj)
        k = cls.GROUND_NBR
        pad = np.pad(grid, k, constant_values=np.inf)
        local = np.full_like(grid, np.inf)
        for a in range(2 * k + 1):
            for b in range(2 * k + 1):
                local = np.minimum(local, pad[a:a + ni, b:b + nj])
        ground = local.reshape(-1)[flat]
        ground = np.where(np.isfinite(ground), ground, ground_z)
        above = pw[:, 2] - ground

        mask = np.zeros((H, W), bool)
        sel = (above > cls.MIN_HEIGHT) & (above < cls.MAX_HEIGHT)
        mask[vv[valid][sel], uu[valid][sel]] = True
        world = np.full((H, W, 3), np.nan, np.float32)
        world[vv[valid], uu[valid]] = pw
        hmap = np.full((H, W), np.nan, np.float32)
        hmap[vv[valid], uu[valid]] = above
        gmap = np.full((H, W), np.nan, np.float32)
        gmap[vv[valid], uu[valid]] = ground

        h, s, v = cls._rgb_to_hsv(rgb)
        out = []
        for vs, us in _blobs(mask, cls.MIN_PIX):
            wp = world[vs, us]
            if not np.isfinite(wp).any():
                continue
            plo, phi = np.nanpercentile(wp[:, :2], cls.SPAN_PCT, axis=0)
            extent = np.abs(phi - plo)
            span = float(np.linalg.norm(extent))
            top = float(np.nanpercentile(hmap[vs, us], 95))
            if not np.isfinite(span) or span > cls.MAX_EXTENT or top > cls.MAX_HEIGHT:
                continue
            cname = cls._classify(np.nanmedian(h[vs, us]), np.nanmedian(s[vs, us]),
                                  np.nanmedian(v[vs, us]), float(max(extent)), top)
            if cname is None:
                continue
            pos = np.array([np.nanmedian(wp[:, 0]), np.nanmedian(wp[:, 1]),
                            float(np.nanmedian(gmap[vs, us])) + top * 0.5])
            rng = float(np.nanmedian(depth[vs, us]))
            out.append({"cls": cname, "pos_w": pos.astype(float), "n_pix": int(len(vs)),
                        "dist": rng, "span": span, "above": top,
                        "hue": float(np.nanmedian(h[vs, us])),
                        "sat": float(np.nanmedian(s[vs, us]))})
        return out

    # colour of each class in (hue, sat, val) as the renderer shows it, plus the two
    # dimensions the silhouette can actually reveal (longest horizontal extent, height)
    CLS_APPEARANCE = {
        "can":       (58.0, 0.60, 0.75),
        "bottle":    (205.0, 0.20, 0.85),
        "rind":      (137.0, 0.45, 0.45),
        "lunchbox":  (40.0, 0.06, 0.70),
        "redbag":    (0.0, 0.60, 0.60),
        "purplebag": (293.0, 0.45, 0.45),
    }
    COLOUR_TOL = 2.0       # max normalised colour distance before a cluster is rejected

    @classmethod
    def _classify(cls, hue, sat, val, extent, height):
        """Nearest class in colour, with a size sanity check. None = not a competition
        object (terrain relief, the boundary strips, a bin)."""
        if not np.isfinite(hue) or not np.isfinite(sat):
            return None
        best, bestd = None, np.inf
        for name, (hh, ss, vvv) in cls.CLS_APPEARANCE.items():
            dh = abs((hue - hh + 180) % 360 - 180) / 30.0
            if ss < 0.12:                       # achromatic classes: hue is meaningless
                dh = 0.0
            d = dh ** 2 + ((sat - ss) / 0.22) ** 2 + ((val - vvv) / 0.30) ** 2
            dims = sorted(OBJ_DIMS[name], reverse=True)
            # a lying object shows its longest horizontal dimension; a standing one its
            # footprint — accept anything between the smallest and largest extent
            lo, hi = min(dims) * 0.5, max(dims) * 1.6
            if not (lo <= extent <= hi):
                d += 2.0
            if height > max(dims) * 1.6 + 0.05:
                d += 2.0
            if d < bestd:
                best, bestd = name, d
        return best if bestd < cls.COLOUR_TOL else None

    def detect(self, ground_z: float):
        """ground_z is kept for the caller's API (height is NOT used as a gate — see the
        class docstring). -> [{cls, pos_w(3,), n_pix, dist, span, above}]"""
        d = self.cam.data
        if self.mount is not None:
            if getattr(self.mount, "use_est", False):
                # honest navigation: the estimated extrinsics (mount offset + IMU
                # attitude + the S-frame estimate), so detections land in the
                # robot's own frame, never the simulator's
                cam_p, cam_R = self.mount.pos_est, self.mount.R_est
            else:
                cam_p, cam_R = self.mount.pos_w, self.mount.R_ros
        else:
            cam_p, cam_R = d.pos_w[0].cpu().numpy(), d.quat_w_ros[0].cpu().numpy()
        return self.detect_arrays(
            d.output["rgb"][0, ..., :3].cpu().numpy(),
            d.output["distance_to_image_plane"][0].cpu().numpy(),
            d.intrinsic_matrices[0].cpu().numpy(),
            cam_p, cam_R, ground_z)

    @classmethod
    def detect_arrays(cls, rgb, depth, K, cam_p, cam_rot, ground_z: float,
                      add_noise: bool | None = None):
        """The detector proper, on plain arrays, so scripts/tune_detector.py can score a
        change offline against recorded frames instead of burning a GPU slot per idea.
        detect() is a thin wrapper — there is only ever one copy of this logic."""
        self = cls
        # the camera hands back (H, W, 1); recorded frames are already (H, W)
        depth = np.asarray(depth)
        depth = depth.reshape(depth.shape[0], depth.shape[1])
        # OPTIONAL DOWNSCALE (DETECT_DOWNSCALE=N, default 1 = off).  detect_arrays is the
        # detector's dominant cost on the N100 (label_map + unproject + blobs all scale with
        # pixel count); an N:1 subsample cuts every one of them by N**2.  rgb/depth are
        # decimated by stride and K is scaled to match, so the unprojection stays correct;
        # object pixel counts fall by N**2, so MIN_PIX is scaled down too.  Nearest-stride
        # (not averaging) keeps depth valid/NaN semantics and colour prototypes intact.
        # A/B this against the full-res path before trusting it (recall of small/far objects).
        _ds = int(os.environ.get("DETECT_DOWNSCALE", "1"))
        if _ds > 1:
            rgb = rgb[::_ds, ::_ds]
            depth = depth[::_ds, ::_ds]
            K = K.copy()
            K[0, 0] /= _ds; K[1, 1] /= _ds; K[0, 2] /= _ds; K[1, 2] /= _ds
        if REAL_NOISE if add_noise is None else add_noise:
            # a real RGB-D module: sensor read/photon noise on colour, and a depth error
            # that grows with range (both stereo and structured light behave this way)
            rgb = np.clip(rgb.astype(np.float32)
                          + np.random.normal(0, CAM_RGB_SIGMA, rgb.shape), 0, 255)
            sig = CAM_DEPTH_ABS + CAM_DEPTH_REL * np.nan_to_num(depth, nan=0.0, posinf=0.0)
            depth = depth + np.random.normal(0, 1.0, depth.shape).astype(np.float32) * sig
        H, W = depth.shape
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        R = _as_rot(cam_rot)

        vv, uu = np.mgrid[0:H, 0:W]
        valid = np.isfinite(depth) & (depth > self.NEAR_CUT) & (depth < self.FAR_CUT)
        if valid.sum() < self.MIN_PIX:
            return []
        d = depth[valid]
        pts = np.stack([(uu[valid] - cx) / fx * d, (vv[valid] - cy) / fy * d, d], axis=1)
        pw = cam_p[None, :] + pts @ R.T
        world = np.full((H, W, 3), np.nan, np.float32)
        world[vv[valid], uu[valid]] = pw

        h, s, v = self._rgb_to_hsv(rgb)
        lab, keys = self.label_map(h, s, v)
        # HEIGHT CARVE for the achromatic (lunchbox) mask. The user's 0826 photo
        # terrain sits INSIDE the S/V window that used to exclude it (new mud S 0.18
        # med / pebble S 0.06 vs the box's 0.05-0.13; V overlaps fully), so entire
        # sections labelled as lunchbox and the real box drowned in one 80k-px blob —
        # measured 0/4 at every range on mud AND pebble while colour was 4/4 on cans.
        # Colour cannot separate them; HEIGHT can: the box top reads +0.035..+0.084
        # above ground, terrain +-0.02. Ground is self-estimated from the frame (median
        # world-z of the lower half) so a walking robot's whole-frame depth bias
        # cancels instead of shifting a fixed threshold.
        if "lunchbox" in keys:
            _wz = world[:, :, 2]
            _low = _wz[H // 2:, :]
            _gest = np.nanmedian(_low) if np.isfinite(_low).any() else ground_z
            _li = keys.index("lunchbox") + 1
            lab[(lab == _li) & ~(_wz - _gest > 0.025)] = 0
        out = []
        for i, cname in enumerate(keys):
            m = (lab == i + 1) & valid
            cname = self.PROTO_ALIAS.get(cname, cname)
            if m.sum() < self.MIN_PIX:
                continue
            kept = 0
            for vs, us in _blobs(m, self.MIN_PIX,
                                 min_cell=self.CELL_FILL.get(cname, 0)):
                if kept >= 3:
                    break
                wp = world[vs, us]
                if not np.isfinite(wp).any():
                    continue
                plo, phi = np.nanpercentile(wp[:, :2], self.SPAN_PCT, axis=0)
                span = float(np.linalg.norm(phi - plo))
                cap = self.CLS_MAX_SPAN.get(cname, self.MAX_SPAN)
                if not np.isfinite(span) or span > cap:
                    continue
                # Saturation, per BLOB rather than per pixel: a floor in label_map would
                # delete the object's own dim pixels, whereas a real rind blob measures a
                # median 0.286-0.380 against terrain fragments at 0.080.
                sat_med = float(np.median(s[vs, us]))
                if sat_med < self.MIN_BLOB_SAT.get(cname, 0.0):
                    continue
                # POSITION comes from ray/ground intersection, not from the depth value.
                # The depth-derived median put objects 0.2-0.3 m below the terrain even
                # with the robot standing still, while the camera pose itself is accurate
                # (a flat ground unprojects to its true height). Every competition object
                # rests ON the ground, so the ray through the blob's bottom edge — its
                # contact line — hits the ground exactly where the object stands.
                vb = vs >= np.percentile(vs, 80)      # bottom fifth of the blob
                ub, vbot = us[vb].mean(), vs[vb].mean()
                ray_c = np.array([(ub - cx) / fx, (vbot - cy) / fy, 1.0])
                ray_w = R @ ray_c
                if ray_w[2] > -1e-3:                  # not pointing at the ground
                    continue
                t = (cam_p[2] - ground_z) / (-ray_w[2])
                pos = cam_p + t * ray_w
                if not np.isfinite(pos).all() or t > self.FAR_CUT * 1.6:
                    continue
                # size consistency: a real object subtends a predictable pixel area at
                # its measured range; terrain patches that happen to match a colour do not
                rng = float(np.median(depth[vs, us]))
                dims = sorted(OBJ_DIMS[cname], reverse=True)[:2]
                exp_px = dims[0] * dims[1] * fx * fy / max(rng, 0.3) ** 2
                if not (self.AREA_LO.get(cname, 0.08) * exp_px
                        <= len(vs) <= 5.0 * exp_px):
                    continue
                kept += 1
                yaw, elong = _blob_axis(wp[:, :2])
                # HOW TALL IS IT? The same ray trick, run on the blob's TOP edge: the
                # object stands on `pos`, so its top is where the vertical line through
                # `pos` meets the ray through the topmost rows. This is the one thing a
                # grasp needs that the catalogue cannot supply — a can reads 0.115 m
                # standing and 0.066 m on its side, and aiming at the wrong one puts
                # the fingers 5 cm over the object (measured: hs5b's cans, grasp point
                # z error +0.058). NaN when the geometry degenerates (top ray level
                # with the camera, or the object at the frame edge).
                vt = vs <= np.percentile(vs, 20)          # top fifth of the blob
                ut_, vtop = us[vt].mean(), vs[vt].mean()
                ray_t_w = R @ np.array([(ut_ - cx) / fx, (vtop - cy) / fy, 1.0])
                den = float(np.hypot(ray_t_w[0], ray_t_w[1]))
                h_est = np.nan
                if den > 1e-6:
                    tt = float(np.hypot(pos[0] - cam_p[0], pos[1] - cam_p[1])) / den
                    h_est = float(cam_p[2] + tt * ray_t_w[2] - pos[2])
                    if not (0.0 <= h_est <= 0.6):
                        h_est = np.nan
                # WHERE THE FOOTPRINT REALLY IS. `pos` above averages the pixels of the
                # blob's bottom fifth, and for an object seen at an angle those rows
                # cover only its NEAREST corner — the average then sits toward that
                # corner in BOTH axes (measured on the object-honest probe: the grasp
                # point landed 25-30 mm off, against a policy trained on 12 mm of
                # bias). The silhouette's bottom PROFILE has no such bias: take the
                # lowest pixel of every column, drop it on the ground, and read the
                # near edge's depth and its lateral MIDPOINT off the resulting curve.
                # `edge_w` is that midpoint of the near edge; the caller adds half the
                # class's depth along `view_w` to get the centre.
                edge_w, view_w = None, None
                try:
                    _uu, _inv = np.unique(us, return_inverse=True)
                    _vb = np.zeros(len(_uu))
                    np.maximum.at(_vb, _inv, vs)                    # lowest row per column
                    _rc = np.stack([(_uu - cx) / fx, (_vb - cy) / fy,
                                    np.ones(len(_uu))], axis=1)
                    _rw = _rc @ R.T
                    _ok = _rw[:, 2] < -1e-3
                    if _ok.sum() >= 3:
                        _tt = (cam_p[2] - ground_z) / (-_rw[_ok, 2])
                        _pp = cam_p[None, :] + _tt[:, None] * _rw[_ok]
                        _d = np.median(_pp[:, :2], axis=0) - cam_p[:2]
                        _d = _d / max(float(np.linalg.norm(_d)), 1e-6)
                        _n = np.array([-_d[1], _d[0]])
                        _lat = _pp[:, :2] @ _n
                        _dep = _pp[:, :2] @ _d
                        _latc = 0.5 * (np.percentile(_lat, 5) + np.percentile(_lat, 95))
                        _depn = float(np.percentile(_dep, 10))      # the near edge
                        # (_d, _n) is an orthonormal basis of the ground plane, so a
                        # point is just its two coordinates recombined
                        _xy = _depn * _d + _latc * _n
                        edge_w = np.array([_xy[0], _xy[1], float(pos[2])])
                        view_w = np.array([_d[0], _d[1], 0.0])
                except Exception:
                    edge_w, view_w = None, None
                # FLAT-PAINT VETO (0908): every redbag false positive in dataset_0908
                # clustered at ONE course arc — the red FINISH LINE (201 reports,
                # 16% good). Paint has no height; the bag stands 0.10-0.27 m. Reject
                # a class-listed detection whose measured h_est is confidently flat.
                # NaN h_est (degenerate geometry) passes — uncertainty is not paint.
                _minh = cls.CLS_MIN_H.get(cname)
                if _minh is not None and np.isfinite(h_est) and h_est < _minh:
                    continue
                out.append({"cls": cname, "pos_w": pos.astype(float),
                            "edge_w": None if edge_w is None else edge_w.astype(float),
                            "view_w": None if view_w is None else view_w.astype(float),
                            "n_pix": int(len(vs)), "dist": rng, "ray_t": float(t),
                            "span": span, "above": float(pos[2] - ground_z),
                            "hue": float(np.median(h[vs, us])), "h_est": h_est,
                            "sat": sat_med, "yaw": yaw, "elong": elong})
        return out


# ======================================================================================
# venue navigation: the course SHAPE is unknown at the real venue. The only lane
# information the robot has is the red belt strung along both edges (guide 296: 4.5 cm
# webbing at 0.40 m) and the only "end of course" is the moment the three bins are SEEN.
# Both are estimated here from the head camera alone -- no course map, no depth.
# ======================================================================================

BELT_HEIGHT = 0.40          # webbing centre above local ground (guide line 296)
BELT_HUE_LO, BELT_HUE_HI = 330.0, 25.0     # red, wrapping 0
# Loose on purpose: the webbing is 4.5 cm tall, so at 4-8 m it is a 3-8 px ribbon and a
# strict mask starves the fit -- measured on 217 frames, 41 % had under 40 strict-red
# pixels while the belt was plainly in view.
BELT_SAT_MIN = 0.35
BELT_VAL_MIN = 0.15
BELT_MIN_PX = 40
BIN_COLORS_NAV = {"red": (0.0, 25.0, 330.0), "green": (90.0, 150.0, None),
                  "blue": (200.0, 250.0, None)}


def _belt_mask(rgb, add_noise=None):
    """Red-belt pixel mask, shared by both belt estimators.

    Factored out so the plane-intersection estimator and the bearing-only one cannot
    drift apart on the part that is identical: which pixels are belt. Returns None when
    there is not enough belt in view.
    """
    rgb = np.asarray(rgb)
    if add_noise if add_noise is not None else REAL_NOISE:
        rgb = np.clip(rgb.astype(np.float32)
                      + np.random.normal(0, CAM_RGB_SIGMA, rgb.shape), 0, 255)
    h, s_, v = _rgb_to_hsv_arrays(rgb)
    red = (((h >= BELT_HUE_LO) | (h <= BELT_HUE_HI)) & (s_ > BELT_SAT_MIN)
           & (v > BELT_VAL_MIN))
    if red.sum() < BELT_MIN_PX:
        return None
    lab_img = np.zeros(red.shape, np.int32)
    n_lab = 0
    for vs, us in _blobs(red, 10, min_cell=0):
        n_lab += 1
        lab_img[vs, us] = n_lab
        u_lo, u_hi = us.min(), us.max()
        v_lo, v_hi = vs.min(), vs.max()
        wpx, hpx = u_hi - u_lo + 1, v_hi - v_lo + 1
        edge = (u_lo <= 2) or (u_hi >= red.shape[1] - 3)
        ribbon = wpx >= 4 * hpx or wpx > 0.35 * red.shape[1]
        if not (edge or ribbon):
            lab_img[vs, us] = 0
    red = lab_img > 0
    return red if red.sum() >= BELT_MIN_PX else None


def detect_belt_bearing(rgb, K, cam_rot, add_noise=None):
    """Belt heading + centring error WITHOUT any height or depth assumption.

    WHY THIS EXISTS. detect_belt intersects each red ray with a plane at
    ground + BELT_HEIGHT (0.40 m). Standing, the head camera is only 0.478 m above the
    ground, so that plane sits 7.8 cm below the lens -- and measured over 867 recorded
    frames the camera drops to 0.345 m on the stairs and 0.391 m on grass, i.e. BELOW the
    plane. The intersection then flips sign and the estimator either returns nothing
    (stairs fix rate 61 %) or returns garbage (stairs median error 1671 mm vs 84 mm on
    mud). That is the geometric reason the map-free walk has always failed at the stairs,
    and no amount of filtering fixes a sensor that is blind for the second half of the
    course.

    A straight line in 3D projects to an image line whose VANISHING POINT is the
    projection of the line's direction. Direction needs no height and no range. So:
      * fit the left and right belts as lines in IMAGE space,
      * intersect them -> vanishing point -> belt direction -> heading error,
      * take the centring error from the left/right asymmetry, normalised so a uniform
        scale change (which is what a height error is, to first order) cancels.

    Returns {heading [rad, + = course goes left], asym [-1..1, + = robot right of centre],
             n_left, n_right} or None. `asym` is dimensionless: it says which way to move,
    not how far, which is all a centring controller needs.
    """
    red = _belt_mask(rgb, add_noise)
    if red is None:
        return None
    R = _as_rot(cam_rot)
    Himg, Wimg = red.shape[:2]
    vv, uu = np.nonzero(red)
    if len(uu) > 4000:
        idx = np.random.choice(len(uu), 4000, replace=False)
        vv, uu = vv[idx], uu[idx]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    # split by image side; the belts are left and right of the optical axis
    lines = {}
    for name, m in (("left", uu < cx), ("right", uu >= cx)):
        if m.sum() < 12:
            continue
        # u = a*v + b  (fit u against v: the belts are near-vertical in image space as
        # they run away from the camera, so v is the better independent variable)
        A = np.stack([vv[m], np.ones(int(m.sum()))], axis=1)
        sol, *_ = np.linalg.lstsq(A, uu[m].astype(float), rcond=None)
        r = np.abs(A @ sol - uu[m])
        keep = r <= np.percentile(r, 75)
        if keep.sum() >= 8:
            sol, *_ = np.linalg.lstsq(A[keep], uu[m][keep].astype(float), rcond=None)
        lines[name] = (float(sol[0]), float(sol[1]), int(m.sum()))
    if len(lines) < 2:
        return None
    (aL, bL, nL), (aR, bR, nR) = lines["left"], lines["right"]
    if abs(aL - aR) < 1e-6:
        return None                      # parallel in image = vanishing point at infinity
    v_van = (bR - bL) / (aL - aR)
    u_van = aL * v_van + bL
    # ray through the vanishing point = the belt's direction in camera coords
    d_cam = np.array([(u_van - cx) / fx, (v_van - cy) / fy, 1.0])
    d_w = R @ d_cam
    # heading of that direction relative to the camera's own forward azimuth
    az_belt = np.arctan2(d_w[1], d_w[0])
    az_fwd = np.arctan2(R[1, 2], R[0, 2])
    heading = float((az_belt - az_fwd + np.pi) % (2 * np.pi) - np.pi)
    if abs(heading) > np.pi / 2:         # vanishing point behind: same line, other end
        heading = float((heading + np.pi) % (2 * np.pi) - np.pi)
    # centring: horizontal distance to each belt at the BOTTOM of the frame, where the
    # belts are nearest and the fit is best conditioned
    v_ref = float(Himg - 1)
    uL = aL * v_ref + bL
    uR = aR * v_ref + bR
    dL = max(cx - uL, 1e-3)
    dR = max(uR - cx, 1e-3)
    asym = float((dR - dL) / (dR + dL))
    return {"heading": heading, "asym": asym, "n_left": nL, "n_right": nR,
            "u_van": float(u_van), "v_van": float(v_van)}


# Minimum camera height ABOVE the belt plane for the ray/plane intersection to be usable.
# Measured over 867 recorded frames: at 0.478 m standing height (0.078 above the plane) the
# plane estimator is excellent on mud (99 % fix, 84 mm) and fine on the slope, degrades on
# pebble (0.442 min -> 67 %, 261 mm), is poor on grass (0.391 min -> 817 mm) and is garbage
# on stairs (0.345 min -> 61 % fix but 1671 mm error, i.e. WORSE than no fix). The sign of
# the intersection flips once the camera drops below the plane, so this is a hard geometric
# limit, not a tuning threshold.
# 0.07 is MEASURED, not chosen: binning the 867 recorded frames by (cam height - plane
# height) and comparing both estimators' heading error against truth gives
#   margin < 0.02 m : plane returns nothing at all      | bearing 44.7 deg
#   0.02 - 0.05 m   : plane 19.3 deg                    | bearing  6.4 deg   <- bearing
#   0.05 - 0.07 m   : plane  6.9 deg                    | bearing  5.7 deg   <- bearing
#   0.07 - 0.08 m   : plane  4.6 deg                    | bearing 12.9 deg   <- plane
#   > 0.08 m        : plane 12.5 deg                    | bearing 23.8 deg   <- plane
# The crossover sits at 0.07, which is barely below the 0.078 m the robot has when
# standing -- i.e. the plane estimator is operating with almost no margin everywhere and
# falls off a cliff the moment the body dips.
BELT_PLANE_MARGIN = 0.07


def detect_belt_hybrid(rgb, K, cam_p, cam_rot, ground_z, add_noise=None):
    """Belt fix that switches estimator on whether the plane geometry is valid here.

    Returns {lat, heading, source, ...} with `lat` possibly None. The caller (navfuse)
    already accepts a fix with only one component, which is the point: on the stairs the
    lateral offset is simply not observable with this camera mounting, but the HEADING
    still is -- via the vanishing point, which needs no height at all. Losing heading is
    what put the map-free walk off the course; losing lat is survivable for a while
    because leg odometry carries it.
    """
    cam_h = float(np.asarray(cam_p)[2]) - float(ground_z)
    if cam_h > BELT_HEIGHT + BELT_PLANE_MARGIN:
        p = detect_belt(rgb, K, cam_p, cam_rot, ground_z, add_noise)
        if p is not None:
            p = dict(p)
            p["source"] = "plane"
            return p
    b = detect_belt_bearing(rgb, K, cam_rot, add_noise)
    if b is None:
        return None
    return {"lat": None, "heading": b["heading"], "halfwidth": None,
            "source": "bearing", "n_left": b["n_left"], "n_right": b["n_right"]}


def detect_belt(rgb, K, cam_p, cam_rot, ground_z, add_noise=None):
    """Estimate the corridor from the red belt: -> dict or None.

    Red pixels are unprojected by intersecting their camera ray with the BELT PLANE
    (z = ground + 0.40) -- pure geometry, no depth channel, which is the point: the
    depth camera is the sensor we trust least outdoors. Points are split left/right of
    the camera axis and a line is fitted to each side; the corridor is their mean.

    Returns {lat: robot's lateral offset from the corridor centre (+left),
             heading: corridor direction relative to the robot's forward axis [rad],
             halfwidth, n_left, n_right}.
    """
    rgb = np.asarray(rgb)
    # NOTE: detect_belt is deliberately NOT downscaled (unlike detect_arrays).  The belt is
    # a thin 1-2 px ribbon, not a chunky blob: subsampling by 2 drops its pixel count below
    # what the left/right line fit needs and the lateral estimate diverges catastrophically
    # (measured on dataset_bench: lat p90 jumps to 2.8 m, single frames to +92 m as one side
    # falls to ~16 px).  Since lat/heading here drive STEERING, that is unsafe.  detect_belt
    # is only ~10 ms so it is left full-res; the win from downscaling lives in detect_arrays.
    if add_noise if add_noise is not None else REAL_NOISE:
        rgb = np.clip(rgb.astype(np.float32)
                      + np.random.normal(0, CAM_RGB_SIGMA, rgb.shape), 0, 255)
    R = _as_rot(cam_rot)
    Himg, Wimg = rgb.shape[:2]
    hsv = _rgb_to_hsv_arrays(rgb)
    h, s_, v = hsv
    red = (((h >= BELT_HUE_LO) | (h <= BELT_HUE_HI)) & (s_ > BELT_SAT_MIN)
           & (v > BELT_VAL_MIN))
    if red.sum() < BELT_MIN_PX:
        return None
    # Drop COMPACT red blobs before fitting: the red PAPER BAGS (and the red bin at the
    # course end) pass any colour test the belt passes, sit mid-corridor, and drag the
    # line fit toward the centre. The belt is the opposite shape -- a ribbon that runs to
    # the frame edge or stretches wide -- so keep only wide/edge-touching structures.
    lab_img = np.zeros(red.shape, np.int32)
    n_lab = 0
    for vs, us in _blobs(red, 10, min_cell=0):
        n_lab += 1
        lab_img[vs, us] = n_lab
        u_lo, u_hi = us.min(), us.max()
        v_lo, v_hi = vs.min(), vs.max()
        wpx, hpx = u_hi - u_lo + 1, v_hi - v_lo + 1
        edge = (u_lo <= 2) or (u_hi >= red.shape[1] - 3)
        ribbon = wpx >= 4 * hpx or wpx > 0.35 * red.shape[1]
        if not (edge or ribbon):
            lab_img[vs, us] = 0                  # bag-shaped: out of the fit
    red = lab_img > 0
    if red.sum() < BELT_MIN_PX:
        return None
    vv, uu = np.nonzero(red)
    # subsample for speed; the fit needs hundreds, not tens of thousands
    if len(uu) > 4000:
        idx = np.random.choice(len(uu), 4000, replace=False)
        vv, uu = vv[idx], uu[idx]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    rays_c = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu, float)], axis=1)
    rays_w = rays_c @ R.T
    dz = rays_w[:, 2]
    keep = dz < -1e-3                          # pointing down toward the belt plane
    plane_z = ground_z + BELT_HEIGHT
    tt = (plane_z - cam_p[2]) / dz[keep]
    ok = (tt > 0.3) & (tt < 12.0)
    pts = cam_p[None, :] + rays_w[keep][ok] * tt[ok, None]
    if len(pts) < 30:
        return None
    # robot frame: x forward along camera azimuth, y left
    az = np.arctan2(R[1, 2], R[0, 2])
    ca, sa = np.cos(az), np.sin(az)
    rel = pts[:, :2] - cam_p[None, :2]
    xf = rel[:, 0] * ca + rel[:, 1] * sa
    yl = -rel[:, 0] * sa + rel[:, 1] * ca
    # 0.3..9 m. A near-field-only cut was tried and it inverted the results (55 fixes,
    # 1.5 m median error): the belt NEAR the robot runs along the frame edges at +-2 m
    # lateral and is mostly OUT of view -- what the camera actually sees is the far belt
    # converging toward the vanishing point, so the far field carries the fit.
    fwd = (xf > 0.3) & (xf < 9.0)
    xf, yl = xf[fwd], yl[fwd]
    if len(xf) < 30:
        return None
    out = {}
    for name, m in (("left", yl > 0.3), ("right", yl < -0.3)):
        if m.sum() >= 12:
            # y = a*x + b fitted with a least-squares line; median-ish robustness via a
            # single re-fit that drops the worst quartile of residuals
            A = np.stack([xf[m], np.ones(m.sum())], axis=1)
            sol, *_ = np.linalg.lstsq(A, yl[m], rcond=None)
            r = np.abs(A @ sol - yl[m])
            keep2 = r <= np.percentile(r, 75)
            if keep2.sum() >= 8:
                sol, *_ = np.linalg.lstsq(A[keep2], yl[m][keep2], rcond=None)
            out[name] = (float(sol[0]), float(sol[1]), int(m.sum()))
    if not out:
        return None
    if len(out) == 2:
        aL, bL, nL = out["left"]; aR, bR, nR = out["right"]
        a_mid = 0.5 * (aL + aR)
        b_mid = 0.5 * (bL + bR)
        halfw = 0.5 * abs(bL - bR)
    else:
        (a1, b1, n1) = next(iter(out.values()))
        side = 1.0 if "left" in out else -1.0
        a_mid, b_mid = a1, b1 - side * 1.9775     # nominal halfwidth from the guide strip
        halfw = 1.9775
        nL = out.get("left", (0, 0, 0))[2]; nR = out.get("right", (0, 0, 0))[2]
        nL, nR = (n1, 0) if side > 0 else (0, n1)
    lat = -b_mid
    heading = float(np.arctan(a_mid))
    # SANITY GATES on the fit itself. Without them the error tail was p90 1.87 m --
    # useless fixes from degenerate fits are worse than "no fix", because the caller
    # falls back to dead reckoning on None but steers off a bad number.
    # NO output gates. They were tried (halfwidth window + |lat|/heading caps) and they
    # cut the fix count from 114 to 45 while the survivors got WORSE (median error 275 mm
    # -> 1.5 m): on the curved sections the far-field line extrapolated back to the robot
    # legitimately reports an out-of-window halfwidth, so the gate was discarding exactly
    # the good fixes. Callers should SMOOTH consecutive fixes instead of trusting a gate.
    return {"lat": lat,                          # robot left of centre -> positive lat
            "heading": heading,
            "halfwidth": halfw, "n_left": out.get("left", (0, 0, 0))[2],
            "n_right": out.get("right", (0, 0, 0))[2]}


def detect_bins(rgb, K, cam_p, cam_rot, ground_z, add_noise=None, depth=None):
    """Spot the bin row: three tall saturated blobs (red/green/blue) side by side.

    The END of the course is DEFINED by this sighting at the venue -- there is no map
    to say where the bins are. Position again by ray-plane geometry: the bins are 0.47 m
    tall, so their blob centres sit ~0.24 m above ground.

    With `depth` (z-depth image, same shape as rgb) the RANGE comes from the blob's
    median depth instead of the plane cut. The plane cut is hopeless in range: the
    head cam sits ~0.5 m up, so at 3 m the ray meets the 0.24 m plane at a 5 deg
    grazing angle and a 5 cm error in where the blob's centre pixel really is on the
    face becomes 0.5 m of range (hs1: first drop 0.31 m short of the blue bin). The
    depth-ranged point is the visible FACE of the bin; the caller adds the face-to-
    centre half depth. `ranges` in the result carries the range used per colour.
    """
    rgb = np.asarray(rgb)
    dep = None if depth is None else np.asarray(depth, np.float64)
    if dep is not None and dep.ndim == 3:
        dep = dep[..., 0]
    if add_noise if add_noise is not None else REAL_NOISE:
        rgb = np.clip(rgb.astype(np.float32)
                      + np.random.normal(0, CAM_RGB_SIGMA, rgb.shape), 0, 255)
        if dep is not None:
            # the same range-dependent depth error every other depth consumer sees
            _sig = CAM_DEPTH_ABS + CAM_DEPTH_REL * np.nan_to_num(dep, nan=0.0, posinf=0.0)
            dep = dep + np.random.normal(0, 1.0, dep.shape) * _sig
    hsv = _rgb_to_hsv_arrays(rgb)
    h, s_, v = hsv
    found = {}
    R = _as_rot(cam_rot)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    Himg, Wimg = h.shape
    for name in ("green", "blue", "red"):
        lo, hi, wrap = BIN_COLORS_NAV[name]
        m = ((h >= wrap) | (h <= hi)) if wrap is not None else ((h >= lo) & (h <= hi))
        # sat gate 0.45 -> 0.30 and fill 0.35 -> 0.25 (both re-fit 0828 on head-cam
        # frames of the PHOTO-textured bins: at 8 m the mipmap-averaged faces drop
        # below sat 0.45 for all three colours — zero fixes across an entire mission —
        # while sky and belt are already rejected downstream by the ray/aspect gates,
        # not by saturation)
        # per-colour sat floor: the pastel GREEN bin face sits below 0.30 saturation
        # (frame dumps 0828 — only its shaded edges ever passed), so green gets 0.22;
        # geometry gates carry the false-positive load, verified 0 FP on the
        # photo_range mid-course suite
        m = m & (s_ > (0.22 if name == "green" else 0.30)) & (v > 0.2)
        # A bin is a COMPACT blob — but evaluate PER CONNECTED COMPONENT, not on the
        # whole colour mask. The whole-mask version worked on flat-colour bins behind a
        # sat>0.45 wall; the photo world's sky (blue) and belt (red) share the hue
        # windows at sat 0.30+, and pooling them with the bin blob explodes the
        # percentile box, so every gate downstream rejects the pooled mess: measured
        # zero fixes across all of basket38 with the robot standing AT the bins. Each
        # blob is judged alone; sky still dies by size/aspect, the belt by aspect,
        # and the bin blob no longer inherits their pixels.
        best = None
        for b_ys, b_xs in _blobs(m, min_pix=250, max_blobs=6, min_cell=3):
            uu, vv = b_xs, b_ys
            # a blob cut by the left/right image border is a partial face: its median
            # pixel and depth are biased toward whatever half is in view (bins probe
            # 0905: the red bin at 29 deg off-axis read 0.18 m deep / 0.12 m wide)
            if int(uu.min()) <= 1 or int(uu.max()) >= Wimg - 2:
                continue
            u_lo, u_hi = np.percentile(uu, [5, 95])
            v_lo, v_hi = np.percentile(vv, [5, 95])
            wpx, hpx = max(u_hi - u_lo, 1.0), max(v_hi - v_lo, 1.0)
            npx = len(uu)
            t_d = None
            pix_d = None
            if dep is not None:
                dz = dep[vv, uu]
                okd = np.isfinite(dz) & (dz > 0.3) & (dz < 15.0)
                dz = dz[okd]
                if len(dz) >= max(30, npx // 3):
                    # z-depth of the face along the optical axis -> scale the unit-z
                    # ray by it (same convention as _unproject)
                    t_d = float(np.median(dz))
                    pix_d = (uu[okd], vv[okd], dz)
            # frame-filling = scenery; a depth-ranged blob may stand taller (the bin
            # face fills ~70 % of the frame height from the 0.6 m drop stance)
            if wpx > 0.35 * Wimg or hpx > (0.85 if t_d is not None else 0.6) * Himg:
                continue
            if not (0.4 <= hpx / wpx <= 3.0):            # ribbons out
                continue
            if npx / (wpx * hpx) < 0.18:                 # scattered speckle out
                continue
            u0, v0 = float(np.median(uu)), float(np.median(vv))
            ray_c = np.array([(u0 - cx) / fx, (v0 - cy) / fy, 1.0])
            ray_w = R @ ray_c
            # t = the ray parameter (point = cam_p + t * ray_w; ray_c has unit z, so
            # t IS the z-depth): the plane cut when there is no depth
            if t_d is None:
                if ray_w[2] > -1e-3:
                    continue
                t = (ground_z + 0.24 - cam_p[2]) / ray_w[2]
            else:
                t = t_d
            if not (0.5 < t < 12.0):
                continue
            # size-at-range: a 0.40 x 0.47 bin face must subtend ~exp px at range t
            exp = 0.40 * 0.47 * fx * fy / max(t, 0.3) ** 2
            if not (0.25 * exp <= npx <= 4.0 * exp):
                continue
            if best is None or npx > best[1]:
                best = (cam_p + t * ray_w, npx, t * float(np.linalg.norm(ray_c)), pix_d)
        if best is not None:
            front = None
            if best[3] is not None:
                front = _bin_front_face(best[3], K, cam_p, R)
                if front is False:
                    # the blob is not one bin (belt rope merged behind it, or the side
                    # face alone): its median point is off by 0.1-0.2 m either way
                    continue
            found[name] = (best[0], int(best[1]), float(best[2]), front)
    if len(found) < 2:                           # need at least two of the trio
        return None
    ps = np.stack([p for p, _, _, _ in found.values()])
    centre = ps.mean(axis=0)
    d2 = np.linalg.norm(ps[:, :2] - centre[None, :2], axis=1)
    # a ROW: neighbours ~0.9 m apart, so pair distances live in 0.4..2.5 m -- identical
    # positions (one object passing two colour masks) and far-flung scatter both fail
    if float(d2.max()) > 2.0 or (len(found) >= 2 and float(d2.max()) < 0.15):
        return None
    return {"centre_w": centre, "colors": {k: v[0] for k, v in found.items()},
            "ranges": {k: v[2] for k, v in found.items()},
            "front": {k: v[3] for k, v in found.items() if v[3] is not None},
            "n_seen": len(found)}


# front-band median -> centre of a bin along the course (bins are 0.40 m deep; the
# band median sits ~2 cm inside the rim: bins probe b4, clean and noisy depth, x error
# +0.021 / +0.018 with 0.20 -> 0.18 centres it)
BIN_FRONT_TO_CENTRE = 0.18


def _bin_front_face(pix_d, K, cam_p, R):
    """FRONT-FACE point of a bin blob from its depth point cloud, in the frame of
    `cam_rot` (x = course direction when called from navest's S frame).

    The blob's median pixel at its median depth is somewhere on the front face for a
    bin dead ahead, but for a SIDE bin it mixes the front and the near side face, and
    the face->centre correction along the viewing ray then adds ~0.05 m OUTWARD per
    sighting (bins probe 0905: blue/red at 2 m read +0.05 m past their true lateral,
    green 0.01 — the drops in hs4a landed 0.11-0.24 m past the blue bin). The front
    rim is the blob's smallest x (5th percentile), and the centre line the median y of
    the points within 6 cm of it: 1 cm on all three colours from every stance probed.
    -> np.array([x_front, y_centre, z]) or False when the blob is not one bin face
    (under 30 % of its points near the front rim -- the belt rope merged behind the
    red bin gave 23 % and a 0.11 m lateral miss -- or 1.5 m of depth extent).
    """
    uu, vv, dz = pix_d
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    rays = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(dz, dtype=np.float64)], axis=1)
    pts = np.asarray(cam_p, np.float64)[None, :] + dz[:, None] * (rays @ R.T)
    # A bin face is never more than ~0.2 m ahead of its own blob's median depth, so
    # same-hue clutter lying further in front (a green rind dropped short of the green
    # bin) must not become the "rim": search the rim only within 0.25 m of the median.
    x_med = float(np.median(pts[:, 0]))
    near = pts[pts[:, 0] > x_med - 0.25]
    x_f, x_b = np.percentile(near[:, 0], [5, 95])
    # "smallest x = front" only holds with the camera on the near side of the row;
    # looking back from past the bins the compact BACK face would pass as the front
    # (+0.36 m along the course), so refuse that geometry outright.
    if float(cam_p[0]) > x_f:
        return False
    # The 5th percentile is a tail statistic: with the sensor's range noise
    # (CAM_DEPTH_ABS + CAM_DEPTH_REL * d, ~2 cm at 2 m) it sits ~1.6 sigma in front of
    # the face. Take the band from that tail out to 3 sigma past the face and report
    # the band's MEDIAN x, which the noise does not shift; BIN_FRONT_TO_CENTRE is
    # calibrated to this median (bins probe b4), not to the rim itself.
    sig = CAM_DEPTH_ABS + CAM_DEPTH_REL * float(np.median(dz))
    fr = near[near[:, 0] < x_f + 0.06 + 3.0 * sig]
    if len(fr) < 0.30 * len(pts) or (x_b - x_f) > 1.5:
        return False
    # The face is a LINE in (x, y), and on the curving course it is tilted 5-9 deg
    # from the S y-axis (probe b6): across a 0.4 m bin that is 5 cm of x, so a strip
    # cut by x alone favours the near corner -- under range noise the 5th-pct anchor
    # picks the near-corner tail and the centreline reads 2 cm off (b6n). Fit the
    # face line to the band (two passes: the near SIDE face of a bin seen at an angle
    # sits 0.1-0.4 m behind it and must not tilt the fit), then cut the rim strip by
    # the residual to that line, which has no preference in y.
    a, b = _face_line(fr)
    if a is None:
        return False
    res = near[:, 0] - (a + b * near[:, 1])
    r_f = float(np.percentile(res, 5))
    rim = near[res < r_f + 0.06]
    if len(rim) < 0.15 * len(pts):
        # the strip is a few per cent of the blob: same-hue clutter just ahead of the
        # face (a rind 10-20 cm short of the bin) anchored it, and its y would be the
        # rind's, not the bin's -- take the strip on the fitted face itself instead
        rim = near[np.abs(res) < 0.04 + 2.0 * sig]
        if len(rim) < 0.15 * len(pts):
            return False
    # Centreline = midpoint of the strip's y EXTENT, not its median: the near side
    # face of a bin seen at an angle starts at the front plane, so its nearest points
    # sit in the strip and would drag a median toward that edge (synthetic: 30 % side
    # points -> 4 cm), while the extent they occupy is the edge itself.
    y_lo, y_hi = np.percentile(rim[:, 1], [2, 98])
    y_c = 0.5 * float(y_lo + y_hi)
    # the face line at the centreline column: the band median (probe b4) plus the
    # tilt, unbiased under symmetric range noise
    return np.array([a + b * y_c, y_c, float(np.median(rim[:, 2]))])


def _face_line(fr):
    """Face line x = a + b*y through the front-band points.

    The slope comes from the FRONT-MOST x (20th percentile) of 3 cm columns in y: a
    visible side face lies behind the front plane and starts AT it, so it cannot pull
    a column's front percentile back the way it tilts a least-squares fit through
    every point (synthetic: 30 % side-face points turned +7 deg into -2 deg). The
    offset is the median of x - b*y over the band, unbiased under range noise.
    -> (a, b) or (None, None) when the band is too thin or the tilt implausible."""
    y, x = fr[:, 1], fr[:, 0]
    if len(fr) < 20 or float(np.ptp(y)) < 0.10:
        return None, None
    yc, xq = [], []
    for lo in np.arange(float(y.min()), float(y.max()), 0.03):
        m = (y >= lo) & (y < lo + 0.03)
        if int(m.sum()) >= 5:
            yc.append(lo + 0.015)
            xq.append(float(np.percentile(x[m], 20)))
    if len(yc) < 4:
        return None, None
    _, b = _lsq_line(np.asarray(yc), np.asarray(xq))
    if abs(b) > 0.5:                                 # > 27 deg: not a bin face we can use
        return None, None
    return float(np.median(x - b * y)), float(b)


def _lsq_line(y, x):
    ym, xm = float(np.mean(y)), float(np.mean(x))
    vy = float(np.sum((y - ym) ** 2))
    b = float(np.sum((y - ym) * (x - xm)) / vy) if vy > 1e-9 else 0.0
    return xm - b * ym, b


def _rgb_to_hsv_arrays(rgb):
    r, g, b = rgb[..., 0] / 255.0, rgb[..., 1] / 255.0, rgb[..., 2] / 255.0
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    df = mx - mn
    hh = np.zeros_like(mx)
    m1 = (mx == r) & (df > 0); hh[m1] = (60 * ((g[m1] - b[m1]) / df[m1]) + 360) % 360
    m2 = (mx == g) & (df > 0); hh[m2] = (60 * ((b[m2] - r[m2]) / df[m2]) + 120) % 360
    m3 = (mx == b) & (df > 0); hh[m3] = (60 * ((r[m3] - g[m3]) / df[m3]) + 240) % 360
    ss = np.where(mx > 0, df / np.maximum(mx, 1e-9), 0.0)
    return hh, ss, mx


def _blob_axis(xy: np.ndarray):
    """Principal horizontal axis of a blob's unprojected ground points.

    -> (yaw of the long axis in world, mod pi, or nan; elongation sqrt(l1/l2)). The grasp
    policy never sees the object's yaw (its observation is the offset vector + OBB
    lengths), so the mission must square the robot up itself for anything whose short
    side alone fits the jaws -- the lunchbox lid (85-105 mm across, 170+ along) can
    only be straddled with the fingers within ~25 deg of its short axis.
    """
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) < 12:
        return float("nan"), 1.0
    c = xy - np.median(xy, axis=0)
    # trim the 10 % farthest points: depth noise on a blob's rim dominates the tail
    r = np.hypot(c[:, 0], c[:, 1])
    c = c[r <= np.percentile(r, 90)]
    if len(c) < 12:
        return float("nan"), 1.0
    cov = c.T @ c / len(c)
    w, v = np.linalg.eigh(cov)
    ax = v[:, 1]
    elong = float(np.sqrt(max(w[1], 1e-12) / max(w[0], 1e-12)))
    return float(np.arctan2(ax[1], ax[0]) % np.pi), elong


def _as_rot(r):
    """Accept a (w,x,y,z) quaternion or an already-built 3x3 rotation matrix."""
    r = np.asarray(r, dtype=np.float64)
    return r if r.shape == (3, 3) else _quat_to_mat(r)


def _quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


_CELL = 12          # px; the coarse grid both blob backends cluster on
_BLOBS_BACKEND = os.environ.get("BLOBS_BACKEND", "auto")   # auto | fast | py


def _blobs(mask: np.ndarray, min_pix: int, max_blobs: int = 16, min_cell: int = 0):
    """Connected components of `mask` on a coarse 12 px grid.

    Two backends produce the SAME result (same grid, same 8-neighbour merge, same
    min_cell / min_pix / max_blobs semantics); `_blobs_fast` replaces the pure-Python
    dict + union-find of `_blobs_py` -- which held the GIL and dominated the perception
    frame on the mini-PC's N100 E-cores -- with a vectorised grid reduction and a single
    `scipy.ndimage.label` (an order of magnitude faster, per requirements-onboard.txt).

    BLOBS_BACKEND=py forces the reference implementation; =fast forces the accelerated one;
    the default (auto) uses fast when scipy is importable and the reference otherwise, so a
    bare machine still works.  The A/B and the equivalence test drive it through this env.
    """
    if _BLOBS_BACKEND == "py":
        return _blobs_py(mask, min_pix, max_blobs, min_cell)
    if _BLOBS_BACKEND == "fast":
        return _blobs_fast(mask, min_pix, max_blobs, min_cell)
    try:
        import scipy.ndimage  # noqa: F401
    except ImportError:
        return _blobs_py(mask, min_pix, max_blobs, min_cell)
    return _blobs_fast(mask, min_pix, max_blobs, min_cell)


def _blobs_fast(mask: np.ndarray, min_pix: int, max_blobs: int = 16, min_cell: int = 0):
    """Vectorised equivalent of `_blobs_py`.

    The reference groups pixels into 12 px cells and 8-connects the cells.  That is exactly
    an 8-connected labelling of the DOWNSAMPLED cell grid, so we build the cell grid, run
    one `scipy.ndimage.label`, and scatter the component id back to each pixel -- no Python
    loop over cells or pixels.
    """
    from scipy import ndimage as _ndi

    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return []
    cy, cx = ys // _CELL, xs // _CELL
    gh, gw = int(cy.max()) + 1, int(cx.max()) + 1
    lin = cy * gw + cx

    # occupancy + per-cell pixel count on the coarse grid
    cnt_flat = np.bincount(lin, minlength=gh * gw)
    occ = cnt_flat.reshape(gh, gw) > 0
    if min_cell > 1:
        # A cell must be genuinely filled before it may join the graph, else a frame-wide
        # speckle bridges cells into one component that swallows the object (see _blobs_py).
        occ = cnt_flat.reshape(gh, gw) >= min_cell
        keep = occ.reshape(-1)[lin]
        if not keep.any():
            return []
        ys, xs, lin = ys[keep], xs[keep], lin[keep]

    labels, n = _ndi.label(occ, structure=np.ones((3, 3), int))   # 8-connected cells
    if n == 0:
        return []
    pix_lab = labels.reshape(-1)[lin]        # component id per surviving pixel

    order = np.argsort(pix_lab, kind="stable")
    pix_lab, ys, xs = pix_lab[order], ys[order], xs[order]
    bounds = np.flatnonzero(np.diff(pix_lab)) + 1
    starts = np.r_[0, bounds]
    ends = np.r_[bounds, len(pix_lab)]

    out = []
    for a, b in zip(starts, ends):
        if b - a >= min_pix:
            out.append((ys[a:b], xs[a:b]))
    out.sort(key=lambda blob: -len(blob[0]))
    return out[:max_blobs]


def _blobs_py(mask: np.ndarray, min_pix: int, max_blobs: int = 16, min_cell: int = 0):
    """Reference: tiny 2-pass connected components on a coarse grid (no scipy dependency)."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return []
    # cluster by simple grid quantisation then merge neighbours — enough to separate
    # objects that are metres apart in the image
    cell = _CELL
    keys = (ys // cell) * 10000 + (xs // cell)
    uniq, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
    # A 12x12 cell holding a handful of noise-flipped pixels is not evidence of an object,
    # but the 8-neighbour merge below lets it BRIDGE cells, so a sparse frame-wide speckle
    # becomes one component that swallows the real object. Measured on dataset_fixed: a
    # rind on pebble at 2.2 m goes from 306 px / span 0.089 (detected) to 8,655 px /
    # span 1.75 (rejected by MAX_SPAN) the moment sensor noise is switched on, 3 draws of
    # 3. Requiring a cell to be genuinely filled before it may join the graph is what
    # keeps the terrain sea from touching the object.
    if min_cell > 1:
        keep = cnt[inv] >= min_cell          # O(n); np.isin here would be O(n log n)
        if not keep.any():
            return []
        ys, xs, keys = ys[keep], xs[keep], keys[keep]
        uniq, inv = np.unique(keys, return_inverse=True)
    groups = {}
    for i, k in enumerate(uniq):
        groups[k] = np.nonzero(inv == i)[0]
    # merge 8-neighbour cells
    parent = {k: k for k in groups}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for k in groups:
        r, c = divmod(int(k), 10000)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nk = (r + dr) * 10000 + (c + dc)
                if nk in parent:
                    ra, rb = find(k), find(nk)
                    if ra != rb:
                        parent[ra] = rb
    merged = {}
    for k, idx in groups.items():
        merged.setdefault(find(k), []).append(idx)
    out = []
    for _, parts in merged.items():
        idx = np.concatenate(parts)
        if len(idx) >= min_pix:
            out.append((ys[idx], xs[idx]))
    out.sort(key=lambda b: -len(b[0]))
    return out[:max_blobs]


# ======================================================================================
# CORRIDOR SENSOR (map-free): the belt STANCHIONS -- black posts every 2 m down both
# edges of the course -- and, close in, the red belt itself, from the head RGB-D camera.
#
# Output frame = whatever frame `cam_p`/`cam_rot` are given in. Pass the camera pose in
# the gravity-aligned body frame (cam_pose_body) and the posts come back as x forward /
# y left / z relative to the base origin, which is what a corridor fitter wants and what
# the real robot can supply (mount extrinsics + IMU attitude, no map, no GPS).
#
# Why posts and not the belt: with the head camera pitched 22 deg down the top image row
# is only 1.8 deg below horizontal, so the 0.40 m webbing leaves the frame beyond ~2.5 m
# and is only ever seen when the robot is already at the edge. The posts stand on the
# ground and reach camera height, so their lower part is in frame from 1 m to 9 m.
#
# Why geometry and not colour alone: the posts are black, but so are shadows, dark
# photo-texture patches and the shaded sides of the bins. What makes a post a post is that
# its pixels stand up out of the ground at ONE range -- a dark ground patch has zero
# height, terrain depth grows with image row, a bin is 0.4 m wide. Height above ground is
# the most precise measurement this camera has at range (the rays are nearly horizontal,
# so depth noise goes into range, not height: 4 mm of z error at 8 m for 68 mm of range
# error), so the column test is what carries the detector; colour only proposes.
# ======================================================================================

POST_NEAR, POST_FAR = 0.5, 9.5     # depth window [m]; nearer is the robot's own arm
# The posts are (28, 28, 30) in the mesh but render mid-grey: 60-90 LSB up close, 110-150
# at 8-9 m where a 2-3 px shaft is antialiased into the bright off-course floor behind it
# (measured on probe_posts p1). So the colour gate is loose and only proposes; the
# standing-height test is what separates a post from dark ground texture.
POST_V_MAX = 0.62                  # candidate = darker than this (max rgb, 0..1) ...
POST_CHROMA_MAX = 0.12             # ... and grey (max-min of rgb, 0..1); bins are saturated
POST_MIN_PIX = 6                   # a post at 9 m is ~2.5 px wide x 7 standing rows
POST_ABOVE = 0.07                  # m above the local ground grid to count as standing
POST_MIN_EXTENT_CUT = 0.08         # foot-to-top when the frame edge or an occluder truncates
POST_MIN_VIS_H = 0.06              # ... and the rows themselves must span this much post [m],
                                   # else the z-spread is a curved object's depth, not height
                                   # the post (a ramp crest can leave an 8 cm sliver)
POST_OCCLUDER_GAP = 0.4            # rows under the blob this much nearer = bottom occluded
POST_EDGE_TOL = 0.06               # ... and then its top must reach the top-row ray [m]
POST_MIN_EXTENT_FULL = 0.22        # when the top is inside the frame (post 0.46, and a step
                                   # edge or the ramp crest often hides its lower half)
POST_MERGE_XY = 0.25               # detections closer than this are one post (belt split)
POST_MAX_EXTENT = 0.60             # bins are 0.47 (+ foot slack); nothing on course is taller
POST_MAX_FOOT = 0.30               # p10-p90 horizontal spread of the core points [m]
POST_MAX_WIDTH = 0.16              # p10-p90 spread across the viewing ray [m]
POST_MAX_ROW_W = 3.0               # median per-row width <= this x expected px + 2
POST_MIN_ASPECT = 0.8              # pixel height >= this x pixel width (p5-p95)
POST_MAX_FLOAT = 0.45              # |foot - neighbourhood ground| beyond this = not on the ground
POST_DEPTH_TOL = 0.15              # core = pixels within max(this, 4 sigma) of the blob depth
POST_DIAMETER = 0.044
POST_GROUND_CELL = 0.25            # m, xy cell of the coarse ground grid
POST_GROUND_PCT = 5                # per-cell z percentile that counts as the ground
POST_RING_SLACK = 0.06             # a neighbour cell's ground counts only this much lower
BELTPT_Z_LO, BELTPT_Z_HI = 0.28, 0.54   # belt centre 0.40 +- 0.0225 webbing, wide margin
BELTPT_MAX_ZSPAN = 0.12            # a red blob taller than this is a bag/bin, not webbing


def cam_pose_body(mount_pos_w, mount_R_ros, base_pos_w, base_yaw):
    """Camera pose in the GRAVITY-ALIGNED body frame: origin at the base, x along the
    base yaw, z up. Only the base yaw and position enter (they cancel in the relative
    measurement); the body's roll/pitch stay inside the camera rotation, i.e. what the
    IMU supplies on hardware. -> (cam_p (3,), cam_R (3x3) ROS-convention columns)."""
    p = np.asarray(mount_pos_w, np.float64) - np.asarray(base_pos_w, np.float64)
    c, s = float(np.cos(base_yaw)), float(np.sin(base_yaw))
    RzT = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    return RzT @ p, RzT @ _as_rot(mount_R_ros)


def _label_cc(mask: np.ndarray):
    """8-connected components of a boolean image -> (labels int32, n). scipy when it is
    there (it ships with the simulator's python), a numpy min-label propagation otherwise
    so the offline tuning scripts keep working on a bare machine."""
    try:
        from scipy import ndimage as _ndi
        lab, n = _ndi.label(mask, structure=np.ones((3, 3), int))
        return lab.astype(np.int32), int(n)
    except ImportError:
        pass
    H, W = mask.shape
    idx = np.flatnonzero(mask)
    n = len(idx)
    out = np.zeros(mask.shape, np.int32)
    if n == 0:
        return out, 0
    pos = np.full(H * W, -1, np.int64)
    pos[idx] = np.arange(n)
    ys, xs = np.divmod(idx, W)
    ea, eb = [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        ok = (ys + dy < H) & (xs + dx >= 0) & (xs + dx < W)
        nb = pos[(ys[ok] + dy) * W + (xs[ok] + dx)]
        m = nb >= 0
        ea.append(np.flatnonzero(ok)[m])
        eb.append(nb[m])
    a, b = np.concatenate(ea), np.concatenate(eb)
    lab = np.arange(n)
    for _ in range(64):
        mn = np.minimum(lab[a], lab[b])
        new = lab.copy()
        np.minimum.at(new, a, mn)
        np.minimum.at(new, b, mn)
        for _ in range(3):
            new = new[new]
        if np.array_equal(new, lab):
            break
        lab = new
    _, rel = np.unique(lab, return_inverse=True)
    out.flat[idx] = rel + 1
    return out, int(rel.max()) + 1


def _unproject(depth, K, cam_p, R, ys, xs):
    d = depth[ys, xs].astype(np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pc = np.stack([(xs - cx) / fx * d, (ys - cy) / fy * d, d], axis=1)
    return np.asarray(cam_p, np.float64)[None, :] + pc @ R.T, d


class _GroundGrid:
    """Coarse local-ground model: POST_GROUND_PCT-th percentile z per xy cell of every
    valid non-belt depth pixel (subsampled), as a dense array over the frame's xy box.
    Ground rays are near-grazing, so depth noise barely moves a ground point in z and a
    low percentile is safe; it is what keeps a cell that a post shares with a sliver of
    ground reporting the ground. Two read-outs, both precomputed on the dense array so a
    query is one fancy-index: `fill` (own cell, else the nan-min over a growing 3x3 /
    5x5 neighbourhood) for the belt-point height band and the post-foot sanity check,
    and `ring` (min of the own cell and the 8 neighbours' min + POST_RING_SLACK, then
    the same growing fill) for the standing test: the lowest plausible ground next to a
    post, so a post that owns its cell (foot hidden, or 7 m+ where it outnumbers the
    ground pixels) still stands, while the slack keeps a 9-degree ramp's own crest
    (0.04 m per cell, shaded dark on its far side) from standing on its downhill
    neighbour. The post's own foot is measured from its pixels, not from this."""

    def __init__(self, pw, cell=POST_GROUND_CELL, box=None):
        self.cell = cell
        if box is None:
            box = self.box_of(pw, cell)
        self.i0, self.j0, ni, nj = box
        self.fill = np.full((ni, nj), np.nan)
        self.ring = self.fill
        if len(pw) == 0:
            self.global_z = float("nan")
            return
        gi = np.floor(pw[:, 0] / cell).astype(np.int64) - self.i0
        gj = np.floor(pw[:, 1] / cell).astype(np.int64) - self.j0
        inside = (gi >= 0) & (gi < ni) & (gj >= 0) & (gj < nj)
        gi, gj, z = gi[inside], gj[inside], pw[inside, 2]
        key = gi * nj + gj
        # sort by (cell, z) once; the percentile of each run is then an index
        order = np.lexsort((z, key))
        ks, zs = key[order], z[order]
        starts = np.r_[0, np.flatnonzero(np.diff(ks)) + 1]
        ends = np.r_[starts[1:], len(ks)]
        pick = starts + ((ends - starts - 1) * (POST_GROUND_PCT / 100.0)).astype(np.int64)
        g = np.full((ni, nj), np.nan)
        g.flat[ks[starts]] = zs[pick]
        self.global_z = float(np.percentile(z, POST_GROUND_PCT))
        nb1 = self._nb_min(g)                       # 8-neighbour nan-min
        fill1 = np.where(np.isnan(g), nb1, g)
        self.fill = np.where(np.isnan(fill1), self._nb_min(fill1), fill1)
        ring = np.fmin(g, nb1 + POST_RING_SLACK)
        ring = np.where(np.isnan(ring), self._nb_min(ring) + POST_RING_SLACK, ring)
        self.ring = ring

    @staticmethod
    def box_of(pw, cell=POST_GROUND_CELL):
        if len(pw) == 0:
            return (0, 0, 1, 1)
        gi = np.floor(pw[:, 0] / cell).astype(np.int64)
        gj = np.floor(pw[:, 1] / cell).astype(np.int64)
        return (int(gi.min()), int(gj.min()), int(gi.max() - gi.min() + 1),
                int(gj.max() - gj.min() + 1))

    @staticmethod
    def _nb_min(g):
        p = np.pad(g, 1, constant_values=np.nan)
        out = np.full_like(g, np.nan)
        for di in (0, 1, 2):
            for dj in (0, 1, 2):
                if di != 1 or dj != 1:
                    out = np.fmin(out, p[di:di + g.shape[0], dj:dj + g.shape[1]])
        return out

    def _read(self, arr, xy):
        xy = np.asarray(xy, np.float64).reshape(-1, 2)
        gi = np.floor(xy[:, 0] / self.cell).astype(np.int64) - self.i0
        gj = np.floor(xy[:, 1] / self.cell).astype(np.int64) - self.j0
        ni, nj = arr.shape
        ok = (gi >= 0) & (gi < ni) & (gj >= 0) & (gj < nj)
        out = np.full(len(gi), np.nan)
        out[ok] = arr[gi[ok], gj[ok]]
        return out

    def at_many(self, xy, ring_min=False, fallback=True):
        out = self._read(self.ring if ring_min else self.fill, xy)
        if fallback:
            out[np.isnan(out)] = self.global_z
        return out

    def at(self, x, y):
        return float(self.at_many(np.array([[x, y]]))[0])


def _post_prep(rgb, depth, K, cam_p, cam_rot, add_noise):
    """Shared front half of the two corridor detectors: noise, validity, the dark-grey
    mask, and a coarse ground grid built from the NON-dark pixels (a post 2-3 px wide at
    8 m would otherwise own its own ground cell and stand 0 m above itself)."""
    depth = np.asarray(depth, np.float32)
    depth = depth.reshape(depth.shape[0], depth.shape[1])
    rgb = np.asarray(rgb)
    if REAL_NOISE if add_noise is None else add_noise:
        # same model as detect_arrays (CAM_RGB_SIGMA LSB, CAM_DEPTH_ABS + REL*d), drawn
        # float32 from a generator seeded off the global RNG: same reproducibility under
        # np.random.seed, half the cost of the legacy float64 path on a 640x360x3 frame
        g = np.random.default_rng(int(np.random.randint(1 << 31)))
        rgb = np.clip(rgb.astype(np.float32)
                      + g.standard_normal(rgb.shape, dtype=np.float32) * CAM_RGB_SIGMA,
                      0, 255).astype(np.uint8)          # back to 8-bit: 4x cheaper masks
        sig = CAM_DEPTH_ABS + CAM_DEPTH_REL * np.nan_to_num(depth, nan=0.0, posinf=0.0)
        depth = depth + g.standard_normal(depth.shape, dtype=np.float32) * sig
    R = _as_rot(cam_rot)
    cam_p = np.asarray(cam_p, np.float64)
    valid = np.isfinite(depth) & (depth > POST_NEAR) & (depth < POST_FAR)
    r_, g_, b_ = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.maximum(np.maximum(r_, g_), b_)          # channel-wise: 10x faster than
    mn = np.minimum(np.minimum(r_, g_), b_)          # a reduction over the last axis
    dark = (mx < POST_V_MAX * 255.0) & ((mx - mn) < POST_CHROMA_MAX * 255.0) & valid
    # ground grid from every valid pixel except the belt: it hangs 0.40 m up exactly where
    # the posts stand, and at < 2 m the base disc hides most of that cell's ground, so the
    # webbing would become "ground" and the post would start 0.4 m up. Dark pixels stay
    # in (mud and pebble ground are mostly "dark"; excluding them left those cells to the
    # post's own grey fringe); the low percentile + ring-min lookup handle post-owned cells
    keep = valid[::3, ::3] & ~_red_mask(rgb[::3, ::3])
    ys, xs = np.nonzero(keep)
    pw, _ = _unproject(depth, K, cam_p, R, ys * 3, xs * 3)
    box = _GroundGrid.box_of(pw)
    ground = _GroundGrid(pw, box=box)
    # second grid WITHOUT the dark pixels and their anti-aliased 1-px fringe: where a
    # post is all a cell holds (its foot hidden behind a ramp crest or a step edge) the
    # all-pixel grid reports the post's own lowest pixel as ground; this one is empty
    # there and falls through to the neighbourhood instead (3x3 dilation, separable)
    fr = dark.copy()
    fr[:, 1:] |= dark[:, :-1]
    fr[:, :-1] |= dark[:, 1:]
    fr[1:, :] |= fr[:-1, :].copy()
    fr[:-1, :] |= fr[1:, :].copy()
    nd = ~fr[::3, ::3][ys, xs]
    return rgb, depth, R, cam_p, valid, dark, ground, _GroundGrid(pw[nd], box=box)


def _red_mask(rgb):
    """Belt-red pixels: the BELT_HUE_LO..BELT_HUE_HI / BELT_SAT_MIN / BELT_VAL_MIN gate of
    the belt detector, evaluated arithmetically (hue in [330, 360) u [0, 25] with r the
    max channel <=> (g - b) / (max - min) in [-0.5, 25/60]) instead of through the HSV
    conversion: a 640x360 frame in ~2 ms rather than ~10."""
    r_ = np.asarray(rgb[..., 0], np.float32)
    g_ = np.asarray(rgb[..., 1], np.float32)
    b_ = np.asarray(rgb[..., 2], np.float32)
    mx = np.maximum(np.maximum(r_, g_), b_)
    df = mx - np.minimum(np.minimum(r_, g_), b_)
    gb = g_ - b_
    lo = (BELT_HUE_LO - 360.0) / 60.0
    hi = BELT_HUE_HI / 60.0
    return ((r_ >= mx) & (df > 0) & (gb >= lo * df) & (gb <= hi * df)
            & (df > BELT_SAT_MIN * mx) & (mx > BELT_VAL_MIN * 255.0))


def _toprow_height(K, cam_p, R, d):
    """Height (in the cam_p frame) of the top image row's central ray at depth d: where a
    post truncated by the frame edge must end."""
    fy, cy = K[1, 1], K[1, 2]
    ray = R @ np.array([0.0, (0.0 - cy) / fy, 1.0])
    return float(cam_p[2] + d * ray[2])


def detect_posts(rgb, depth, K, cam_p, cam_rot, add_noise=None, debug=None):
    """Belt stanchions from one RGB-D frame -> [{xy, z_ground, range, n_pix, height, ...}]
    in the frame of cam_p/cam_rot (see the section comment above), nearest first.

    Pipeline: dark-grey pixels with valid depth -> keep those standing more than POST_ABOVE
    above the local ground (dark GROUND texture -- mud ruts, pebble shadows -- drops out
    here, so a post is never merged into the ground it stands on) -> 8-connected components
    -> per blob keep the pixels at the blob's own depth (edge pixels bleed into whatever is
    behind), refine the foot from the dark pixels just below the blob (base disc), then
    require: tall (0.30 m, or 0.10 m when the frame edge truncates it AND its top sits at
    the edge ray's height), thin across the viewing ray (< POST_MAX_WIDTH), thin in pixels
    (per-row width vs what 44 mm subtends), upright (taller than wide), and not floating
    above the neighbourhood ground. `debug`, if a list, receives every candidate blob with
    the gate that rejected it."""
    return _posts_from_prep(_post_prep(rgb, depth, K, cam_p, cam_rot, add_noise), K, debug)


def _posts_from_prep(prep, K, debug=None):
    rgb, depth, R, cam_p, valid, dark, ground, ground_nd = prep
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    ys_d, xs_d = np.nonzero(dark)
    if len(ys_d) < POST_MIN_PIX:
        return []
    Pd, dd = _unproject(depth, K, cam_p, R, ys_d, xs_d)
    # ground under a dark pixel: the lower of the all-pixel grid and the non-dark grid
    # (the latter without the frame-wide fallback, which on the stair platform is the
    # floor 0.6 m down and would stand every dark ground pixel up)
    zg_d = np.fmin(ground.at_many(Pd[:, :2], ring_min=True),
                   ground_nd.at_many(Pd[:, :2], ring_min=True, fallback=False))
    above_d = Pd[:, 2] - zg_d
    stand = above_d > POST_ABOVE
    if int(stand.sum()) < POST_MIN_PIX:
        return []
    mask = np.zeros((H, W), bool)
    mask[ys_d[stand], xs_d[stand]] = True
    lab, n = _label_cc(mask)
    if n == 0:
        return []
    sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    order = np.argsort(-sizes)
    order = order[sizes[order] >= POST_MIN_PIX][:60]
    if len(order) == 0:
        return []
    # index the dark-pixel arrays by image position for the foot refinement
    lab_d = lab[ys_d, xs_d]
    out = []
    for li in order:
        sel = lab_d == li + 1
        ys, xs, P, d = ys_d[sel], xs_d[sel], Pd[sel], dd[sel]
        d_med = float(np.median(d))
        sig = CAM_DEPTH_ABS + CAM_DEPTH_REL * d_med
        tol = max(POST_DEPTH_TOL, 4.0 * sig)
        core = np.abs(d - d_med) < tol
        rec = {"u": float(np.median(xs)), "v_bot": int(ys.max()), "v_top": int(ys.min()),
               "n_blob": int(len(ys)), "d": d_med}
        n_core = int(core.sum())
        if n_core < POST_MIN_PIX:
            rec["reject"] = "core"
            if debug is not None:
                debug.append(rec)
            continue
        Pc, yc, xc = P[core], ys[core], xs[core]
        z_top = float(np.percentile(Pc[:, 2], 97))
        z_low = float(np.percentile(Pc[:, 2], 3))
        # foot: dark pixels at this depth in a window just below the blob (the base disc
        # and the lowest shaft pixels, which the standing test cut off)
        u0, u1 = int(xc.min()) - 3, int(xc.max()) + 3
        v0 = int(yc.max())
        v1 = v0 + max(6, int(0.14 * fy / max(d_med, 0.3)))
        win = ((ys_d >= v0) & (ys_d <= v1) & (xs_d >= u0) & (xs_d <= u1)
               & (np.abs(dd - d_med) < tol + 0.11))
        if int(win.sum()) >= 3:
            z_ground = float(np.percentile(Pd[win, 2], 5))
            z_ground = min(z_ground, z_low - POST_ABOVE)
            foot_src = "disc"
        else:
            z_ground = z_low - POST_ABOVE
            foot_src = "est"
        extent = z_top - z_ground
        top_cut = int(yc.min()) <= 1
        side_cut = int(xc.min()) <= 1 or int(xc.max()) >= W - 2
        # bottom hidden behind something nearer (a step edge, the ramp crest): the rows
        # just under the blob are much closer than the blob
        vb0, vb1 = min(H - 1, v0 + 2), min(H, v0 + 7)
        below = depth[vb0:vb1, max(0, int(xc.min())):int(xc.max()) + 1]
        below = below[np.isfinite(below)]
        bot_cut = bool(len(below) >= 3 and np.median(below) < d_med - POST_OCCLUDER_GAP)
        if bot_cut and foot_src == "disc":
            foot_src = "est"          # what the window found was shaft, not a disc
        hpx = int(yc.max() - yc.min() + 1)
        vis_h = hpx * d_med / fy        # metres of post the blob's rows actually span
        rec.update(n_core=n_core, extent=extent, top_cut=top_cut, side_cut=side_cut,
                   bot_cut=bot_cut, z_ground=z_ground, foot_src=foot_src, vis_h=vis_h)
        if (side_cut or bot_cut) and not top_cut and vis_h < POST_MIN_VIS_H:
            # the relaxed extent gate below must not pass a few rows of a curved object
            # at the frame edge whose z-spread is depth variation, not height
            rec["reject"] = "rows"
        elif top_cut:
            # a truncated post ends where the top row's ray is; a short object that merely
            # touches the top row (nose-down on the slope) ends below it
            z_edge = _toprow_height(K, cam_p, R, d_med)
            rec["z_edge"] = z_edge
            if extent < POST_MIN_EXTENT_CUT:
                rec["reject"] = "short"
            elif z_top < z_edge - POST_EDGE_TOL:
                rec["reject"] = "edge"
        elif extent < (POST_MIN_EXTENT_CUT if (side_cut or bot_cut) else POST_MIN_EXTENT_FULL):
            # the tilted projection leaves the frame sideways at close range, so a post
            # against the left/right edge shows only a slice of its height; likewise one
            # whose lower part a step edge hides
            rec["reject"] = "short"
        elif extent > POST_MAX_EXTENT:
            rec["reject"] = "tall"
        if "reject" in rec:
            if debug is not None:
                debug.append(rec)
            continue
        xy = Pc[:, :2]
        c = np.median(xy, axis=0)
        lo, hi = np.percentile(xy, [10, 90], axis=0)
        foot = float(np.linalg.norm(hi - lo))
        ray = c - cam_p[:2]
        rng = float(np.linalg.norm(ray))
        ray_u = ray / max(rng, 1e-6)
        perp = np.array([-ray_u[1], ray_u[0]])
        w_perp = float(np.diff(np.percentile((xy - c) @ perp, [10, 90]))[0])
        # per-row pixel width of the standing part vs what a 44 mm post subtends
        counts = np.bincount(yc - yc.min())
        w_row = float(np.median(counts[counts > 0]))
        w_exp = POST_DIAMETER * fx / max(d_med, 0.3)
        hpx = int(yc.max() - yc.min() + 1)
        wpx = float(np.diff(np.percentile(xc, [5, 95]))[0]) + 1.0
        z_nb = ground.at(float(c[0]), float(c[1]))
        rec.update(foot=foot, w_perp=w_perp, w_row=w_row, w_exp=w_exp, hpx=hpx, wpx=wpx,
                   z_nb=z_nb, range=rng)
        if foot > POST_MAX_FOOT:
            rec["reject"] = "foot"
        elif w_perp > POST_MAX_WIDTH:
            rec["reject"] = "width"
        elif w_row > max(4.0, POST_MAX_ROW_W * w_exp + 2.0):
            rec["reject"] = "roww"
        elif hpx < POST_MIN_ASPECT * wpx:
            rec["reject"] = "aspect"
        elif np.isfinite(z_nb) and abs(z_ground - z_nb) > POST_MAX_FLOAT:
            rec["reject"] = "floating"
        if "reject" in rec:
            if debug is not None:
                debug.append(rec)
            continue
        # the pixels are the post's near surface: step half a diameter along the ray
        c_axis = c + ray_u * (POST_DIAMETER / 2)
        out.append({"xy": (float(c_axis[0]), float(c_axis[1])), "z_ground": z_ground,
                    "range": rng + POST_DIAMETER / 2, "n_pix": n_core, "height": extent,
                    "top_cut": top_cut, "u": float(np.median(xc)),
                    "v_top": int(yc.min()), "v_bot": int(yc.max()),
                    "width": w_perp, "foot": foot, "z_nb": z_nb, "d": d_med,
                    "foot_src": foot_src, "side_cut": side_cut, "bot_cut": bot_cut})
    # the belt crossing a shaft splits it into two standing blobs: keep the bigger one
    # of any pair closer than POST_MERGE_XY (posts are 2 m apart)
    out.sort(key=lambda r: -r["n_pix"])
    kept = []
    for r in out:
        if all(np.hypot(r["xy"][0] - k["xy"][0], r["xy"][1] - k["xy"][1]) > POST_MERGE_XY
               for k in kept):
            kept.append(r)
        elif debug is not None:
            r["reject"] = "dup"
            debug.append(r)
    kept.sort(key=lambda r: r["range"])
    return kept


def detect_belt_points(rgb, depth, K, cam_p, cam_rot, add_noise=None):
    """3D points on the red belt WITH depth (no plane assumption): red pixels with valid
    depth whose height above the local ground falls in the webbing band, from blobs that
    are ribbon-thin in z (a red bag or the red bin spans 0.27-0.47 m of height, the
    webbing 0.045). -> (N, 3) in the frame of cam_p/cam_rot, or None."""
    rgb, depth, R, cam_p, valid, _dark, ground, _nd = _post_prep(rgb, depth, K, cam_p,
                                                                 cam_rot, add_noise)
    return _belt_points(rgb, depth, K, cam_p, R, valid, ground)


def _belt_points(rgb, depth, K, cam_p, R, valid, ground):
    red = _red_mask(rgb) & valid
    if int(red.sum()) < 5:
        return None
    lab, n = _label_cc(red)
    ys_all, xs_all = np.nonzero(red)
    P, _ = _unproject(depth, K, cam_p, R, ys_all, xs_all)
    zg = ground.at_many(P[:, :2])
    above = P[:, 2] - zg
    band = (above > BELTPT_Z_LO) & (above < BELTPT_Z_HI)
    lab_flat = lab[ys_all, xs_all]
    keep = np.zeros(len(ys_all), bool)
    for li in range(1, n + 1):
        sel = lab_flat == li
        if int(sel.sum()) < 5:
            continue
        zspan = float(np.diff(np.percentile(above[sel], [5, 95]))[0])
        if zspan > BELTPT_MAX_ZSPAN:
            continue
        keep |= sel & band
    if int(keep.sum()) < 5:
        return None
    return P[keep].astype(np.float64)


def detect_corridor(rgb, depth, K, cam_p, cam_rot, add_noise=None, debug=None):
    """Both corridor cues from ONE frame with the shared front half (noise draw, masks,
    ground grids) run once: -> (posts list as detect_posts, belt points as
    detect_belt_points)."""
    prep = _post_prep(rgb, depth, K, cam_p, cam_rot, add_noise)
    posts = _posts_from_prep(prep, K, debug)
    rgb_n, depth_n, R, cam_p, valid, _dark, ground, _nd = prep
    return posts, _belt_points(rgb_n, depth_n, K, cam_p, R, valid, ground)
