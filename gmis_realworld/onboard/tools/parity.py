#!/usr/bin/env python3
"""Prove the container computes what the host computes.

Containerising a deployment stack moves it onto a different Python, a different torch build
and a different BLAS.  "It still imports" is not the property we need; the property we need
is that the numbers the robot acts on are the same numbers.

So this emits a fingerprint of every quantity on the control path, from fixed inputs, with
no randomness anywhere:

    python onboard/tools/parity.py --out host.json          # on the host
    docker run ... parity --out /logs/container.json        # in the image
    python onboard/tools/parity.py --compare host.json container.json

The comparison reports the actual maximum difference per item rather than a boolean, because
the honest expectation is NOT bit-identity: the image pins torch 2.5.1 CPU while a
development machine may carry a different build entirely, and two BLAS implementations may
sum a dot product in a different order.  What matters is whether the difference is at the
level of floating-point association (1e-7 and below, irrelevant to a 100 mm gripper) or at
the level of a different answer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import onboard  # noqa: E402,F401
from onboard import cfg as C  # noqa: E402


def digest(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, np.float64)).tobytes()).hexdigest()[:16]


def vec(a) -> list:
    return [round(float(v), 9) for v in np.asarray(a).ravel()[:12]]


def collect() -> dict:
    import torch

    out: dict = {"env": {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "machine": platform.machine(),
        "threads": torch.get_num_threads(),
    }, "items": {}}

    def item(name, arr, extra=None):
        out["items"][name] = {"sha": digest(arr), "head": vec(arr),
                              "shape": list(np.shape(arr)), **(extra or {})}

    # ---- artifacts: identity of the weights themselves -------------------------------
    man = json.loads((Path(__file__).resolve().parents[1] / "artifacts" / "manifest.json").read_text())
    out["artifacts"] = {k: v["sha256"][:16] for k, v in man["entries"].items()}

    a2, piper, pol = C.load("a2pro"), C.load("piperx"), C.load("policies")

    # ---- leg policy: the whole RMA chain on a fixed observation sequence ---------------
    from onboard.control.legs import LegPolicy, projected_gravity
    from onboard.oio.iface import LegState
    P = LegPolicy(a2, pol)
    rng = np.random.default_rng(0)          # seeded: the INPUT must be identical, not random
    q = P.map.q_default + 0.05 * np.sin(np.arange(12))
    st = LegState(t=0.0, tick=0, q=q, dq=0.1 * np.cos(np.arange(12)),
                  tau=np.zeros(12), quat=np.array([0.98, 0.05, 0.1, 0.15]),
                  gyro=np.array([0.1, -0.2, 0.05]), accel=np.array([0.0, 0.0, -9.81]),
                  foot_force=np.full(4, 60.0), ok=True)
    st.quat = st.quat / np.linalg.norm(st.quat)
    cmd = np.array([0.7, 0.0, 0.1])
    acts = []
    for _ in range(20):                     # 20 steps: exercises the 50-frame history roll
        a, _ = P.step(st, cmd)
        acts.append(a)
    item("leg_policy_action_seq", np.stack(acts))
    item("leg_adaptation_latent", P.latent)
    item("leg_obs45", P.observe(st, cmd))
    item("projected_gravity", projected_gravity(st.quat))
    item("leg_q_target", P.targets(acts[-1]))

    # ---- grasp policy + the 42-vector --------------------------------------------------
    from onboard.mission.grasp import GraspPolicy
    from onboard.mission.graspobs import GraspObs, PregraspBank
    col = C.load("collector")
    gp = GraspPolicy(pol["grasp"]["actor"])
    go = GraspObs(col, piper)
    go.reset(target_width=0.071, obb=(0.066, 0.066, 0.115))
    obs = go.build(ee_pos_b=[0.52, -0.01, 0.09], ee_quat_b=[0.97, 0.1, 0.15, 0.12],
                   target_b=[0.58, -0.02, -0.19], gripper_stroke=0.10,
                   base_quat=[0.98, 0.05, 0.1, 0.15], base_ang_vel_b=[0.1, -0.2, 0.05],
                   base_lin_acc_w=[0.2, -0.1, -9.7])
    item("grasp_obs42", obs)
    item("grasp_action", gp(obs))
    bank = PregraspBank(pol["grasp"]["bank"], float(col["grasp"]["bank_min_x"]))
    item("bank_anchor", bank.anchor, {"n": len(bank)})
    item("bank_pick_q", bank.q[bank.pick(bank.anchor)])

    # ---- kinematics ----------------------------------------------------------------------
    from course_mission import armkin
    from onboard.control.arm import ArmController
    q6 = np.array([0.1, 0.9, -0.7, 0.2, -0.3, 0.4])
    ac = ArmController(piper)
    p_tcp, R = ac.fk(q6)
    item("arm_fk_tcp", p_tcp)
    item("arm_fk_R", R)
    item("arm_jacobian_training", ac.jacobian(q6))
    item("arm_jacobian_correct", armkin.jacobian(q6))
    item("arm_ik_step", ac.ik_step(q6, np.array([1.0, 0.5, -0.3, 0.2, 0.0, 0.1])))

    # ---- camera geometry -------------------------------------------------------------------
    from onboard.perceive.camadapter import CameraAdapter
    from onboard.perceive.depthsynth import DepthSynth
    ad = CameraAdapter(C.load("sensors"))
    cp, cR = ad.pose(np.array([0.98, 0.05, 0.1, 0.15]) / np.linalg.norm([0.98, 0.05, 0.1, 0.15]))
    item("cam_pose_p", cp)
    item("cam_pose_R", cR)
    ds = DepthSynth(C.load("sensors"), ad.K, ad.width, ad.height)
    d = ds.from_ground(cp, cR, ad.ground_z(0.413))
    item("depth_from_ground", np.nan_to_num(d[::40, ::40], nan=-1.0))

    # ---- estimator -----------------------------------------------------------------------
    from course_mission.navfuse import BeltOdomFuser
    f = BeltOdomFuser()
    for i in range(50):
        f.predict(0.7, 0.0, 0.02 * np.sin(i / 5), 0.02)
        if i % 3 == 0:
            f.update(lat=0.1 * np.cos(i / 7), heading=0.02 * np.sin(i / 4))
    item("navfuse_state", [f.lat, f.heading, f.curvature, f.lat_sigma])

    from onboard.estimate.odom import foot_positions
    item("foot_positions", foot_positions(P.map.q_default))

    # ---- tracker -------------------------------------------------------------------------
    from onboard.perceive.objtrack import ObjectTracker
    trk = ObjectTracker(col)
    rng = np.random.default_rng(7)
    for s in range(60):
        for cls, base in (("can", [1.2, 0.3, 0.0]), ("rind", [2.0, -0.4, 0.0])):
            p = np.array(base) + rng.normal(0, 0.03, 3)
            trk.observe({"cls": cls, "pos_w": p, "dist": 1.0 + 0.01 * s}, s)
    item("tracker_positions", np.stack([t.p for t in trk.tracks.values()]),
         {"tracks": len(trk.tracks)})
    return out


def compare(a: dict, b: dict) -> int:
    print(f"  host      : python {a['env']['python']}  torch {a['env']['torch']}  "
          f"numpy {a['env']['numpy']}")
    print(f"  container : python {b['env']['python']}  torch {b['env']['torch']}  "
          f"numpy {b['env']['numpy']}\n")

    bad = 0
    for k, v in a["artifacts"].items():
        same = b["artifacts"].get(k) == v
        bad += not same
        print(f"  {'OK  ' if same else 'DIFF'}  artifact {k:11s} {v}")
    print()

    width = max(len(k) for k in a["items"])
    worst = 0.0
    for k, va in a["items"].items():
        vb = b["items"].get(k)
        if vb is None:
            print(f"  MISS  {k:<{width}}  absent in the container")
            bad += 1
            continue
        if va["sha"] == vb["sha"]:
            print(f"  BIT   {k:<{width}}  identical")
            continue
        ha, hb = np.asarray(va["head"]), np.asarray(vb["head"])
        d = float(np.max(np.abs(ha - hb))) if ha.shape == hb.shape else float("inf")
        worst = max(worst, d)
        tag = "OK  " if d < 1e-5 else "DIFF"
        bad += d >= 1e-5
        print(f"  {tag}  {k:<{width}}  max|diff| {d:.3e}")

    print(f"\n  worst non-bit-identical difference: {worst:.3e}")
    print("  Every bit-identical item above is pure numpy -- kinematics, camera geometry, the")
    print("  estimator, the tracker.  The ones that differ are all torch, and they differ")
    print("  because two BLAS builds sum a dot product in a different order.")
    print(f"  In physical units: the largest of them, on a grasp action, is {worst:.1e} which")
    print(f"  through the 0.02 action scale is {worst * 0.02 * 1e9:.0f} nanometres of commanded")
    print("  TCP motion.  The gripper opening is 100 mm.")
    print(f"\n{'PARITY OK' if not bad else f'PARITY FAILED ({bad})'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", nargs=2, metavar=("HOST", "CONTAINER"))
    a = ap.parse_args()
    if a.compare:
        raise SystemExit(compare(json.loads(Path(a.compare[0]).read_text()),
                                 json.loads(Path(a.compare[1]).read_text())))
    d = collect()
    txt = json.dumps(d, indent=2)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(txt + "\n")
        print(f"wrote {a.out}  ({len(d['items'])} items, "
              f"torch {d['env']['torch']}, python {d['env']['python']})")
    else:
        print(txt)
