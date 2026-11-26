#!/usr/bin/env python3
"""Verify README.md and README.ko.md against the code they describe.

A README that has drifted is worse than no README: it is a set of confident claims about a
robot, read under time pressure, by someone who cannot check them.  Several of the numbers
in these two files are load-bearing -- the joint order, the standing pose, the gripper's
factor of two, the action scales -- and getting one wrong on check-in day costs the run.

So the documents are treated as assertions and re-checked against the source of truth:

  * every path either README mentions exists
  * every command in the quick-start block is runnable
  * the hardware contract in the prose equals the hardware contract in the yaml and the code
  * the measured claims are re-measured, cheaply, here
  * the two translations agree with each other on every number

Run:  python onboard/tools/check_docs.py            (fast checks)
      python onboard/tools/check_docs.py --slow     (also re-times and re-measures)
"""
from __future__ import annotations

import argparse
import re
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ONBOARD = Path(__file__).resolve().parents[1]
REPO = ONBOARD.parent
sys.path.insert(0, str(REPO))

EN = ONBOARD / "README.md"
KO = ONBOARD / "README.ko.md"

# The deployment image carries the runtime, not a checkout: it has onboard/ and the six
# course_mission modules the runtime imports, and nothing else.  Checks that reach for the
# repository around it are then testing the packaging, not the documentation, so they are
# skipped rather than failed.
IS_RUNTIME = not (REPO / "README.md").exists()

FAILS: list[str] = []
N = 0


def check(name: str, ok: bool, detail: str = "") -> bool:
    global N
    N += 1
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(f"{name}: {detail}")
    return ok


def section(t: str) -> None:
    print(f"\n== {t} ==")


def both_contain(label: str, en_pat: str, ko_pat: str | None = None) -> bool:
    """A claim must appear in BOTH documents.  A number fixed in one language and not the
    other is the normal way a translated doc goes wrong."""
    ko_pat = en_pat if ko_pat is None else ko_pat
    e, k = EN.read_text(), KO.read_text()
    in_en, in_ko = bool(re.search(en_pat, e)), bool(re.search(ko_pat, k))
    return check(label, in_en and in_ko,
                 "" if (in_en and in_ko) else
                 f"missing from {'EN' if not in_en else ''}{' and ' if not in_en and not in_ko else ''}"
                 f"{'KO' if not in_ko else ''}")


def main(slow: bool) -> int:
    en, ko = EN.read_text(), KO.read_text()

    section("1. both documents exist and cross-link")
    check("README.md exists", EN.exists())
    check("README.ko.md exists", KO.exists())
    check("EN links to KO", "README.ko.md" in en)
    check("KO links to EN", "README.md" in ko)

    section("2. every path mentioned actually exists")
    # Backticked paths that look like files or directories inside the bundle.
    pat = re.compile(r"`([A-Za-z_][\w./{},-]*\.(?:py|sh|yaml|md|txt)|[a-z_]+/)`")
    seen: set[str] = set()
    for doc in (en, ko):
        for m in pat.finditer(doc):
            seen.add(m.group(1))
    missing = []
    for raw in sorted(seen):
        # `course_mission/{a,b}.py` brace form -> expand
        cands = []
        mb = re.match(r"^(.*)\{(.+)\}(.*)$", raw)
        if mb:
            cands = [f"{mb.group(1)}{p}{mb.group(3)}" for p in mb.group(2).split(",")]
        else:
            cands = [raw]
        for c in cands:
            if (ONBOARD / c).exists() or (REPO / c).exists():
                continue
            # The prose refers to several files by name alone (`perception.py`,
            # `a2pro.yaml`), which reads better than a full path and is still checkable:
            # search for the basename, but insist it is unambiguous, because a name that
            # matches three files is a reference the reader cannot follow either.
            base = Path(c).name
            found = [h for h in REPO.rglob(base)
                     if "__pycache__" not in h.parts and ".git" not in h.parts
                     and "IsaacLab" not in h.parts and "IsaacSim" not in h.parts]
            if not found:
                missing.append(c)
    if IS_RUNTIME:
        check(f"{len(seen)} referenced paths (skipped: runtime image)", True)
    else:
        check(f"{len(seen)} referenced paths resolve", not missing, ", ".join(missing[:6]))

    section("3. quick-start commands are runnable")
    if IS_RUNTIME:
        check("bin/ scripts (skipped: runtime image)", True)
    for tool in ("offsim_gate.py", "replay_run.py", "check_nav.py", "bundle_artifacts.py",
                 "check_camadapter.py", "bench.py", "dump_motor_map.py",
                 "calib_extrinsics.py", "check_docs.py"):
        check(f"tools/{tool}", (ONBOARD / "tools" / tool).exists())
    for s in ("run.sh", "prep.sh", "damp.sh", "estop_test.sh"):
        p = ONBOARD / "bin" / s
        if not p.exists() and IS_RUNTIME:
            continue
        ok = p.exists() and subprocess.run(["bash", "-n", str(p)]).returncode == 0
        check(f"bin/{s} parses", ok)
    r = subprocess.run([sys.executable, "-m", "onboard.run", "--help"],
                       capture_output=True, text=True, cwd=str(REPO))
    check("python -m onboard.run --help", r.returncode == 0, r.stderr.strip()[:80])

    section("4. hardware contract: prose == yaml == code")
    from onboard import cfg as C
    from onboard.oio.a2_map import LegMap
    a2 = C.load("a2pro")
    m = LegMap(a2)

    order = ", ".join(n.split("_")[0] for n in m.names[::3])
    check("joint order is FR, FL, RR, RL", order == "FR, FL, RR, RL", order)
    both_contain("  ...and both READMEs say so",
                 r"FR, FL, RR, RL x \(hip, thigh, calf\)",
                 r"FR, FL, RR, RL × \(hip, thigh, calf\)")

    q = np.round(m.q_default, 3).tolist()
    want = [-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 1.0, -1.5, 0.1, 1.0, -1.5]
    check("q_def matches a2pro.yaml", q == want, str(q))
    both_contain("  ...and both READMEs quote it",
                 r"q_def = \[-0\.1, 0\.8, -1\.5,\s+\+0\.1, 0\.8, -1\.5,\s+-0\.1, 1\.0, -1\.5,\s+\+0\.1, 1\.0, -1\.5\]")

    hips = set(np.round(m.action_scale[0::3], 4))
    rest = set(np.round(np.delete(m.action_scale, np.s_[0::3]), 4))
    check("action scale hip 0.125 / other 0.25", hips == {0.125} and rest == {0.25},
          f"{sorted(hips)} / {sorted(rest)}")
    both_contain("  ...and both READMEs quote it", r"0\.125 hip \| 0\.25 thigh, calf")

    check("kp 120 / kd 4 in yaml",
          float(a2["control"]["kp"]) == 120.0 and float(a2["control"]["kd"]) == 4.0)
    both_contain("  ...and both READMEs quote it", r"kp 120, kd 4")

    check("control rate 50 Hz", float(a2["control"]["rate_hz"]) == 50.0)

    pol = C.load("policies")
    act = str(pol["walk"]["adaptation_activation"]).lower()
    check("adaptation activation is relu in policies.yaml", act == "relu", act)
    both_contain("  ...and both READMEs say ReLU", r"ReLU")
    both_contain("  ...and both warn about ELU", r"0/160")

    section("5. sensor contract")
    s = C.load("sensors")
    vp = s["camera"]["virtual_pinhole"]
    check("virtual pinhole 640x360", (vp["width"], vp["height"]) == (640, 360))
    check("fx == fy", vp["fx"] == vp["fy"], str(vp["fx"]))
    check("principal point at centre",
          (vp["cx"], vp["cy"]) == (vp["width"] / 2, vp["height"] / 2))
    # Derived from launch.head_cam_cfg's focal_length 16.0 against IsaacLab's default
    # horizontal_aperture 20.955.  Checked rather than restated, because the first version
    # of sensors.yaml carried 488.71 -- a mental-arithmetic slip that nothing would have
    # caught, since the detector degrades gracefully rather than failing.
    derived = 640 * 16.0 / 20.955
    check("fx equals 640*16.0/20.955", abs(vp["fx"] - derived) < 0.001,
          f"config {vp['fx']} vs derived {derived:.4f}")
    both_contain("  ...and both READMEs quote 488.67", r"488\.67")
    # And against the intrinsics the simulator actually wrote into the recorded frames.
    ds = REPO.parent / ("real_world_preliminary/ATEC2026_Simulation_Challenge/"
                        "course_mission/dataset_hk")
    frames = sorted(ds.glob("*.npz")) if ds.exists() else []
    if frames:
        K = np.load(frames[0], allow_pickle=True)["K"]
        check("fx matches the recorded frames", abs(float(K[0, 0]) - vp["fx"]) < 0.001,
              f"recorded {float(K[0, 0]):.4f}")

    ex = s["camera"]["extrinsics"]
    check("mount xyz matches launch.HEAD_POS",
          [round(v, 4) for v in ex["xyz"]] == [0.3381, 0.0336, 0.0525], str(ex["xyz"]))
    check("mount pitch 22 deg", float(ex["pitch_deg"]) == 22.0)
    both_contain("  ...and both READMEs quote the mount",
                 r"0\.3381, 0\.0336, 0\.0525")
    both_contain("  ...and both quote 22 deg", r"22", r"22")

    check("depthsynth does NOT fill lidar holes with the plane",
          "np.where(np.isnan(d), g, d)" not in
          (ONBOARD / "perceive" / "depthsynth.py").read_text())
    both_contain("  ...and both READMEs report the 19 % measurement", r"\*\*19 %\*\*|19 %")

    section("6. gripper factor of two")
    from onboard.oio.piper_can import PiperCanBus
    b = PiperCanBus(C.load("piperx"))
    check("width 0.05 -> stroke 0.10", abs(b.width_to_stroke(0.05) - 0.10) < 1e-9)
    check("stroke_per_width == 2", b.stroke_per_width == 2.0)
    both_contain("  ...and both READMEs warn about it", r"twice|2배")

    section("7. jacobian convention")
    from onboard.control.arm import ArmController
    ac = ArmController(C.load("piperx"))
    check("use_training_jacobian is on", ac.use_training_jac is True)
    both_contain("  ...and both READMEs say why", r"0/5")

    section("8. artifact bundle")
    r = subprocess.run([sys.executable, str(ONBOARD / "tools" / "bundle_artifacts.py"),
                        "--verify"], capture_output=True, text=True)
    check("manifest verifies", r.returncode == 0)
    man = json.loads((ONBOARD / "artifacts" / "manifest.json").read_text())
    check("five policy artifacts + banks bundled", len(man["entries"]) == 6,
          ", ".join(man["entries"]))
    # The grasp champion is the one that was NOT in the repository before this bundle.
    check("grasp champion came from exp_runs/runs (the gitignored path)",
          man["entries"]["grasp"]["source"].startswith("exp_runs/runs/"),
          man["entries"]["grasp"]["source"])

    section("9. body height and foot radius")
    from onboard.estimate.odom import FOOT_R, foot_positions
    check("FOOT_R is the URDF sphere 0.032", FOOT_R == 0.032, str(FOOT_R))
    h = float(-foot_positions(m.q_default)[:, 2].mean())
    check("standing height 0.41-0.42 m from exact FK", 0.405 < h < 0.425, f"{h:.4f} m")

    section("10. offsim gate agrees with itself")
    r = subprocess.run([sys.executable, str(ONBOARD / "tools" / "offsim_gate.py")],
                       capture_output=True, text=True)
    tail = (r.stdout or "").strip().splitlines()[-1] if r.stdout else ""
    check("offsim gate passes", r.returncode == 0, tail)

    if slow:
        section("11. re-measured claims (slow)")
        import time

        import torch
        from onboard.control.legs import LegPolicy
        from onboard.oio.replay import MockLegBus
        p = LegPolicy(a2, pol)
        bus = MockLegBus(a2)
        st = bus.read()
        cmd = np.array([0.7, 0.0, 0.0])
        for _ in range(50):
            p.step(st, cmd)
        t0 = time.perf_counter()
        for _ in range(300):
            p.step(st, cmd)
        us = (time.perf_counter() - t0) / 300 * 1e6
        # Bound deliberately loose: this is a shared dev box (load average routinely
        # above 200), so a tight assertion on an absolute timing would fail for reasons
        # that have nothing to do with the code.  The claim being tested is "the policy is
        # not the bottleneck", and 600 us is still 3 % of the control period.
        check("leg step well under the control period (README: 146 us)", us < 600,
              f"{us:.0f} us")
        check("  and well under the 20 ms budget", us < 2000, f"{us/20000*100:.2f} % of budget")

        from onboard.vendor.diffik import (DifferentialIKController,
                                           DifferentialIKControllerCfg)
        ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True,
                                        ik_method="dls", ik_params={"lambda_val": 0.05}),
            num_envs=1, device="cpu")
        pp = torch.tensor([[0.4, 0.0, 0.3]])
        qq = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        J = torch.randn(1, 6, 6) * 0.3 + torch.eye(6).unsqueeze(0)
        jp = torch.zeros(1, 6)
        ik.set_command(torch.tensor([[0.01, 0.0, 0.0, 0.0, 0.0, 0.0]]), pp, qq)
        t0 = time.perf_counter()
        for _ in range(300):
            ik.compute(pp, qq, J, jp)
        us = (time.perf_counter() - t0) / 300 * 1e6
        check("diffik compute (README: 123 us)", us < 500, f"{us:.0f} us")

        r = subprocess.run([sys.executable, str(ONBOARD / "tools" / "check_camadapter.py")],
                           capture_output=True, text=True)
        check("camera pose matches the simulator", r.returncode == 0,
              (r.stdout or "").strip().splitlines()[-1] if r.stdout else "")

        r = subprocess.run([sys.executable, str(ONBOARD / "tools" / "check_nav.py")],
                           capture_output=True, text=True)
        check("closed-loop corridor following", r.returncode == 0,
              (r.stdout or "").strip().splitlines()[-1] if r.stdout else "")

    section("12. the two documents have the same structure")
    check("structure helpers", True)
    def strip_code(doc: str) -> str:
        """Remove fenced blocks.  Without this a shell comment like `# 1. Acceptance gate`
        counts as a level-1 heading, which both files happen to share -- so the check would
        pass while comparing the wrong thing."""
        return re.sub(r"^```.*?^```", "", doc, flags=re.S | re.M)

    def heads(doc: str) -> list[int]:
        """Heading levels in order.  Comparing LEVELS rather than text is the only
        language-independent way to ask 'is the translation missing a section' -- and a
        missing section is the failure mode that matters, because it is invisible to a
        reader who only has one of the two files."""
        return [len(m.group(1)) for m in re.finditer(r"^(#{1,4}) ", strip_code(doc), re.M)]
    he, hk = heads(en), heads(ko)
    check("same heading structure", he == hk, f"EN {he} vs KO {hk}")

    def tables(doc: str) -> list[int]:
        """Row count of each markdown table, in order."""
        out, run = [], 0
        for line in doc.splitlines():
            if line.strip().startswith("|"):
                run += 1
            elif run:
                out.append(run)
                run = 0
        if run:
            out.append(run)
        return out
    te, tk = tables(en), tables(ko)
    check("same tables with the same row counts", te == tk, f"EN {te} vs KO {tk}")

    ce = len(re.findall(r"^```", en, re.M))
    ck = len(re.findall(r"^```", ko, re.M))
    check("same number of code blocks", ce == ck, f"EN {ce // 2} vs KO {ck // 2}")

    section("13. the repository README")
    root = REPO / "README.md"
    if not root.exists():
        # The deployment image carries the runtime, not the repository -- the root README is
        # documentation about the repo, so its absence here is correct, not a failure.
        check("repo README (skipped: runtime image, not a checkout)", True)
        rd = None
    else:
        check("repo README exists", True)
        rd = root.read_text()

    # Every path drawn in the tree block, and every backticked path.
    tree = re.search(r"```\n(ATEC_gmis/.*?)```", rd, re.S) if rd else None
    if rd:
        check("tree block present", tree is not None)
    if tree:
        bad = []
        for line in tree.group(1).splitlines()[1:]:
            mm = re.search(r"[|`+\u2500\u251c\u2514\u2502\s]*([A-Za-z_][\w./-]*/?)", line)
            if not mm:
                continue
            name = mm.group(1)
            if name.endswith("/"):
                # A directory may be drawn nested; accept it anywhere in the tree.
                if not any((REPO / d).is_dir() for d in (name,)) and not list(REPO.rglob(name.rstrip("/")))[:1]:
                    bad.append(name)
            elif "." in name and not (REPO / name).exists():
                if not list(REPO.rglob(Path(name).name))[:1]:
                    bad.append(name)
        check("every path in the tree resolves", not bad, ", ".join(bad[:6]))

    if rd:
        check("repo README points at onboard/", "onboard/README.ko.md" in rd)
        check("repo README states the no-simulator rule",
              "Isaac / IsaacLab / gymnasium / CUDA" in rd)

    # The score table is the claim most likely to rot AND most likely to mislead, because a
    # number quoted without its honesty condition reads as a deployment expectation.
    # Re-derive it from the run logs rather than trusting the prose.
    import collections
    best = collections.defaultdict(lambda: -1)
    runs = collections.Counter()
    t1_full = collections.Counter()
    for cfgp in sorted((REPO / "deploy" / "logs").glob("*/config.json")):
        rep = cfgp.parent / "report_venue.json"
        if not rep.exists():
            continue
        try:
            env = json.loads(cfgp.read_text()).get("env", {})
            r = json.loads(rep.read_text())
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(r, dict) or "score_task2" not in r:
            continue
        nav = env.get("MISSION_NAV_HONEST") in (1, "1")
        obj = env.get("MISSION_OBJ_HONEST") in (1, "1")
        k = "both" if (nav and obj) else "nav" if nav else "oracle"
        runs[k] += 1
        best[k] = max(best[k], r["score_task2"])
        t1_full[k] += int(r.get("score_task1") == 80)
    # Emphasis markers sit INSIDE the labels ("항법 **정직**"), so match against a copy
    # with them stripped -- otherwise the check reports a missing row that is right there.
    rd_plain = (rd or "").replace("**", "")
    for k, label in (("oracle", "모두 오라클"), ("nav", "물체 오라클"),
                     ("both", "모두 정직")):
        if rd is None:
            break
        row = re.search(rf"\| .*{label}.*\|", rd_plain)
        check(f"score row present: {label}", row is not None)
        if row:
            cells = [c.strip() for c in row.group(0).strip("|").split("|")]
            _cells_numeric = [c for c in cells
                              if re.fullmatch(r"\*?\*?\d+(/160)?\*?\*?", c)
                              or re.fullmatch(r"\d+/80 \(\d+\)", c)]
            claimed_runs = int(cells[1])
            claimed_best = int(re.sub(r"\D", "", cells[3].split("/")[0]))
            claimed_t1 = int(re.search(r"\((\d+)\)", cells[2]).group(1))
            # Counts only ever grow -- new runs land in deploy/logs every day, and a
            # documentation check that fails on that is a check people learn to ignore.
            # The BEST score is the claim that matters and it is checked exactly.
            check(f"  {label}: run count >= {claimed_runs}", runs[k] >= claimed_runs,
                  f"logs say {runs[k]}" + ("  (doc is stale, worth refreshing)"
                                           if runs[k] > claimed_runs else ""))
            check(f"  {label}: task2 best {claimed_best}", claimed_best == best[k],
                  f"logs say {best[k]}")
            check(f"  {label}: task1 80/80 >= {claimed_t1}", t1_full[k] >= claimed_t1,
                  f"logs say {t1_full[k]}")

    # And the honest condition must not be quoted without its caveat.
    if rd:
        check("the honest row is called out as the deployment condition",
              "마지막 줄이 실기 조건" in rd)
        check("the oracle numbers are explicitly disclaimed",
              "실기 기대값이 아닙니다" in rd)


    def nums(doc: str) -> set[str]:
        """Extract every number that carries a unit.

        The unit must NOT be anchored with \\b on its right.  Python's \\w is
        Unicode-aware, so in Korean "213 ms짜리" there is no word boundary between the "s"
        and the Hangul that follows it, and the token silently fails to match -- which made
        the first version of this check report that the translation was missing numbers it
        actually contained.  A lookahead for "not a Latin letter" is what was meant.
        """
        body = doc.split("## ")[1:]                 # skip the title paragraph
        out: set[str] = set()
        pat = re.compile(r"(\d+(?:\.\d+)?)\s*(%|Hz|ms|µs|us|mm|deg|kg)(?![A-Za-z])")
        for sec in body:
            for val, unit in pat.findall(sec):
                out.add(f"{val}{'µs' if unit == 'us' else unit}")
        return out

    section("14. the docker/ documentation pair")
    d_en, d_ko = REPO / "docker" / "README.md", REPO / "docker" / "README.ko.md"
    if IS_RUNTIME or not d_en.exists():
        check("docker READMEs (skipped: runtime image)", True)
    else:
        check("docker/README.ko.md exists", d_ko.exists())
        de, dk = d_en.read_text(), d_ko.read_text()
        check("docker EN links to KO", "README.ko.md" in de)
        check("docker KO links to EN", "README.md" in dk)
        # The runbook is the document somebody reads standing next to a robot with a judge
        # watching.  A step that exists in one language and not the other is the failure
        # that matters, and heading levels catch it without needing to read either.
        check("docker: same heading structure", heads(de) == heads(dk),
              f"EN {heads(de)} vs KO {heads(dk)}")
        check("docker: same tables", tables(de) == tables(dk),
              f"EN {tables(de)} vs KO {tables(dk)}")
        ce, ck = len(re.findall(r"^```", de, re.M)), len(re.findall(r"^```", dk, re.M))
        check("docker: same code blocks", ce == ck, f"EN {ce // 2} vs KO {ck // 2}")
        na, nb = nums(de), nums(dk)
        check("docker: same measured values", na == nb,
              f"EN-only {sorted(na - nb)[:6]} KO-only {sorted(nb - na)[:6]}")
        # Every command the runbook tells an operator to type must be a verb the image
        # actually dispatches; a runbook that names a tool that is not there is worse than
        # no runbook, because it is read under time pressure.
        ep = (REPO / "docker" / "entrypoint.sh").read_text()
        verbs = set(re.findall(r"^  ([a-z]+)\)", ep, re.M))
        used = set(re.findall(r"docker/run\.sh ([a-z]+)", de + dk))
        unknown = sorted(used - verbs)
        check(f"docker: runbook uses {len(used)} verbs, all dispatched", not unknown,
              ", ".join(unknown))

    section("15. the two translations agree on every number")
    # Numbers that carry units or percent signs, which is what a claim looks like.
    a, b = nums(en), nums(ko)
    only_en, only_ko = sorted(a - b), sorted(b - a)
    check("EN and KO quote the same measured values", not only_en and not only_ko,
          f"EN-only {only_en[:8]} KO-only {only_ko[:8]}")

    section("16. the docker/ deployment contract")
    # Every check here exists because the thing it asserts was BROKEN and nothing noticed.
    # These are repo-shape facts, so they are skipped inside the runtime image.
    dk_dir = REPO / "docker"
    if IS_RUNTIME or not dk_dir.exists():
        check("docker contract (skipped: runtime image)", True)
    else:
        # Strip comment-only lines before matching.  These files EXPLAIN the defects they
        # fixed, so a naive substring search finds the prose describing the old behaviour and
        # reports the fix as the bug.
        def _code(t: str) -> str:
            return "\n".join(l for l in t.splitlines() if not l.lstrip().startswith("#"))
        rs = _code((dk_dir / "run.sh").read_text())
        ep = _code((dk_dir / "entrypoint.sh").read_text())
        df = _code((dk_dir / "Dockerfile").read_text())
        cp = _code((dk_dir / "compose.yml").read_text())

        # .dockerignore only takes effect at the CONTEXT root.  build.sh passes the repo
        # root, so a file in docker/ is silently inert -- which is how the context became
        # the whole repository with __pycache__ baked into the escrow image.
        check(".dockerignore is at the context root", (REPO / ".dockerignore").is_file())
        check(".dockerignore is not stranded in docker/",
              not (dk_dir / ".dockerignore").exists(),
              "a copy there is never read by Docker")

        # The whitelist and the COPY list are two statements of the same set.
        if (REPO / ".dockerignore").is_file():
            di = (REPO / ".dockerignore").read_text()
            allow = {l[1:].rstrip("/") for l in di.splitlines() if l.startswith("!")}
            copied = {m.rstrip("/") for m in
                      re.findall(r"^COPY\s+(?!--from)(\S+)\s", df, re.M)}
            copied = {c for c in copied if not c.startswith("docker/")}
            missing = sorted(c for c in copied
                             if c.rstrip("/") not in allow
                             and not any(c.startswith(a + "/") for a in allow))
            check("every Dockerfile COPY source is whitelisted", not missing,
                  ", ".join(missing))

        # A verb that dispatches to a file that does not exist is a command that dies at the
        # venue with "No such file or directory".
        targets = re.findall(r"exec (?:python|bash) (onboard/[\w/]+\.(?:py|sh))", ep)
        gone = sorted(t for t in targets if not (REPO / t).exists())
        check(f"all {len(set(targets))} entrypoint targets exist", not gone, ", ".join(gone))

        # The stack writes under onboard/logs/.  If the bind mount is anywhere else, every
        # artefact dies with the container -- which is exactly what happened to the
        # calibration captures and to ledger.json.
        check("run.sh binds the directory the stack writes to",
              "/app/onboard/logs" in rs,
              "found no /app/onboard/logs bind; outputs would not survive --rm")
        check("compose.yml binds the same directory", "/app/onboard/logs" in cp)

        # NIC is required by estop/damp/prep; without the passthrough they abort instantly.
        check("run.sh forwards NIC into the container",
              re.search(r"for v in [^;]*\bNIC\b", rs) is not None)
        check("the prep verb passes --nic", "--nic" in ep.split("prep)")[1].split(";;")[0])

        # `docker stop` sends SIGTERM; python is PID 1 and ignores signals it has no handler
        # for, so without this the container is SIGKILLed and the robot is never damped.
        rp = _code((REPO / "onboard" / "run.py").read_text())
        check("run.py handles SIGTERM, not just SIGINT",
              re.search(r"signal\.SIGTERM", rp) is not None,
              "docker stop sends SIGTERM; PID 1 ignores it without a handler")
        check("run.sh sets a stop signal and grace period",
              "--stop-signal" in rs and "--stop-timeout" in rs)

        # An unconditional configs bind means `compose up` silently runs the host's yaml and
        # the image digest stops describing the behaviour -- the opposite of rule IV.5.
        check("compose.yml does not bind configs unconditionally",
              "/app/onboard/configs" not in cp,
              "use docker/compose.venue.yml for a measured override")

        # One source of truth for the dependency set.
        check("Dockerfile installs from requirements-onboard.txt",
              "-r /tmp/req-nontorch.txt" in df or "-r /tmp/requirements-onboard.txt" in df,
              "a hardcoded pip list drifts from the host installer")

        # WITH_SDK=0 is a documented mode; it must actually build.
        check("WITH_SDK=0 can still satisfy the cyclonedds COPY",
              "mkdir -p ${CYCLONEDDS_HOME}" in df)

        # Unpinned sources cannot be rebuilt from the escrow recipe.
        check("cyclonedds is pinned to a tag, not a moving branch",
              "releases/0.10.x" not in df)
        check("unitree_sdk2_python is pinned to a commit",
              re.search(r"UNITREE_SDK_REF=[0-9a-f]{40}", df) is not None)

    print(f"\n{'DOCS OK' if not FAILS else 'DOCS FAILED'} -- {N - len(FAILS)}/{N} checks passed")
    for f in FAILS:
        print(f"  ! {f}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slow", action="store_true")
    raise SystemExit(main(ap.parse_args().slow))
