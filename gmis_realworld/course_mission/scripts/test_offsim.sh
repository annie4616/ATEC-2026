#!/bin/bash
# WP1 acceptance gate: everything that must work on a machine with NO Isaac Sim, NO GPU
# and no simulator packages at all — i.e. the robot's on-board PC.
#
#   bash course_mission/scripts/test_offsim.sh
#
# Uses a plain python (numpy + torch only). If this passes, the perception half, the
# course geometry and the vendored maths are deployable; what remains simulator-bound is
# the arm (PhysX Jacobian, DEPLOY.md WP2) and the motor bridge (WP5).
set -u
RW=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=${PY:-/home/yewon/.conda/envs/ltpo/bin/python3}
cd "$RW"
fail=0

echo "== python =="
$PY -c "import sys, numpy, torch; print(' ', sys.version.split()[0], 'numpy', numpy.__version__, 'torch', torch.__version__)"
$PY -c "import gymnasium" 2>/dev/null && echo "  NOTE gymnasium IS installed here — the no-gym path is not being exercised" \
  || echo "  gymnasium absent (good: this is the robot's situation)"

echo "== 1. modules that must import with no simulator =="
$PY - <<'EOF' || fail=1
import sys
sys.path.insert(0, ".")
ok = True
for m in ("course_mission.catalogue", "course_mission.deploy", "course_mission.perception",
          "course_mission.mathx", "course_mission.navfuse", "course_sim.terrain"):
    try:
        __import__(m)
        print("   OK  ", m)
    except Exception as e:
        print("   FAIL", m, type(e).__name__, e)
        ok = False
assert "isaaclab" not in sys.modules, "something pulled in isaaclab"
sys.exit(0 if ok else 1)
EOF

echo "== 2. course geometry values (must match the simulator's) =="
$PY - <<'EOF' || fail=1
import sys
sys.path.insert(0, ".")
import course_sim.terrain as ct
exp = {"COURSE_LEN": 36.0, "COURSE_X0": -18.0, "SECTION_LEN": 6.0,
       "BOUNDARY_Y": 1.9775, "TASK1_MAX": 80, "FINISH_ARC": 16.2, "BIN_HEIGHT": 0.47}
bad = [(k, getattr(ct, k), v) for k, v in exp.items() if abs(float(getattr(ct, k)) - v) > 1e-9]
cm = ct.curve_map(3.0, 0.5)
if abs(cm[0] - 3.2671831851610547) > 1e-9 or abs(cm[1] + 1.518516353572084) > 1e-9:
    bad.append(("curve_map(3.0,0.5)", cm, (3.2671831851610547, -1.518516353572084)))
for k, got, want in bad:
    print(f"   FAIL {k}: {got} != {want}")
print("   OK   36 m HK course, 5 checkpoints, 80 pts, S-curve intact" if not bad else "")
sys.exit(1 if bad else 0)
EOF

echo "== 3. vendored maths == isaaclab maths =="
$PY course_mission/scripts/check_mathx.py > /tmp/_mathx.$$ 2>&1 && tail -1 /tmp/_mathx.$$ || { cat /tmp/_mathx.$$; fail=1; }
rm -f /tmp/_mathx.$$

echo "== 4. belt+odometry fusion self-test =="
$PY course_mission/navfuse.py || fail=1

echo "== 5. detector scores recorded frames end to end =="
if [ -d "$RW/course_mission/dataset_final" ]; then
  $PY course_mission/scripts/tune_detector.py --data course_mission/dataset_final --limit 20 --seeds 1 \
    2>&1 | grep -E "precision|recall" | head -3 || fail=1
else
  # The recorded-frame dataset lives only in the original dev tree; this clone has no
  # frames to score, so the step is a skip, not a failure.
  echo "SKIP: course_mission/dataset_final not present in this tree"
fi

echo
[ "$fail" -eq 0 ] && echo "OFFSIM OK — deployable surface intact" || echo "OFFSIM FAILURES ABOVE"
exit $fail
