"""Solution v5 — the same mission, re-cut for the 30-MINUTE venue slot.

30 min / 0.02 s per step (sim.dt 0.005 x decimation 4) = 90,000 steps, and the clock does
not pause. v4 was tuned against a 260,000-step budget that turned out to be dead code, so
its knobs authorise about three times the work the venue allows. Measured on the six
persist runs (videos_persist/*/report_persist.json), the damage is not that runs overran —
they ended at 24k-86k — but that they spent the budget in the wrong places:

  * every run reached its FINAL score by ~step 50,000 and then produced nothing for the
    remaining 3,400-61,800 steps
  * zero of 21 scoring unloads happened after the first "sweeping again" re-sweep
  * of 12 targetable objects a run ever attempted 5-7, oracle detection included

So the fix is not to hurry. It is to stop paying for work that has never scored, and to
harvest the objects that are cheap to deliver.

WHERE THE MONEY IS — AND WHERE IT IS NOT. The obvious lever looks like haul distance:
scatter_layout (env_cfg.py:110) puts one copy of each class in mud, pebble, grass and
mixed, both bins sit at arc +17.4, and the search only walks forward (runner.py), so the
robot always harvests the copies FARTHEST from the bins:

    class copy   arc      round trip      cost at 85 steps/m + 620
    mud        -17.2        ~69 m            ~6,500 steps
    pebble     -12.5        ~60 m            ~5,700
    grass       -5.7        ~46 m            ~4,500
    mixed      +12.7        ~9.4 m           ~1,400

Skipping the far copies saves ~18,700 steps. It also LOSES POINTS, so it is not done:
every persist run banked its last scoring unload by step 41,500 of 88,000, leaving 46,500+
steps unused, while the mud copies produced 34% of all targeted objects, 36% of all scored
ones, and the highest target->score conversion of any section (73%). Optimising steps here
optimises a constraint that never binds, and pays for it in objects.

What actually caps the score is COVERAGE: across six runs the mission targeted 5.33 of 12
scorable objects, and target->score conversion is 0.69, so
    E(points) = 20 x SUM_class E[min(2, Binomial(n_class, 0.69))]
predicts the observed mean of 70.8 to within 2%. Raising n_class from ~1.8 to 3 is worth
about +33 points and takes P(120) from ~0 to ~45%. Everything in this profile and the
guards added to runner.py serves that: never walk off the course (which killed 2 of 3
completed runs and froze the detector for 4,500-11,500 steps in half of them), never let a
phantom detection blacklist real ground, never step over a remembered object, and scan on
the legs that were already being walked anyway.

MISSION_HARVEST is kept only for A/B; "all" is what ships:
    all    (default) nothing excluded
    near             mud excluded          -> REFUTED, see above
    far              mud + pebble excluded -> refuted harder; retire

NOT changed, deliberately: the basket stays off. pick_from_basket (runner.py:~1012) servos
to the object origin + 5 mm and closes fully, never consulting grasp_hint — that is 25 mm
below a lunchbox lid, the exact depth catalogue.py:73-75 records as sliding out every
time, and videos_auto events 71-74 show it: 0.007 m servo residual, object rose +0.000 m,
4 tries, all missed. Gripper carry delivers 21/23 overall and 7/7 lunchboxes; the basket
delivers 4/6 and 0/1 lunchboxes. It would trade 40 points of lunchbox for ~6,500 steps.

Run:  scripts/run_mission_venue.sh          (ORACLE=1 to isolate perception)
"""

import argparse
import sys
from pathlib import Path

_S = Path(__file__).resolve()
RW = _S.parents[1]
for _p in (str(RW / "source" / "atec_rl_lab"), str(RW / "eunha_rl"), str(RW / "a2pro_rl"), str(RW)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Course mission — 30-minute venue profile.")
parser.add_argument("--speed", type=str, default="s075")
parser.add_argument("--depth", type=str, default="0.24")
parser.add_argument("--seed", type=int, default=0)
# default=None -> env_cfg.PER_TYPE, which is 3 on the Hong Kong course (guide 265:
# three sets of the four target categories) and 4 on the legacy one. Hard-coding 4
# here silently overrode that.
parser.add_argument("--per_type", type=int, default=None)
parser.add_argument("--max_picks", type=int, default=7)
parser.add_argument("--walk_ckpt", type=str, default="")
parser.add_argument("--grasp_ckpt", type=str, default="")
parser.add_argument("--record_head", action="store_true")
parser.add_argument("--out_dir", type=str, default=str(RW / "course_mission" / "videos_venue"))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os  # noqa: E402

# MISSION_USE_BASKET=1 puts the rear side-saddle basket back on the robot and carries
# several objects per trip (MISSION_MAX_CARRY, default 4). The gripper-carry default
# stays because it was measured better per-object (21/23 vs the basket's 4/6 deposits) —
# but at max_carry=1 every object costs a full round trip to the bins, and under the
# 30-minute budget the trip economics can dominate the per-object reliability. This
# switch exists to measure that trade end to end.
if os.environ.get("MISSION_USE_BASKET") != "1":
    os.environ["MISSION_NO_BASKET"] = "1"

from course_mission.catalogue import TERRAIN_RANK  # noqa: E402
from course_mission.launch import build_and_run  # noqa: E402
from course_mission.runner import Profile  # noqa: E402

# Excluding a terrain rather than merely deprioritising it: terrain_allowed() rejects any
# section whose rank exceeds max_terrain_rank (2), so pushing mud to 3 takes its objects
# off the table entirely. That is the intent — a mud copy is never worth 6,500 steps when
# the same class is sitting next to the bins.
# DEFAULT REVERTED TO "all". The step-economy argument above is arithmetically right and
# strategically wrong: MEASURED across the six persist runs, every one of them banked its
# LAST scoring unload by step 41,500 of 88,000, leaving 46,500+ steps idle. Steps were
# never the binding constraint, so trading objects for steps buys nothing — and the mud
# copies are the ones worth keeping: they produced 34% of all targeted objects, 36% of all
# scored ones, and the best target->score conversion of any section (73%, against pebble
# 67%, mixed 71%, grass 60%). near/far stay available for A/B; do not ship them.
HARVEST = os.environ.get("MISSION_HARVEST", "all")
if HARVEST in ("near", "far"):
    TERRAIN_RANK["mud"] = 3
if HARVEST == "far":
    TERRAIN_RANK["pebble"] = 3
print(f"[MISSION] harvest={HARVEST} -> TERRAIN_RANK={TERRAIN_RANK}", flush=True)

PROFILE = Profile(
    name="venue",
    max_terrain_rank=2,
    probe_terrains=("pebble", "mixed"),
    # The RED PAPER BAG IS A SCORING TARGET, worth 40 points. Guide line 260 makes it
    # 하위 과제 4 (파란 통 / 재활용); only the PURPLE bag and the bottle are distractors
    # (line 268). Skipping both bags is why every finished run tops out at exactly 120:
    # the four 120/160 reports break down {can:2, rind:2, lunchbox:2} and the bag category
    # is simply never attempted. 120 was a self-imposed ceiling, not a limit of the robot.
    #
    # MISSION_SKIP_REDBAG=1 restores the old behaviour for an A/B against those runs.
    # ...and under OBJECT HONESTY the red bag is skipped whatever SKIP_REDBAG says: the
    # only bag grasp that exists is scripted onto the handle bar's SIMULATOR pose (a
    # 25 mm bar the camera never resolves), so runner.run_grasp refuses it outright with
    # "the bag handle IK needs the simulator pose". Leaving it in the target list only
    # buys a walk, a crouch and a refusal per bag. The 40 points are not honestly
    # reachable until a camera-based handle detector exists — that is a real gap, not a
    # tuning choice, and it is stated here rather than hidden in a wasted approach.
    skip_classes=((("redbag", "purplebag")
                   if (os.environ.get("MISSION_SKIP_REDBAG") == "1"
                       or os.environ.get("MISSION_OBJ_HONEST") == "1")
                   else ("purplebag",))
                  # MISSION_SKIP_LUNCHBOX=1: the frozen-retrained grasp ckpts are
                  # trained on the can/rind family only, and fv2's honest lunchbox
                  # pinches went 0/12 (11 hover-unreachable) — under those ckpts the
                  # lunchbox is pure cycle tax. Cans+rinds alone cap task2 at 80.
                  + (("lunchbox",)
                     if os.environ.get("MISSION_SKIP_LUNCHBOX") == "1" else ())),
    # The red bag is worth 40 points and has never been grasp-tuned, so it is attempted
    # LAST rather than skipped or treated as a peer. MISSION_BAG_RANK=0 turns the
    # deprioritisation off (peer ordering, which is what the two runs above measured).
    class_rank={"redbag": int(os.environ.get("MISSION_BAG_RANK", "1"))},
    max_picks=args_cli.max_picks,
    # MEASURED over the 27 successful deliveries in every report_*.json: cycle 1 caught 22
    # (81%), cycle 2 caught 4 (15%), cycle 3 caught exactly 1 (4%). Meanwhile 8 of the 9
    # give-ups spent all three cycles first. So the third cycle costs ~4,500 steps a run
    # and returns 0.12 objects — and a near-bin object slot is only ~5,000 steps. The
    # venue run that motivated this measurement burned ~24,000 steps (27% of the whole
    # allowance) on two objects it never got, three cycles each.
    object_attempts=2,
    # 1 loss in transit across ~29 deliveries, so one spare is the insurance the record
    # justifies; at 2 the quota allowed up to 4 picks of a class that can only score 2,
    # and each surplus pick is a full object slot.
    spare_picks=1,
    grasp_retries=4,            # nearly free: 68% of attempts bail before the 60-step ramp
    grasp_steps=340,
    carry_in_gripper=(os.environ.get("MISSION_USE_BASKET") != "1"),
    # 3-4 per trip on the v2 (wider, torso-hugging) basket — user, 0826. The 2-carry
    # interlude existed because the SMALL basket shoved 3+ objects into its unreachable
    # rear corner; the wider floor is the intended fix for that, not a lower carry count.
    max_carry=(int(os.environ.get("MISSION_MAX_CARRY", "4"))
               if os.environ.get("MISSION_USE_BASKET") == "1" else 1),
    # With max_carry=1 a trip IS an object, and only 6 objects can ever score
    # (MAX_SCORED_PER_CLASS=2 x 3 non-bag classes). 7 leaves one replacement trip.
    max_trips=7,
    # Re-sweeps have never scored: across six runs, zero of 21 scoring unloads came after
    # a "sweeping again" event, and run 0817_0153 burned 40,137 steps (47% of itself) on
    # two of them. Kept at 2 rather than 1 only because coverage is the binding constraint
    # and total_budget now stops a fruitless pass on its own.
    max_passes=2,
    vx=0.6,
    zigzag_amp=1.1,
    # Deliberately high: this crude cap is no longer the governor. afford_object() now
    # refuses any object the clock cannot finish, and it does so per-object using the
    # distance to the bins — so late in the run a copy parked next to the bins is still
    # worth taking while a far one is not. A flat search_budget cannot express that and
    # would just forfeit the last 20,000 steps. Left in only as a backstop against a
    # robot that is stuck but not detectably stalled.
    search_budget=80000,
    # THE VENUE CLOCK, and now actually enforced (runner.py steps_left/must_head_home).
    # 88,000 rather than 90,000 leaves ~40 s for startup, env reset and localisation,
    # which happen before step 1 and still come out of the 30 minutes.
    # MISSION_TOTAL_BUDGET: short-mission override for module-style debugging — the
    # full stack end to end, just less of it (e.g. 25000 ≈ 8 sim-min reaches the
    # first 2-4 objects). The venue default stays 88000.
    total_budget=int(os.environ.get("MISSION_TOTAL_BUDGET", "88000")),
)

if __name__ == "__main__":
    build_and_run(args_cli, PROFILE)
    simulation_app.close()
