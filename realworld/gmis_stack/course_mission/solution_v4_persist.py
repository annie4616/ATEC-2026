"""Solution v4 — go for 120/160 by refusing to give up on the objects we CAN grasp.

The guide scores 4 categories x 2 objects x 20 points. Paper bags are 0/6 in every probe:
lying on their side they present a 130 x 270 mm face and the gripper opens 100 mm, so
there is no axis to close on and each attempt still costs an approach, a squat and 320
policy steps. Leaving them out caps this run at 2 cans + 2 rinds + 2 lunchboxes = 120,
and spends the whole budget on objects that do come up:

    can       2/2 in the regression probe
    rind      1/2
    lunchbox  3/4 across two probes  (0/3 before the grasp-geometry work)

So: skip the bags, allow three full approach->squat->grasp cycles per object instead of
one, four grasp retries inside each cycle, and a budget big enough for the shuttling that
six picks needs (basket holds two, so three round trips minimum).

Run:  scripts/run_mission_persist.sh   (add MISSION_ORACLE_DETECT=1 to isolate perception)
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

parser = argparse.ArgumentParser(description="Course mission — persistent, bags skipped.")
parser.add_argument("--speed", type=str, default="s075")
parser.add_argument("--depth", type=str, default="0.24")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--per_type", type=int, default=4)
parser.add_argument("--max_picks", type=int, default=8)
parser.add_argument("--walk_ckpt", type=str, default="")
parser.add_argument("--grasp_ckpt", type=str, default="")
parser.add_argument("--record_head", action="store_true")
parser.add_argument("--out_dir", type=str, default=str(RW / "course_mission" / "videos_persist"))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os  # noqa: E402

# No basket on the robot at all. This variant carries in the gripper, so the basket is
# 0.3 kg of dead weight and an extra collision body hanging off the back — and the walk
# policy (from_eunha) was trained WITHOUT it, so removing it moves the robot back toward
# the distribution the policy knows.
os.environ["MISSION_NO_BASKET"] = "1"

from course_mission.launch import build_and_run  # noqa: E402
from course_mission.runner import Profile  # noqa: E402

PROFILE = Profile(
    name="persist",
    max_terrain_rank=2,
    probe_terrains=("pebble", "mixed"),
    skip_classes=("redbag", "purplebag"),
    max_picks=args_cli.max_picks,
    # MEASURED across four oracle runs: 3 cycles scored 100 twice, 5 cycles scored 80.
    # Spending more tries on a stubborn object costs the walking budget that would have
    # found an easier one, and the guide only pays for 2 per category either way.
    object_attempts=3,          # full approach -> squat -> grasp cycles per object
    spare_picks=2,              # replace an object lost between basket and bin
    grasp_retries=4,            # policy restarts from a fresh pregrasp inside one squat
    grasp_steps=340,
    carry_in_gripper=True,      # basket bypassed: hold it and walk
    max_carry=1,
    max_trips=10,
    max_passes=3,               # the search only walks forward; sweep the course again                # one object per trip when carrying by hand
    vx=0.6,
    # Objects are scattered to +-1.15 m laterally (env_cfg.scatter_layout). At 0.85 the
    # sweep never brought the outermost ones into the camera cone: a run swept the whole
    # course three times and targeted exactly one of the four cans — the other three were
    # never even considered, which is what left it on 100 instead of 120.
    zigzag_amp=1.1,
    # Persistence costs steps: three cycles on a stubborn object is ~4000, and each
    # shuttle round trip is ~60 m of walking. Budget for it rather than timing out at
    # the point where the last two objects would have scored.
    search_budget=160000,
    total_budget=260000,
)

if __name__ == "__main__":
    build_and_run(args_cli, PROFILE)
    simulation_app.close()
