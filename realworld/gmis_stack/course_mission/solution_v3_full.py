"""V3 FULL: every flat terrain incl. slope approach, more picks and retries.

Run:
  source eunha_env.sh s075 && python course_mission/solution_v3_full.py --headless --device cuda:0
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

parser = argparse.ArgumentParser(description="V3 FULL: every flat terrain incl. slope approach, more picks and retries.")
from_launch = None
parser.add_argument("--speed", type=str, default="s075", choices=["s075", "s100"])
parser.add_argument("--depth", type=str, default="0.24")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--per_type", type=int, default=4)
parser.add_argument("--walk_ckpt", type=str, default="")
parser.add_argument("--grasp_ckpt", type=str, default="")
parser.add_argument("--record_head", action="store_true")
parser.add_argument("--max_picks", type=int, default=6)
parser.add_argument("--out_dir", type=str, default=str(RW / "course_mission" / "videos"))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from course_mission.launch import build_and_run  # noqa: E402
from course_mission.runner import Profile  # noqa: E402

PROFILE = Profile(
    name="full",
    max_terrain_rank=3,
    probe_terrains=('pebble', 'mixed', 'slope'),
    max_picks=args_cli.max_picks,
    grasp_steps=380,
    grasp_retries=2,
    vx=0.7,
    zigzag_amp=0.95,
)

if __name__ == "__main__":
    build_and_run(args_cli, PROFILE)
    simulation_app.close()
