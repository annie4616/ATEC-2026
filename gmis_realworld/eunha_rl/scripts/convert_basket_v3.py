"""Convert a2_piperx_basket_v{3,4}.urdf -> eunha_rl/assets/a2_piperx_basket_v{3,4}.usd (one-time).

Same converter settings as the v1/v2 baskets: merge_fixed_joints False (the *_foot
links are addressed by name by the contact sensors), force/position drives.

Run (GPU 0, ~2 min):
  cd ATEC_gmis
  CUDA_VISIBLE_DEVICES=0 ~/.conda/envs/isaaclab/bin/python \
      eunha_rl/scripts/convert_basket_v3.py --headless [--v 4]
"""

import argparse
import os as _os
from pathlib import Path

_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--v", type=int, default=3)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

RW = Path(__file__).resolve().parents[2]
URDF = RW / "a2pro_rl" / "assets" / f"a2_piperx_basket_v{args_cli.v}.urdf"

cfg = UrdfConverterCfg(
    asset_path=str(URDF),
    usd_dir=str(RW / "eunha_rl" / "assets"),
    usd_file_name=f"a2_piperx_basket_v{args_cli.v}.usd",
    fix_base=False,
    merge_fixed_joints=False,
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        drive_type="force",
        target_type="position",
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=120.0, damping=4.0),
    ),
)
out = UrdfConverter(cfg).usd_path
print(f"[A2] USD written: {out}", flush=True)
simulation_app.close()
