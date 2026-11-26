"""Inspect a converted robot USD: list bodies + joints. Writes result to <usd>.inspect.txt."""
import argparse
from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--usd", required=True)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(headless=True).app

from pxr import Usd, UsdPhysics

stage = Usd.Stage.Open(args.usd)
lines = []
lines.append("USD: " + args.usd)
dp = stage.GetDefaultPrim()
lines.append("default prim: " + (dp.GetName() if dp else "(none)"))
rev, pris, fixed, links = [], [], [], []
for p in stage.Traverse():
    t = p.GetTypeName()
    if t == "PhysicsRevoluteJoint":
        rev.append(p.GetName())
    elif t == "PhysicsPrismaticJoint":
        pris.append(p.GetName())
    elif t == "PhysicsFixedJoint":
        fixed.append(p.GetName())
    if p.HasAPI(UsdPhysics.RigidBodyAPI):
        links.append(p.GetName())
lines.append(f"rigid bodies ({len(links)}): {links}")
lines.append(f"revolute joints ({len(rev)}): {rev}")
lines.append(f"prismatic joints ({len(pris)}): {pris}")
lines.append(f"fixed joints ({len(fixed)}): {fixed}")
out = args.usd + ".inspect.txt"
with open(out, "w") as f:
    f.write("\n".join(lines) + "\n")
print("WROTE", out, flush=True)
app.close()
