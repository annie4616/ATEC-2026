"""Compose A2 (legs, base articulation root) + PiperX (arm) into a single articulated USD.

Mirrors the existing b2_piper.usda pattern: the arm's base body is rigidly fixed to the
quadruped base_link, so the whole thing is ONE articulation with 12 leg DoF + 8 arm DoF.

Approach (programmatic, robust vs hand-editing .usda):
  - New stage, defaultPrim Xform "a2_piperx".
  - Reference the A2 robot USD under /a2_piperx (brings base_link + 12 leg joints, artic root).
  - Reference the PiperX robot USD under /a2_piperx/arm  (brings Link1..8 + 8 arm joints).
  - Offset the arm prim to sit on the base top plate.
  - Add a PhysicsFixedJoint  base_link <-> arm/base_link  at the mount point.
  - Rename arm joints (joint1..8 -> arm_joint1..8) is NOT done here (USD ref can't rename);
    instead the cfg regex will target the arm by prim-path scope. We DO record the names.

Run: env_isaaclab python (Isaac Sim launcher).
"""
import argparse
from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--a2", required=True, help="A2 robot .usd (articulation root)")
ap.add_argument("--piperx", required=True, help="PiperX arm .usd")
ap.add_argument("--out", required=True)
# arm mount pose on the A2 base_link (x fwd, z up). A2 base is smaller than B2; hips at
# x=+/-0.26. Mount the arm forward-top like B2Piper's (0.2,0,0.1), scaled to A2.
ap.add_argument("--mount_x", type=float, default=0.20)
ap.add_argument("--mount_y", type=float, default=0.0)
ap.add_argument("--mount_z", type=float, default=0.10)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(headless=True).app

import os
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf

# Make references RELATIVE to the output layer so the asset dir stays relocatable.
_out_dir = os.path.dirname(os.path.abspath(args.out))
a2_ref = os.path.relpath(os.path.abspath(args.a2), _out_dir)
px_ref = os.path.relpath(os.path.abspath(args.piperx), _out_dir)

ROOT = "/a2_piperx"
stage = Usd.Stage.CreateNew(args.out)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
stage.SetMetadata("metersPerUnit", 1.0)

# defaultPrim wrapper
root = UsdGeom.Xform.Define(stage, ROOT)
stage.SetDefaultPrim(root.GetPrim())

# --- reference A2 (legs + base) directly under root so base_link is /a2_piperx/base_link ---
# The A2 usd's defaultPrim is "a2"; reference it onto ROOT so its children flatten in.
root.GetPrim().GetReferences().AddReference(a2_ref)

# --- reference PiperX under an "arm" xform, offset to the mount point ---
arm_path = ROOT + "/arm"
arm = UsdGeom.Xform.Define(stage, arm_path)
arm.AddTranslateOp().Set(Gf.Vec3d(args.mount_x, args.mount_y, args.mount_z))
arm.GetPrim().GetReferences().AddReference(px_ref)

# The PiperX USD carries its own ArticulationRootAPI on its base_link. If left on, PhysX
# resolves TWO articulations under /a2_piperx and IsaacLab refuses to load. Remove the
# arm's root API (and PhysX articulation API) so the whole thing is ONE articulation rooted
# at the A2 base_link; the FixedJoint below merges the arm's body tree into it.
from pxr import UsdPhysics as _UP, PhysxSchema as _PX
arm_base_prim = stage.GetPrimAtPath(arm_path + "/base_link")
if arm_base_prim and arm_base_prim.IsValid():
    arm_base_prim.RemoveAPI(_UP.ArticulationRootAPI)
    arm_base_prim.RemoveAPI(_PX.PhysxArticulationAPI)

# --- FixedJoint: A2 base_link  <->  arm/base_link (PiperX arm base) ---
base_body = ROOT + "/base_link"
arm_base_body = arm_path + "/base_link"
fj = UsdPhysics.FixedJoint.Define(stage, ROOT + "/arm_mount_fixed_joint")
fj.CreateBody0Rel().SetTargets([Sdf.Path(base_body)])
fj.CreateBody1Rel().SetTargets([Sdf.Path(arm_base_body)])
# localPos0 is on base_link frame = mount point; localPos1 on arm base = origin.
fj.CreateLocalPos0Attr().Set(Gf.Vec3f(args.mount_x, args.mount_y, args.mount_z))
fj.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
fj.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
fj.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))

stage.GetRootLayer().Save()
print("WROTE", args.out, flush=True)
app.close()
