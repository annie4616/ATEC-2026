"""Build photo-textured USD assets for the competition props (can, bottle, redbag,
purplebag) from the venue photographs in course_sim/image/.

Why meshes and not the CuboidCfg/CylinderCfg primitives these replace: a USD primitive
carries no `st` primvar, so there is nothing for a texture to sample, and IsaacLab's
PreviewSurfaceCfg only exposes a flat diffuse colour. Textures therefore require authored
meshes with UVs.

Geometry is kept IDENTICAL to catalogue.OBJ_DIMS so grasping is unaffected:
  - the cylinders use 32 segments, whose convex hull is within 0.25 % of the analytic
    radius, and are collided as convexHull (exact for a convex body)
  - the boxes are the same extents as the cuboids, collided as convexHull (also exact)
Mass is set here to the same value env_cfg.py passed inline. Friction is NOT set here:
UsdFileCfg takes no physics_material, so env_cfg.py's per-object startup event must cover
these classes exactly as it already does for the lunchbox and rind.

Run:  ~/.conda/envs/isaaclab/bin/python a2pro_rl/scripts/build_prop_usds.py --headless
"""

import argparse
import math
from pathlib import Path

# Shared machine: GPUs 1-7 are other people's training runs. Pin to GPU 0 BEFORE any
# CUDA-aware import so forgetting the CUDA_VISIBLE_DEVICES prefix cannot land this on
# someone else's device.
import os as _os
_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--only", default="", help="comma-separated prop names, default all")
AppLauncher.add_app_launcher_args(parser)
_args = parser.parse_args()
app = AppLauncher(_args).app

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
TEX = ASSETS / "textures"


# --------------------------------------------------------------------------- materials
def photo_material(stage, path, tex_png, roughness=0.6, metallic=0.0):
    """UsdPreviewSurface whose diffuse comes from a PNG sampled through the `st` primvar.

    Same shader graph as course_sim/terrain._apply_photo_texture, which is known to render
    in this Isaac build; wrap is `clamp` here because a prop face shows the photo once
    rather than tiling it, and `repeat` would mirror seams onto the silhouette edge.
    """
    mat = UsdShade.Material.Define(stage, path)
    reader = UsdShade.Shader.Define(stage, path + "/stReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex = UsdShade.Shader.Define(stage, path + "/Tex")
    tex.CreateIdAttr("UsdUVTexture")
    # RELATIVE asset path, resolved by USD against this layer's own directory. An absolute
    # path works only on the machine that authored it: handing the .usd to anyone else --
    # or moving the repo -- silently drops the texture and the prop renders untextured,
    # with no error anywhere. The textures sit in assets/textures/ beside assets/*.usd.
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"./textures/{tex_png}")
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.ConnectableAPI(), "result")
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        tex.ConnectableAPI(), "rgb")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def put_mesh(stage, path, pts, idx, cnt, st, mat, collide=True):
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
    m.CreateFaceVertexIndicesAttr(idx)
    m.CreateFaceVertexCountsAttr(cnt)
    UsdGeom.PrimvarsAPI(m.GetPrim()).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.faceVarying).Set([Gf.Vec2f(*t) for t in st])
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    if collide:
        UsdPhysics.CollisionAPI.Apply(m.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr("convexHull")
    return m


# --------------------------------------------------------------------------- geometry
def cylinder_side(r, h, segs=32):
    """Quad strip around the axis, origin at the CENTRE of the body (matches CylinderCfg).

    u runs 0..1 once around, v runs 0..1 bottom to top, so the label photo wraps exactly
    once and reads upright from every angle.
    """
    pts, idx, cnt, st = [], [], [], []
    for i in range(segs):
        a0, a1 = 2 * math.pi * i / segs, 2 * math.pi * (i + 1) / segs
        b = len(pts)
        pts += [(r * math.cos(a0), r * math.sin(a0), -h / 2),
                (r * math.cos(a1), r * math.sin(a1), -h / 2),
                (r * math.cos(a1), r * math.sin(a1), +h / 2),
                (r * math.cos(a0), r * math.sin(a0), +h / 2)]
        idx += [b, b + 1, b + 2, b + 3]
        cnt.append(4)
        u0, u1 = i / segs, (i + 1) / segs
        st += [(u0, 0.0), (u1, 0.0), (u1, 1.0), (u0, 1.0)]
    return pts, idx, cnt, st


def cylinder_cap(r, z, up, segs=32):
    """Triangle fan, UV = the unit disc of the cap photo.

    st is faceVarying like every other primvar here -- one entry per FACE VERTEX, not per
    point. Emitting one per point built a 33-entry array against 32 triangles x 3 = 96
    expected, and Isaac rejected it at load: "corrupted data in primvar 'st': buffer size
    33 doesn't match expected size 96". It only showed up in a running mission's log,
    because build-time verification counted the texture bindings and never the UVs.
    """
    pts = [(0.0, 0.0, z)]
    disc = []
    for i in range(segs):
        a = 2 * math.pi * i / segs
        pts.append((r * math.cos(a), r * math.sin(a), z))
        disc.append((0.5 + 0.5 * math.cos(a), 0.5 + 0.5 * math.sin(a)))
    idx, cnt, st = [], [], []
    for i in range(segs):
        j, k = 1 + i, 1 + (i + 1) % segs
        idx += [0, j, k] if up else [0, k, j]
        cnt.append(3)
        st += [(0.5, 0.5), disc[i], disc[(i + 1) % segs]] if up else \
              [(0.5, 0.5), disc[(i + 1) % segs], disc[i]]
    return pts, idx, cnt, st


BOX_FACES = {                       # name -> (4 corner signs, outward winding)
    "front": [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)],
    "back":  [(1, 1, -1), (-1, 1, -1), (-1, 1, 1), (1, 1, 1)],
    "left":  [(-1, 1, -1), (-1, -1, -1), (-1, -1, 1), (-1, 1, 1)],
    "right": [(1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)],
    "top":   [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)],
    "bottom": [(-1, 1, -1), (1, 1, -1), (1, -1, -1), (-1, -1, -1)],
}


def box_face(size, face):
    sx, sy, sz = (s / 2 for s in size)
    pts = [(cx * sx, cy * sy, cz * sz) for cx, cy, cz in BOX_FACES[face]]
    return pts, [0, 1, 2, 3], [4], [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


# --------------------------------------------------------------------------- props
def build_cylinder(name, prim, radius, height, mass, side_png, cap_png):
    out = ASSETS / f"{name}.usd"
    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{prim}")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(mass)
    side_mat = photo_material(stage, f"/{prim}/Looks/side", side_png)
    cap_mat = photo_material(stage, f"/{prim}/Looks/cap", cap_png)
    put_mesh(stage, f"/{prim}/side", *cylinder_side(radius, height), side_mat)
    put_mesh(stage, f"/{prim}/top", *cylinder_cap(radius, height / 2, True), cap_mat)
    put_mesh(stage, f"/{prim}/bottom", *cylinder_cap(radius, -height / 2, False), cap_mat)
    stage.GetRootLayer().Save()
    return out


def build_bag(name, prim, size, mass, face_png, side_png, top_png,
              wall=0.003, floor=0.004, handle=False):
    """An OPEN-TOP, THIN-WALLED paper bag: a floor plus four walls, no lid.

    The previous model was six flat faces sealed into a box, which gave the gripper nothing
    to close on but a 55 mm void between two zero-thickness sheets. Measured across two
    Hong Kong runs, every single red-bag grasp failed -- 6 of 8 attempts in one, 6 of 6 in
    the other, zero conversions -- while cans went 2 for 2.

    A real paper bag is picked up the way a hand does it: hook into the mouth and pinch ONE
    LAYER of the wall. That layer has to exist for the gripper to find it, so the walls are
    given a real 3 mm thickness (paper is ~0.3 mm, but a contact pair that thin is below
    what the solver resolves reliably at this contact offset) and the top is left open.

    Each wall is its OWN mesh with its own convexHull. That matters: one hull over the whole
    bag would seal the mouth shut and put us right back where we started.
    """
    out = ASSETS / f"{name}.usd"
    if out.exists():
        out.unlink()
    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{prim}")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    _mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
    _mass_api.CreateMassAttr(mass)
    L, W, H = size
    # A real paper bag is bottom-heavy (contents settle); the uniform-mass model
    # toppled at the slightest handle contact — 20+ instrumented handle attempts
    # lost to tip-overs. CoM in the bottom third is the honest prior.
    _mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, -H / 3))
    mats = {"face": photo_material(stage, f"/{prim}/Looks/face", face_png),
            "side": photo_material(stage, f"/{prim}/Looks/side", side_png),
            "in": photo_material(stage, f"/{prim}/Looks/inner", top_png)}
    hz = H / 2
    # floor
    put_mesh(stage, f"/{prim}/floor",
             *box_mesh((L, W, floor), (0.0, 0.0, -hz + floor / 2)), mats["in"])
    wh = H - floor                      # wall height above the floor
    wz = -hz + floor + wh / 2
    put_mesh(stage, f"/{prim}/front",
             *box_mesh((L, wall, wh), (0.0, -(W - wall) / 2, wz)), mats["face"])
    put_mesh(stage, f"/{prim}/back",
             *box_mesh((L, wall, wh), (0.0, (W - wall) / 2, wz)), mats["face"])
    put_mesh(stage, f"/{prim}/left",
             *box_mesh((wall, W - 2 * wall, wh), (-(L - wall) / 2, 0.0, wz)), mats["side"])
    put_mesh(stage, f"/{prim}/right",
             *box_mesh((wall, W - 2 * wall, wh), ((L - wall) / 2, 0.0, wz)), mats["side"])
    if handle:
        # A carry HANDLE (user 0827): the mouth pinch never converted a bag (0 HELD
        # across every campaign) because a 3 mm sheet gives the fingers ~2 N of clamp.
        # A real paper bag is carried by its handle, so give it one: two uprights and a
        # crossbar, 8 mm strap section against the 100 mm finger span, 60 mm of air
        # beneath the bar for the finger tips. Each piece is its own convexHull.
        hs, up_h, strap = 0.045, 0.060, 0.008
        put_mesh(stage, f"/{prim}/handle_up1",
                 *box_mesh((strap, strap, up_h), (-hs, 0.0, hz + up_h / 2)), mats["side"])
        put_mesh(stage, f"/{prim}/handle_up2",
                 *box_mesh((strap, strap, up_h), (hs, 0.0, hz + up_h / 2)), mats["side"])
        # The BAR is 20 mm tall (a folded paper handle, not a wire): the pinch's
        # close-plane scatter measured +-10 mm, the same size as an 8 mm strap's
        # half-height — chronic top-grazes and under-passes. 20 mm of catchable
        # height turns that scatter into margin. Closing width stays 8 mm.
        bar_h = 0.020
        put_mesh(stage, f"/{prim}/handle_bar",
                 *box_mesh((2 * hs + strap, strap, bar_h),
                           (0.0, 0.0, hz + up_h + bar_h / 2)), mats["side"])
    stage.GetRootLayer().Save()
    return out


def box_mesh(size, centre):
    """Axis-aligned box as points/indices/counts/st, one 0..1 UV square per face."""
    sx, sy, sz = (v / 2 for v in size)
    pts, idx, cnt, st = [], [], [], []
    for corners in (BOX_FACES[k] for k in ("front", "back", "left", "right",
                                           "top", "bottom")):
        b = len(pts)
        for cx, cy, cz in corners:
            pts.append((centre[0] + cx * sx, centre[1] + cy * sy, centre[2] + cz * sz))
        idx += [b, b + 1, b + 2, b + 3]
        cnt.append(4)
        st += [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    return pts, idx, cnt, st


def build_box(name, prim, size, mass, face_png, side_png, top_png):
    out = ASSETS / f"{name}.usd"
    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{prim}")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(mass)
    mats = {"face": photo_material(stage, f"/{prim}/Looks/face", face_png),
            "side": photo_material(stage, f"/{prim}/Looks/side", side_png),
            "top": photo_material(stage, f"/{prim}/Looks/top", top_png)}
    which = {"front": "face", "back": "face", "left": "side", "right": "side",
             "top": "top", "bottom": "top"}
    for f, key in which.items():
        put_mesh(stage, f"/{prim}/{f}", *box_face(size, f), mats[key])
    stage.GetRootLayer().Save()
    return out


def verify(path, expect_dims):
    """Re-open what landed on disk and check extent + origin.

    The lunchbox asset taught this: its origin sat on the bottom face for weeks and every
    grasp aimed 35 mm low because nobody re-opened the file and looked.
    """
    st = Usd.Stage.Open(str(path))
    lo = [1e9] * 3
    hi = [-1e9] * 3
    ntex = 0
    bad_uv = []
    for p in st.Traverse():
        if p.GetTypeName() == "Mesh":
            m = UsdGeom.Mesh(p)
            for v in m.GetPointsAttr().Get():
                for k in range(3):
                    lo[k] = min(lo[k], v[k]); hi[k] = max(hi[k], v[k])
            # faceVarying st must have exactly one entry per face vertex. Checking only
            # that a texture is bound let a 33-vs-96 mismatch ship and surface as a
            # renderer warning mid-mission.
            pv = UsdGeom.PrimvarsAPI(p).GetPrimvar("st")
            nfv = sum(m.GetFaceVertexCountsAttr().Get() or [])
            nst = len(pv.Get() or []) if pv else 0
            if nst != nfv:
                bad_uv.append(f"{p.GetName()} st={nst} faceverts={nfv}")
        if p.GetTypeName() == "Shader" and p.GetAttribute("info:id").Get() == "UsdUVTexture":
            f = p.GetAttribute("inputs:file").Get()
            # resolve relative to the layer, and REFUSE an absolute path: that is the bug
            # this check exists for, not merely a missing file
            raw = str(f.path) if f else ""
            assert raw and not raw.startswith("/"), f"texture path must be relative: {raw!r}"
            assert (path.parent / raw).resolve().exists(), f"missing texture {raw}"
            ntex += 1
    dims = tuple(round(hi[k] - lo[k], 4) for k in range(3))
    ctr = tuple(round((hi[k] + lo[k]) / 2, 4) for k in range(3))
    ok = all(abs(dims[k] - expect_dims[k]) < 2e-3 for k in range(3)) and \
        all(abs(c) < 1e-3 for c in ctr) and not bad_uv
    print(f"  {path.name:<14s} dims {dims}  centre {ctr}  textures {ntex}  "
          f"{'OK' if ok else '*** BAD: ' + ('; '.join(bad_uv) if bad_uv else 'expected ' + str(expect_dims))}")
    return ok


if __name__ == "__main__":
    jobs = {
        "can": lambda: (build_cylinder("can", "Can", 0.033, 0.115, 0.20,
                                       "can_side.png", "can_cap.png"),
                        (0.066, 0.066, 0.115)),
        "bottle": lambda: (build_cylinder("bottle", "Bottle", 0.030, 0.170, 0.20,
                                          "bottle_side.png", "bottle_cap.png"),
                           (0.060, 0.060, 0.170)),
        "redbag": lambda: (build_bag("redbag", "Redbag", (0.130, 0.055, 0.270), 0.10,
                                     "redbag_face.png", "redbag_side.png",
                                     "redbag_top.png"),
                           (0.130, 0.055, 0.270)),
        "redbag_handle": lambda: (build_bag("redbag_handle", "Redbag",
                                            (0.130, 0.055, 0.270), 0.10,
                                            "redbag_face.png", "redbag_side.png",
                                            "redbag_top.png", handle=True),
                                  (0.130, 0.055, 0.338)),
        "purplebag": lambda: (build_bag("purplebag", "Purplebag", (0.130, 0.055, 0.270),
                                        0.10, "purplebag_face.png", "purplebag_face.png",
                                        "purplebag_top.png"),
                              (0.130, 0.055, 0.270)),
    }
    names = [n for n in (_args.only.split(",") if _args.only else jobs) if n in jobs]
    print("[PROP] a2pro_rl/assets/*.usd")
    allok = True
    for n in names:
        p = ASSETS / f"{n}.usd"
        if p.exists():
            p.unlink()                       # Usd.Stage.CreateNew refuses to clobber
        out, dims = jobs[n]()
        allok &= verify(out, dims)
    print("[PROP] all verified" if allok else "[PROP] *** VERIFY FAILED", flush=True)
    app.close()
