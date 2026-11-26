"""Render Task A env with GROUND-HUGGING red boundary strips, exactly the same env,
into a fixed-length mp4 (default ~30 s).

The red 40 cm road-boundary strips must sit ON the terrain surface (not float / not sink).
The Task A terrain varies in height (flat / random_rough / slopes / stairs), so a single
flat 300 m strip is wrong. Here we sample the actual terrain surface height by ray-casting
the terrain trimesh at each x along the two boundary lines (y = +/- COURSE_HALF_W) and lay
down many short strip segments, each at its local surface z -> the strips hug the ground.

The robot is driven the SAME way as the eval (vision steering head -> wz, or a fixed
straight command) so the recorded env matches what we evaluated.

  CUDA_VISIBLE_DEVICES=1 TMPDIR=/hdd0/sohyun/tmp PYTHONPATH=/hdd0/sohyun \
    python scripts/render_taska_redstrip.py --headless --video \
      --head /hdd0/sohyun/atec_2026_tesk_e/steering_head_recover/steering_head.pt \
      --seconds 30 --out /hdd0/sohyun/atec_2026_tesk_e/taska_redstrip_render
"""
import argparse
import os
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Render Task A with ground-hugging red strips.")
parser.add_argument("--task", type=str, default="ATEC-TaskA-B2Piper")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--head", type=str, default="", help="steering head; empty = straight (wz=0)")
parser.add_argument("--vx", type=float, default=2.0)
parser.add_argument("--wz_max", type=float, default=0.8)
parser.add_argument("--seconds", type=float, default=30.0)
parser.add_argument("--fps", type=int, default=50, help="sim control freq (Task A dt=0.02 -> 50 Hz)")
parser.add_argument("--every", type=int, default=2)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--video", action="store_true", default=False)
# red strip geometry
parser.add_argument("--half_w", type=float, default=4.0, help="course half-width (strip |y|)")
parser.add_argument("--strip_w", type=float, default=0.40, help="red strip width (40 cm)")
parser.add_argument("--seg_len", type=float, default=1.0, help="length of each ground-hugging segment")
parser.add_argument("--x0", type=float, default=-145.0)
parser.add_argument("--x1", type=float, default=150.0)
parser.add_argument("--no_strips", action="store_true", default=False,
                    help="skip drawing red strips entirely (control: is the strip mesh blocking the robot?)")
parser.add_argument("--seed", type=int, default=0, help="seed for reproducibility")
parser.add_argument("--no_illegal_contact", action="store_true", default=False,
                    help="disable the illegal_contact termination (base/hip/thigh touching terrain "
                         "ends the episode). Real-world has no such rule -- the robot just needs to "
                         "keep going without breaking -- so drop it to see if the robot recovers "
                         "from stumbles instead of dying on a hip/thigh graze.")
parser.add_argument("--stop_at_goal", action="store_true", default=False,
                    help="stop the clip at the goal (x>=145) or on any termination, instead of "
                         "resetting and continuing. Use for a clean single-completion video "
                         "(the reset loop follows a fallen robot into the void).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import gymnasium as gym  # noqa: E402
import json  # noqa: E402
import trimesh  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from pxr import UsdGeom, Gf, UsdShade, Sdf  # noqa: E402
import atec_rl_lab.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from atec_rl_lab.tasks.task_base.action_base import apply_safe_action_spec  # noqa: E402
from atec_2026_tesk_e.ATEC2026_Simulation_Challenge.demo.solution_rl import AlgSolution  # noqa: E402

solution = AlgSolution()


class SteeringHead(nn.Module):
    def __init__(self, rgb_dim, lidar_dim, hidden=256):
        super().__init__()
        self.rgb_mlp = nn.Sequential(nn.Linear(rgb_dim, hidden), nn.GELU(), nn.Dropout(0.2),
                                     nn.Linear(hidden, hidden), nn.GELU())
        self.lidar_mlp = nn.Sequential(nn.Linear(lidar_dim, hidden), nn.GELU(), nn.Dropout(0.2),
                                       nn.Linear(hidden, hidden), nn.GELU())
        self.fuse = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear(hidden, 1))

    def forward(self, rgb_feat, lidar):
        return self.fuse(torch.cat([self.rgb_mlp(rgb_feat), self.lidar_mlp(lidar)], -1)).squeeze(-1)


def sample_surface_z(mesh: trimesh.Trimesh, xs, ys):
    """Ray-cast straight down at (x,y, z=+50) to find terrain surface z. Returns z array."""
    origins = np.stack([xs, ys, np.full_like(xs, 50.0)], axis=1)
    dirs = np.tile(np.array([0.0, 0.0, -1.0]), (len(xs), 1))
    locs, ray_idx, tri_idx = mesh.ray.intersects_location(origins, dirs, multiple_hits=False)
    z = np.zeros(len(xs), dtype=np.float32)
    hit = np.zeros(len(xs), dtype=bool)
    for loc, ri in zip(locs, ray_idx):
        z[ri] = loc[2]
        hit[ri] = True
    return z, hit


def draw_red_strips(stage, mesh, args):
    """Draw a CONTINUOUS red ribbon along y=+/-half_w that hugs (and tilts with) the
    terrain surface. Instead of flat cuboids that step at height changes, each segment
    is PITCHED to connect its two endpoints' surface heights, so the ribbon is unbroken
    and slopes with ramps/stairs. Built as one triangle mesh (a thin quad strip) per side."""
    # dense sampling of the surface height along the line
    xs = np.arange(args.x0, args.x1 + args.seg_len, args.seg_len, dtype=np.float32)
    for side, y in (("L", +args.half_w), ("R", -args.half_w)):
        ys = np.full_like(xs, y)
        z, hit = sample_surface_z(mesh, xs, ys)
        if (~hit).any() and hit.any():
            z[~hit] = np.interp(np.flatnonzero(~hit), np.flatnonzero(hit), z[hit])
        z = z + 0.012  # sit just above the surface
        hw = args.strip_w * 0.5
        # build a quad strip: for each x sample two verts at y-hw and y+hw
        verts, faces = [], []
        for i, (x, zz) in enumerate(zip(xs, z)):
            verts.append((float(x), float(y) - hw, float(zz)))
            verts.append((float(x), float(y) + hw, float(zz)))
        for i in range(len(xs) - 1):
            a, b, c, d = 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3
            faces.append((a, b, d))   # two triangles per quad; z tilts with neighbors
            faces.append((a, d, c))
        path = f"/World/red_strip_{side}"
        m = UsdGeom.Mesh.Define(stage, path)
        m.CreatePointsAttr([Gf.Vec3f(*v) for v in verts])
        idx = [vi for f in faces for vi in f]
        m.CreateFaceVertexIndicesAttr(idx)
        m.CreateFaceVertexCountsAttr([3] * len(faces))
        m.CreateDisplayColorAttr([Gf.Vec3f(0.9, 0.05, 0.05)])
        m.CreateDoubleSidedAttr(True)
    print(f"[strips] drew CONTINUOUS tilted ribbon/side ({len(xs)} samples) hugging terrain "
          f"(x {args.x0}..{args.x1}, y=+/-{args.half_w})", flush=True)


def main():
    os.makedirs(args_cli.out, exist_ok=True)
    dev = "cuda"

    head = None
    if args_cli.head:
        ckpt = torch.load(args_cli.head, map_location=dev, weights_only=False)
        head = SteeringHead(ckpt["rgb_dim"], ckpt["lidar_dim"]).to(dev).eval()
        head.load_state_dict(ckpt["model"])
        rf_mu = torch.tensor(ckpt["rf_mu"], dtype=torch.float32, device=dev)
        rf_sd = torch.tensor(ckpt["rf_sd"], dtype=torch.float32, device=dev)
        li_mu = torch.tensor(ckpt["li_mu"], dtype=torch.float32, device=dev)
        li_sd = torch.tensor(ckpt["li_sd"], dtype=torch.float32, device=dev)
        from transformers import AutoModel, AutoImageProcessor
        proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        dino = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval()

        def rgb_to_feat(rgb_np):
            inp = proc(images=[rgb_np], return_tensors="pt").to(dev)
            with torch.inference_mode():
                return dino(**inp).last_hidden_state[:, 0]

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device,
                            num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    action_spec = solution.get_action_spec() if hasattr(solution, "get_action_spec") else None
    env_cfg = apply_safe_action_spec(env_cfg, json.dumps(action_spec))
    env_cfg.sim.physx.enable_enhanced_determinism = True   # reproducible physics for the video
    if args_cli.no_illegal_contact:
        # real-world has no "hip/thigh touched the ground -> instant game over" rule; the robot
        # only needs to stay intact and reach the goal. Dropping this lets it stumble & recover.
        env_cfg.terminations.illegal_contact = None
        print("[cfg] illegal_contact termination DISABLED", flush=True)

    # follow the robot so it's actually visible in the recorded video (default viewer
    # stares at world origin x=0 while the robot spawns at x=-141 -> off-screen).
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.eye = (-6.0, -5.0, 4.0)     # behind-left & above, relative to robot
    env_cfg.viewer.lookat = (2.0, 0.0, 0.0)    # look ahead of the robot

    n_steps = int(args_cli.seconds * args_cli.fps)
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # --- draw ground-hugging red strips onto the terrain mesh ---
    importer = env.unwrapped.scene.terrain
    tgen = getattr(importer, "terrain_generator", None)
    mesh = tgen.terrain_mesh if tgen is not None else None
    import isaacsim.core.utils.stage as stage_utils
    stage = stage_utils.get_current_stage()
    # hide the flat fixed-z strips from env_cfg (they float/sink over non-flat terrain)
    for flat_path in ("/World/red_strip_left", "/World/red_strip_right"):
        prim = stage.GetPrimAtPath(flat_path)
        if prim and prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()
            print(f"[strips] hid flat strip {flat_path}", flush=True)
    if args_cli.no_strips:
        print("[strips] --no_strips: skipping red strips (control run)", flush=True)
    elif mesh is not None:
        draw_red_strips(stage, mesh, args_cli)
    else:
        print("[WARN] no terrain mesh; skipping red strips", flush=True)

    if args_cli.video:
        video_dir = os.path.join(args_cli.out, "video")
        env = gym.wrappers.RecordVideo(
            env, video_folder=video_dir, name_prefix="taska_redstrip",
            step_trigger=lambda s: s == 0, video_length=n_steps, disable_logger=True)
        print(f"[video] recording {args_cli.seconds:.0f}s ({n_steps} steps) -> {video_dir}", flush=True)

    import random as _random
    _random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    torch.cuda.manual_seed_all(args_cli.seed)
    obs, _ = env.reset(seed=args_cli.seed)
    robot = env.unwrapped.scene["robot"]
    head_cam = env.unwrapped.scene["head_camera"]
    try:
        lidar = env.unwrapped.scene["lidar_sensor"]
    except (KeyError, Exception):
        lidar = None

    wz = 0.0
    for step in range(n_steps):
        with torch.inference_mode():
            if head is not None and lidar is not None and step % args_cli.every == 0:
                rgb = head_cam.data.output["rgb"][0].cpu().numpy()[:, :, :3].astype(np.uint8)
                hits = lidar.data.ray_hits_w[0]
                base = robot.data.root_pos_w[0]
                rng = torch.norm(hits - base.unsqueeze(0), dim=1)
                rng = torch.where(torch.isfinite(rng), rng, torch.full_like(rng, 10.0))
                rng = torch.clip(rng, 0.0, 10.0).view(1, -1)
                rfeat = (rgb_to_feat(rgb) - rf_mu) / rf_sd
                lfeat = (rng - li_mu) / li_sd
                wz = float(head(rfeat, lfeat)[0])
                wz = max(-args_cli.wz_max, min(args_cli.wz_max, wz))

            solution.fixed_velocity_commands = torch.tensor(
                [args_cli.vx, 0.0, wz], device=solution.device, dtype=torch.float32).view(1, 3)
            resp = solution.predicts(obs, 0.0)
            act = (torch.zeros((1, 20), device=dev) if (resp.get("giveup") or not resp["action"])
                   else torch.tensor(resp["action"], dtype=torch.float32, device=dev).view(1, -1))
            obs, _, term, trunc, _ = env.step(act)
            p = robot.data.root_pos_w[0]
            xnow = float(p[0])
            if step % 100 == 0:
                print(f"[step {step:4d}] x={xnow:+.1f} y={float(p[1]):+.2f} wz={wz:+.2f}", flush=True)
            # clean completion clip: stop at the goal (or on any termination) instead of
            # resetting and continuing -- the reset-and-continue loop follows a fallen robot
            # off the course edge into the void, which corrupts the clip.
            if args_cli.stop_at_goal and xnow >= 145.0:
                print(f"[GOAL] reached x={xnow:+.1f} at step {step} -- stopping clip.", flush=True)
                break
            if bool(term[0]):
                print(f"[TERM] step {step} x={xnow:+.1f} y={float(p[1]):+.2f}", flush=True)
                if args_cli.stop_at_goal:
                    break                      # stop the clip on termination (no void reset)
                obs, _ = env.reset()           # legacy: keep a fixed-length clip

    print(f"[DONE] rendered {args_cli.seconds:.0f}s", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
