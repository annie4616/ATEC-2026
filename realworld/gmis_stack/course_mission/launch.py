"""Scene/camera/checkpoint wiring shared by the solution_v*.py variants.

Imported only AFTER AppLauncher has started the simulator.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import gymnasium as gym
import imageio.v2 as iio
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab_tasks.utils import parse_env_cfg

import course_mission  # noqa: F401  (registers ATEC-Course-Mission)
from course_mission.env_cfg import scatter_layout
from course_mission.runner import MissionRunner, Profile

RW = Path(__file__).resolve().parents[1]
TASK = "ATEC-Course-Mission"

# head camera: A2's catalogued mount, pitched 22 deg down so the 0.5-3 m ground band
# (where objects are detected and grasped) fills the frame
HEAD_POS = (0.3381, 0.0336, 0.0525)
_TH = np.deg2rad(22.0)
HEAD_ROT = (float(np.cos(_TH / 2)), 0.0, float(np.sin(_TH / 2)), 0.0)



def head_cam_cfg(update_period: float = 0.0) -> CameraCfg:
    """The head camera, as a STANDALONE prim posed every step by HeadCamMount.

    It used to be parented at Robot/base_link/head_cam with an OffsetCfg. That silently
    does not work: with GPU physics the articulation's USD transforms are never written
    (the state lives in Fabric), so a camera prim underneath a link keeps its spawn
    transform forever. Measured over a 250-frame walk, the camera's offset in the base
    frame drifted +0.35 m -> -1.98 m -> -39 m: it filmed the start of the course while the
    robot walked away, which is why the detector only ever found terrain and why the
    mission needed MISSION_ORACLE_DETECT=1. Setting the world pose on the child prim does
    not help either — XFormPrimView derives the local transform from the same stale parent.

    As a root-level prim it has no parent to go stale, exactly like the overview/follow
    cameras that always tracked correctly, and HeadCamMount writes its world pose from the
    robot base state each control step.
    """
    return CameraCfg(
        prim_path="/World/head_cam",
        update_period=update_period, height=360, width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=16.0, clipping_range=(0.05, 30.0)),
    )

def resolve_ckpts(args) -> dict:
    sp = args.speed
    src = "speed_1.0" if sp == "s100" else "speed_0.75"
    E = RW / "eunha_rl"
    depth = args.depth
    _ck_env = os.environ.get("MISSION_CROUCH_CKPT")
    crouch = ([Path(_ck_env)] if _ck_env else
              sorted((E / sp / "crouch_runs").glob(f"crouch{depth}_*/model_2999.pt")))
    if not crouch:
        raise SystemExit(f"no crouch policy for depth {depth} under {E / sp / 'crouch_runs'}")
    champ = E / "detached" / f"champ_eunha3_{sp}_{depth}.txt"
    grasp = os.environ.get("MISSION_GRASP_CKPT") or args.grasp_ckpt \
        or champ.read_text().strip()
    if grasp and not os.path.isabs(str(grasp)):
        grasp = str(RW / grasp)          # champ files in this repo are RW-relative
    walk = args.walk_ckpt or str(RW / "from_eunha" / src / "model_14999.pt")
    adapt = os.environ.get("EUNHA_ADAPT_CKPT", str(RW / "from_eunha" / src / "adaptation.pt"))
    # LINEAGE WARNING, not a fix. eunha_env.sh exports EUNHA_ADAPT_CKPT *and*
    # EUNHA_WALK_CKPT, but nothing here has ever read the second one, so sourcing the
    # s0821 preset moves the adaptation net to the 0821 drop while --speed keeps the walk
    # actor on s075. Every venue run since 2026-08-22 has used that mixed pair -- and it
    # is also the pair that scored 120/160 against pure s075's 100, so it must NOT be
    # "corrected" without an A/B. Say it out loud instead, and record it in config.json.
    if Path(walk).parent != Path(adapt).parent:
        print(f"[LAUNCH] NOTE mixed locomotion lineage: walk={Path(walk).parent.name} "
              f"adapt={Path(adapt).parent.name} act={os.environ.get('EUNHA_ADAPT_ACT', 'elu')}"
              f" (EUNHA_WALK_CKPT={os.environ.get('EUNHA_WALK_CKPT', '-')} is not read)",
              flush=True)
    return {
        "walk": walk,
        "adapt": adapt,
        "crouch": str(crouch[-1]),
        "grasp": grasp,
        "bank": str(E / sp / "banks" / f"pregrasp_bank_{depth}.npz"),
    }


# Every environment variable anywhere in course_mission/ or course_sim/ that changes what
# the run does. Recording nine of them (snapshot_src.sh) is how a 120-scoring run ended up
# with a PROVENANCE file that could not say which colour prototypes it used.
CONFIG_VARS = (
    "COURSE_STRIP_FLAT", "COURSE_TERRAIN_TEX", "COURSE_VENUE",
    "EUNHA_ADAPT_ACT", "EUNHA_ADAPT_CKPT", "EUNHA_WALK_CKPT",
    "GRASP_GRIP_DEPTH", "GRASP_GRIP_OPEN", "GRASP_TCP_OFFSET",
    "MISSION_ACTION_DELAY", "MISSION_APPROACH_CLEAR", "MISSION_APPROACH_STEPS",
    "MISSION_ARM_DELAY", "MISSION_ARM_RATE", "MISSION_ARM_VEL_LIMIT",
    "MISSION_BAG_FRICTION", "MISSION_BAG_GRIP", "MISSION_BAG_IK", "MISSION_BAG_OBB",
    "MISSION_BAG_ORBIT", "MISSION_BAG_RANK", "MISSION_BAG_REPOSITION",
    "MISSION_BANK_SETTLE", "MISSION_CARRY", "MISSION_COMPUTE", "MISSION_DETECT_HZ",
    "MISSION_GRASP_DET_SEED", "MISSION_GRASP_FREEZE", "MISSION_GRASP_HOLD",
    "MISSION_GRASP_POS_CAP",
    "MISSION_GRASP_ROT_CAP", "MISSION_GRASP_STEPS", "MISSION_GRIP_RATE", "MISSION_HARVEST",
    "MISSION_APRON_FALLBACK", "MISSION_FOOT_GUARD", "MISSION_IK_FRAME_FIX",
    "MISSION_IK_LAMBDA", "MISSION_LAYOUT_SEED", "MISSION_NO_BASKET", "MISSION_NO_VIDEO", "MISSION_PERSIST_BUDGET",
    "MISSION_BASKET_USD", "MISSION_BASKET_REACH", "MISSION_MAX_CARRY", "MISSION_USE_BASKET",
    "MISSION_ORACLE_DETECT", "MISSION_PROP_TEX", "MISSION_REAL_NOISE",
    "MISSION_RIND_BOX", "MISSION_SKIP_REDBAG", "MISSION_SQUEEZE", "MISSION_USE_BASKET", "MISSION_MAX_CARRY",
    "MISSION_VENUE_PHOTO", "CUDA_VISIBLE_DEVICES",
)


def dump_config(out_dir: Path, args, ckpts: dict) -> Path:
    """Write the ONE file a run can be reproduced from: every variable the stack reads,
    the five checkpoints it actually loaded with their hashes, and argv.

    The failure this prevents is real and already happened: the 120/160 venue run's
    provenance recorded eight empty variables and hashed nine checkpoints the run never
    opened, so nothing in its record says whether it used the photo or the flat colour
    prototypes -- a 45 %/43 % versus 58 %/76 % difference in the detector.
    """
    import hashlib
    import json as _json

    def _sha(path):
        try:
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()[:16]
        except OSError:
            return None

    cfg = {
        "argv": sys.argv,
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items() if not k.startswith("_")},
        # every MISSION_*/GRASP_*/COURSE_* variable that is SET, plus the fixed list:
        # the route audit found 27 missions whose knobs could not be told apart
        "env": {**{k: os.environ.get(k) for k in CONFIG_VARS},
                **{k: v for k, v in os.environ.items()
                   if k.startswith(("MISSION_", "GRASP_", "COURSE_", "EUNHA_"))}},
        "checkpoints": {k: {"path": v, "sha256_16": _sha(v)} for k, v in ckpts.items()},
    }
    p = Path(out_dir) / "config.json"
    p.write_text(_json.dumps(cfg, indent=1, sort_keys=True, default=str))
    print(f"[LAUNCH] config -> {p}", flush=True)
    return p


def build_and_run(args, profile: Profile):
    device = args.device if getattr(args, "device", None) else "cuda:0"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpts = resolve_ckpts(args)
    for k, v in ckpts.items():
        print(f"[LAUNCH] {k}: {v}", flush=True)
        if not Path(v).exists():
            raise SystemExit(f"missing checkpoint {k}: {v}")
    dump_config(out_dir, args, ckpts)

    # The seed must reach env_cfg BEFORE parse_env_cfg: __post_init__ both builds the
    # layout and immediately turns it into scene spawn entries, so post-hoc assignment
    # (the first fix attempt) changed the bookkeeping list while the WORLD kept the
    # seed-0 placement — a silent registry/world desync. The env var is read inside
    # __post_init__ itself (env_cfg.py, MISSION_LAYOUT_SEED).
    os.environ["MISSION_LAYOUT_SEED"] = str(args.seed)
    env_cfg = parse_env_cfg(TASK, device=device, num_envs=1)
    # None means "whatever the venue says" (env_cfg.PER_TYPE: 3 on the Hong Kong
    # course, 4 on the legacy one). Assigning it unconditionally turned the config
    # default into None and spawned nothing.
    if args.per_type is not None:
        env_cfg.per_type = args.per_type

    # env dt = 0.02 s; render the recording cams at 25 Hz (= the mp4 rate) and the head
    # cam at 10 Hz (= the detection rate) instead of every step — roughly halves the
    # wall-clock of a full run, which is dominated by rendering
    rec = dict(update_period=0.04, height=720, width=1280, data_types=["rgb"],
               spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.1, 250.0)))
    env_cfg.scene.overview_cam = CameraCfg(prim_path="/World/overview_cam", **rec)
    env_cfg.scene.follow_cam = CameraCfg(prim_path="/World/follow_cam", **rec)
    # head cam MUST refresh every step: at 10 Hz its cached pose and its rendered image
    # belong to different moments of the gait, and the unprojected object positions came
    # out 0.2-0.3 m below the ground (every mission acquisition was a phantom).
    env_cfg.scene.head_cam = head_cam_cfg()

    env = gym.make(TASK, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    env.reset()

    cams = {"overview": uenv.scene["overview_cam"], "follow": uenv.scene["follow_cam"],
            "head": uenv.scene["head_cam"]}
    eye = torch.tensor([[0.0, -34.0, 26.0]], device=uenv.device)
    tgt = torch.tensor([[0.0, 2.0, 0.0]], device=uenv.device)
    cams["overview"].set_world_poses_from_view(eye, tgt)

    tag = profile.name
    # MISSION_NO_VIDEO=1 for long scoring runs. Two multi-hour runs went silent at the
    # same instant their mp4s stopped growing (log and video both frozen at 22:04, CPU
    # spinning at 130%, GPU at 2%, no ffmpeg processes left) and never recovered. The
    # score comes out of the log, so the recording is optional weight on a run that has
    # to survive for hours.
    if os.environ.get("MISSION_NO_VIDEO") == "1":
        print("[MISSION] video recording disabled (MISSION_NO_VIDEO=1)", flush=True)
        writers = {}
    else:
        writers = {
            "overview": iio.get_writer(str(out_dir / f"mission_{tag}_overview.mp4"), fps=25),
            "follow": iio.get_writer(str(out_dir / f"mission_{tag}_follow.mp4"), fps=25),
        }
        if args.record_head:
            writers["head"] = iio.get_writer(str(out_dir / f"mission_{tag}_head.mp4"), fps=25)

    runner = MissionRunner(env, uenv, env_cfg, profile, ckpts, cams, writers, out_dir, device)
    try:
        with torch.inference_mode():
            rep = runner.run(float(args.depth))
    finally:
        for w in writers.values():
            w.close()
        env.close()
    print(f"[LAUNCH] videos + report in {out_dir}", flush=True)
    return rep


def add_common_args(parser):
    parser.add_argument("--speed", type=str, default="s075", choices=["s075", "s100"])
    parser.add_argument("--depth", type=str, default="0.24")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per_type", type=int, default=None)
    parser.add_argument("--walk_ckpt", type=str, default="")
    parser.add_argument("--grasp_ckpt", type=str, default="")
    parser.add_argument("--record_head", action="store_true")
    parser.add_argument("--out_dir", type=str, default=str(RW / "course_mission" / "videos"))
    return parser
