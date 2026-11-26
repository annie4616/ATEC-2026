"""B2+Piper grasp+lift RL environment (stage 1: fixed base at the crouch pose).

Locked design decisions implemented here (do not change silently):
  - RL window = grasp + lift only, 4 s episodes (200 steps @ 50 Hz)
  - IK-relative arm action, scale 0.02; gripper as a separate binary action
  - 42-dim obs incl. the reserved grasp-parameter vector (5) and base-state slot (10)
  - procedural primitive objects (no USD assets), width 30-60 mm / height 60-130 mm
    independently sampled prestartup (requires replicate_physics=False), mass/friction/pose
    randomized at reset
  - contact_offset=0.002 / rest_offset=0.0 and solver_position_iteration_count=16 on BOTH
    the object and the robot (covers the gripper fingers)
  - RSI reset from grasp_rl/banks/pregrasp_bank.npz
"""

import json
import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.utils import configclass

from atec_rl_lab.assets.robots import UNITREE_B2_PIPER_CFG

# robot-lineage gate: GRASP_ROBOT=a2piper swaps in the A2 Pro + Piper assembly (arm prim
# names identical, so the rest of the stack is unchanged). Default stays B2+Piper.
if os.environ.get("GRASP_ROBOT") == "a2piper":
    import sys as _sys

    _A2PRO = str(Path(__file__).resolve().parents[2] / "a2pro_rl")
    if _A2PRO not in _sys.path:
        _sys.path.insert(0, _A2PRO)
    from a2pkg.a2piper import UNITREE_A2_PIPER_CFG as _ROBOT_CFG
    print("[GRASPENV] robot lineage: A2 Pro + Piper", flush=True)
else:
    _ROBOT_CFG = UNITREE_B2_PIPER_CFG

from . import mdp
from .mdp.arm_actions import RateLimitedIKActionCfg
from .mdp.leg_actions import FrozenCrouchLegActionCfg

# ---- crouch pose (from Phase 1) ------------------------------------------------------------
# depth selectable via env var so one codebase trains at multiple crouch depths
import os

_GRASP_RL = Path(__file__).resolve().parents[1]
# lineage overrides (all default to the original grasp_rl locations): a new locomotion
# lineage (e.g. wtw) points these at its own folder tree instead of overwriting anything
_POSES_JSON = Path(os.environ.get("GRASP_POSES_JSON", str(_GRASP_RL / "banks" / "crouch_poses.json")))
_POSES = json.loads(_POSES_JSON.read_text())
CROUCH_DEPTH = os.environ.get("GRASP_CROUCH_DEPTH", "0.2")
GRIP_OPEN = float(os.environ.get("GRASP_GRIP_OPEN", "0.035"))
OBJ_MASS = float(os.environ.get("GRASP_OBJ_MASS", "0.15"))  # competition objects: can/bottle 0.2 kg  # PiPER 35mm, PiPER-X 50mm per finger
CROUCH = dict(_POSES["poses"][CROUCH_DEPTH])
# hip clamp: extreme poses (depth 0.26) sit exactly ON the +-0.87 joint limit
CROUCH["joint_pos"] = {k: (max(min(v, 0.868), -0.868) if "_hip_" in k else v)
                       for k, v in CROUCH["joint_pos"].items()}
CROUCH_Z = CROUCH["base_height_mean"]
print(f"[GRASPENV] crouch depth {CROUCH_DEPTH} (base z {CROUCH_Z})", flush=True)

# object base dims [m]; prestartup scale ranges below map them to the spec ranges
_CUBE_BASE = (0.045, 0.045, 0.095)
_CYL_RADIUS, _CYL_HEIGHT = 0.0225, 0.095
# width 30-60 mm on 45 mm base; height 60-130 mm on 95 mm base (hard cap: gripper opens 70 mm)
_SCALE_RANGE = {"x": (0.667, 1.333), "y": (0.667, 1.333), "z": (0.632, 1.368)}

_OBJ_PHYS = sim_utils.RigidBodyMaterialCfg(
    static_friction=0.6, dynamic_friction=0.5, restitution=0.0,
    friction_combine_mode="multiply", restitution_combine_mode="multiply",
)


@configclass
class GraspSceneCfg(InteractiveSceneCfg):
    """Robot fixed in crouch + one procedural object + finger contact sensors."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2500.0)
    )

    robot: ArticulationCfg = _ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.CuboidCfg(size=_CUBE_BASE, physics_material=_OBJ_PHYS),
                sim_utils.CylinderCfg(radius=_CYL_RADIUS, height=_CYL_HEIGHT, physics_material=_OBJ_PHYS),
            ],
            random_choice=True,
            # applied to every child (locked physics values: grasp physics must not lie)
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=OBJ_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.0, 0.06)),
    )

    # NOTE: prim matching is PER PATH SEGMENT — a 4-segment pattern can never match the
    # 5-segment child prims where USD assets carry their rigid body (e.g. Object/_11_banana).
    # List BOTH the root (primitives) and the children (USD assets) explicitly.
    contact_finger_l = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/arm_link7",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object", "{ENV_REGEX_NS}/Object/.*"],
        update_period=0.0,
    )
    contact_finger_r = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/arm_link8",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object", "{ENV_REGEX_NS}/Object/.*"],
        update_period=0.0,
    )


@configclass
class ActionsCfg:
    # RateLimitedIKActionCfg == the stock differential-IK action when GRASP_ARM_RATE and
    # GRASP_ARM_DELAY are unset (its class only adds a clamp/queue around apply_actions),
    # and the REAL command path (10 rad/s slew + 1-step latency) when they are — see
    # graspenv/mdp/arm_actions.py for the measurements that motivated it.
    arm = RateLimitedIKActionCfg(
        asset_name="robot",
        joint_names=["arm_joint[1-6]"],
        body_name="gripper_base",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.02,  # locked: 2 cm/step @ 50 Hz = max 1 m/s; do NOT raise
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=(0.0, 0.0, float(os.environ.get("GRASP_TCP_OFFSET", "0.13")))
        ),
    )
    gripper = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["arm_joint7", "arm_joint8"],
        open_command_expr={"arm_joint7": GRIP_OPEN, "arm_joint8": -GRIP_OPEN},
        close_command_expr={"arm_joint7": 0.0, "arm_joint8": 0.0},
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        ee_pose = ObsTerm(func=mdp.ee_pose_b)                    # 7
        grasp_vec = ObsTerm(func=mdp.grasp_vec_ee)               # 3 (surface point, not center)
        axes = ObsTerm(func=mdp.approach_axes)                   # 6
        gripper_width = ObsTerm(func=mdp.gripper_width)          # 1
        actions = ObsTerm(func=mdp.last_action)                  # 7
        grasp_params = ObsTerm(func=mdp.grasp_params)            # 5 (reserved slot)
        obb_dims = ObsTerm(func=mdp.obb_dims)                    # 3
        base_state = ObsTerm(func=mdp.base_state_const)          # 10 (constant in stage 1)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    # robot default-state reset FIRST (manager-based envs restore nothing by themselves;
    # without these the robot starts at the USD zero pose with zero drive targets)
    reset_robot_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("robot")},
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )
    # size randomization BEFORE sim start (independent per axis; needs replicate_physics=False)
    object_scale = EventTerm(
        func=mdp.randomize_rigid_body_scale,
        mode="prestartup",
        params={"scale_range": _SCALE_RANGE, "asset_cfg": SceneEntityCfg("object")},
    )
    # cache per-env OBB dims once the (scaled) stage exists
    cache_dims = EventTerm(func=mdp.cache_object_dims, mode="startup")

    object_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.2, 0.9),
            "dynamic_friction_range": (0.2, 0.9),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )
    object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.03, 0.4),
            "operation": "abs",
            "recompute_inertia": True,
        },
    )
    grasp_params = EventTerm(func=mdp.sample_grasp_params, mode="reset")
    # RSI reset LAST: writes arm joints + places the object under the fingertips
    rsi_reset = EventTerm(
        func=mdp.reset_pregrasp_rsi,
        mode="reset",
        params={"xy_noise": 0.05},
    )


@configclass
class RewardsCfg:
    reach = RewTerm(func=mdp.reach_grasp_point, weight=2.0, params={"std": 0.1})
    align = RewTerm(func=mdp.align_gripper, weight=0.5)
    grasp = RewTerm(func=mdp.grasp_established, weight=2.0)
    lift = RewTerm(func=mdp.lift_progress, weight=10.0, params={"target_clearance": 0.10})
    # terminal bonus must BEAT the lift reward forgone by ending early (rewards are x dt=0.02:
    # holding to timeout collects up to ~35 of lift; 2500 x 0.02 = 50 makes success dominant)
    success = RewTerm(func=mdp.is_terminated_term, weight=2500.0, params={"term_keys": "grasp_success"})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # keep the wrist/gripper out of the head-camera volume (self-collisions are off, so
    # without this the arm can clip straight through the camera)
    camera_keepout = RewTerm(func=mdp.camera_keepout, weight=-8.0)
    arm_vel = RewTerm(func=mdp.arm_joint_vel_l2, weight=-5.0e-4)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    grasp_success = DoneTerm(
        func=mdp.grasp_success,
        params={"hold_steps": 25, "clearance": 0.05},  # locked: 25-step sustained hold
    )
    object_away = DoneTerm(func=mdp.object_out_of_zone, params={"max_dist": 1.0})


@configclass
class B2PiperGraspEnvCfg(ManagerBasedRLEnvCfg):
    scene: GraspSceneCfg = GraspSceneCfg(num_envs=2048, env_spacing=2.0, replicate_physics=False)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 4.0            # 200 steps @ 50 Hz (locked 150-250 window)
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation

        # ---- robot: fixed base at the crouch pose, stiff-solver + tight contact offsets ----
        r = self.scene.robot
        r.spawn.articulation_props.fix_root_link = True
        r.spawn.articulation_props.solver_position_iteration_count = 16
        r.spawn.articulation_props.solver_velocity_iteration_count = 1
        r.spawn.collision_props = sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0)
        # dedicated finger actuator: the shared "arms" gains (stiffness 80) give only ~2.8 N of
        # grip force on the prismatic fingers -> objects slip out of a closed gripper. Split the
        # fingers into their own group with Piper-like gripper gains.
        r.actuators["arms"].joint_names_expr = "arm_joint[1-6]"
        r.actuators["gripper"] = ImplicitActuatorCfg(
            joint_names_expr="arm_joint[78]",
            effort_limit_sim=30.0,
            velocity_limit_sim=1.0,
            stiffness=1500.0,
            damping=50.0,
            friction=0.01,
            armature=0.01,
        )
        # opt-in arm speed ceiling (GRASP_ARM_VEL_LIMIT=<rad/s>): the asset config lets the
        # arm joints spin at 100 rad/s while the URDF rates them at 10 and the mission
        # (course_mission/env_cfg.py MISSION_ARM_VEL_LIMIT, default 10) clamps them there —
        # a policy trained at 100 sees tracking lag it never met and saturates
        _avl = os.environ.get("GRASP_ARM_VEL_LIMIT")
        if _avl:
            r.actuators["arms"].velocity_limit_sim = float(_avl)
        # opt-in fingertip friction override (GRASP_FINGER_FRICTION=<f>): converted URDFs
        # (PiPER-X) carry default low-friction materials -> smooth cylinders (the 66 mm can)
        # slip during the lift; real gripper kits have rubberized pads
        _ff = os.environ.get("GRASP_FINGER_FRICTION")
        if _ff:
            _f = float(_ff)
            self.events.finger_friction = EventTerm(
                func=mdp.randomize_rigid_body_material, mode="startup",
                params={"asset_cfg": SceneEntityCfg("robot", body_names=["arm_link7", "arm_link8"]),
                        "static_friction_range": (_f, _f), "dynamic_friction_range": (_f * 0.9, _f * 0.9),
                        "restitution_range": (0.0, 0.0), "num_buckets": 2, "make_consistent": True},
            )
            print(f"[GRASPENV] fingertip friction override: {_f}", flush=True)

        arm_ready = {
            "arm_joint1": 0.0, "arm_joint2": 1.2, "arm_joint3": -1.2,
            "arm_joint4": 0.0, "arm_joint5": 0.6, "arm_joint6": 0.0,
            "arm_joint7": GRIP_OPEN, "arm_joint8": -GRIP_OPEN,
        }
        r.init_state = r.init_state.replace(
            pos=(0.0, 0.0, CROUCH_Z),
            joint_pos={**CROUCH["joint_pos"], **arm_ready},
            joint_vel={".*": 0.0},
        )


# crouch policy checkpoint for the frozen leg controller (read-only location)
_CROUCH_RUNS = Path(os.environ.get(
    "GRASP_CROUCH_RUNS",
    str(Path(__file__).resolve().parents[2] / "logs" / "rsl_rl" / "crouch_sweep" / "runs"),
))
_LEG_CLOCK = os.environ.get("GRASP_LEG_CLOCK") == "1"  # WTW lineage: 49-obs frozen legs
_matches = sorted(_CROUCH_RUNS.glob(f"crouch{CROUCH_DEPTH}_*/model_2999.pt"))
if not _matches and CROUCH_DEPTH == "0.3":
    # depth 0.3 = the ORIGINAL crouch policy (pre-sweep, commanded height 0.30)
    _matches = sorted(_CROUCH_RUNS.glob("b2_crouch_cycle_*/model_*.pt"),
                      key=lambda p: int(p.stem.split("_")[1]))
_CROUCH_CKPT = _matches[-1]


@configclass
class B2PiperGraspStage2EnvCfg(B2PiperGraspEnvCfg):
    """Stage 2: base free-floating, legs driven by the FROZEN crouch policy, real base obs."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1024  # locked plan: fewer envs for stage 2
        # free the base; the frozen crouch policy balances it
        self.scene.robot.spawn.articulation_props.fix_root_link = False
        self.actions.legs = FrozenCrouchLegActionCfg(
            asset_name="robot", checkpoint=str(_CROUCH_CKPT), use_clock=_LEG_CLOCK
        )
        # gentle terrain noise (opt-in: GRASP_TERRAIN_NOISE=<amp in m>, stage 2 only —
        # stage 1 welds the base, feet would interpenetrate bumps). ±2 cm verified safe
        # for the origin-z clearance/success measurements (5 cm threshold).
        _tn = float(os.environ.get("GRASP_TERRAIN_NOISE", "0") or 0)
        if _tn > 0:
            import isaaclab.terrains as _terrain_gen
            from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg

            self.scene.ground = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="generator",
                max_init_terrain_level=None,
                collision_group=-1,
                terrain_generator=TerrainGeneratorCfg(
                    size=(2.0, 2.0), border_width=5.0, num_rows=32, num_cols=32,
                    horizontal_scale=0.05, vertical_scale=0.005, slope_threshold=0.75,
                    use_cache=False, curriculum=False,
                    sub_terrains={
                        "noise": _terrain_gen.HfRandomUniformTerrainCfg(
                            proportion=1.0, noise_range=(0.005, _tn), noise_step=0.005,
                            border_width=0.1,
                        ),
                    },
                ),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply", restitution_combine_mode="multiply",
                    static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
                ),
                debug_vis=False,
            )
            # spawn the object clear of local bumps (drop_height >= amplitude)
            self.events.rsi_reset.params["drop_height"] = _tn + 0.005

        # robot USD override (opt-in): e.g. the basket-equipped variant for deposit demos
        if os.environ.get("GRASP_ROBOT_USD"):
            self.scene.robot.spawn.usd_path = os.environ["GRASP_ROBOT_USD"]

        # EUNHA lineage (opt-in): 53-obs RMA crouch policy with frozen conv adaptation
        if os.environ.get("GRASP_LEG_EUNHA") == "1":
            from .mdp.leg_actions_eunha import EunhaCrouchLegActionCfg

            self.actions.legs = EunhaCrouchLegActionCfg(
                asset_name="robot", checkpoint=str(_CROUCH_CKPT),
                adaptation=os.environ["EUNHA_ADAPT_CKPT"],
            )
        # fill the reserved 10-dim base-state slot with real values (obs layout unchanged)
        self.observations.policy.base_state.func = mdp.base_state_real
        self.observations.critic.base_state.func = mdp.base_state_real
        # guard: crouch tipped over / collapsed
        self.terminations.base_fell = DoneTerm(func=mdp.base_fell)
        # camera collision = mission failure (2 control steps = 0.04 s inside the box; the
        # box check itself samples the arm densely, so a single grazing step already fails)
        self.terminations.camera_hit = DoneTerm(func=mdp.camera_hit, params={"hold_steps": 2})
        # NOTE (2026-07-26, user decision): self-collisions stay DISABLED — the real-robot
        # goal is grasping that never touches the body at all, so the policy must learn
        # avoidance purely from the keep-out penalty + camera_hit termination, not lean on
        # the physics engine resolving arm-body contacts.
        # moving front feet stomp objects placed close to the body -> keep the RSI object
        # placement beyond the front-foot zone and tighten the placement noise
        self.events.rsi_reset.params["min_ee_x"] = 0.50
        self.events.rsi_reset.params["xy_noise"] = 0.03

        # ---- anti-roll shaping (opt-in: GRASP_DISTURB_W>0) ---------------------------------
        # penalize object speed while not stably grasped: rolling cylinders (can/bottle)
        # dominate object_away failures — teach a bump-free single-attempt grasp
        _dw = float(os.environ.get("GRASP_DISTURB_W", "0") or 0)
        if _dw > 0:
            self.rewards.object_disturb = RewTerm(func=mdp.object_disturb, weight=-_dw)

        # ---- real-world robustness layer (opt-in: GRASP_REAL_NOISE=1) ----------------------
        # On top of the RGB+LiDAR perception noise: proprioceptive sensor noise (POLICY obs
        # only — the critic keeps exact state, asymmetric actor-critic) plus dynamics
        # randomization (object friction/mass, fingertip friction, arm PD gains).
        if os.environ.get("GRASP_REAL_NOISE") == "1":
            from isaaclab.utils.noise import UniformNoiseCfg as Unoise

            pol = self.observations.policy
            pol.enable_corruption = True
            pol.ee_pose.noise = Unoise(n_min=-0.004, n_max=0.004)        # encoder/FK error
            pol.gripper_width.noise = Unoise(n_min=-0.002, n_max=0.002)
            pol.base_state.noise = Unoise(n_min=-0.02, n_max=0.02)       # IMU quat/gyro/acc
            pol.axes.noise = Unoise(n_min=-0.01, n_max=0.01)
            self.events.real_object_material = EventTerm(
                func=mdp.randomize_rigid_body_material, mode="startup",
                params={"asset_cfg": SceneEntityCfg("object"),
                        "static_friction_range": (0.4, 1.2), "dynamic_friction_range": (0.3, 1.0),
                        "restitution_range": (0.0, 0.1), "num_buckets": 64, "make_consistent": True},
            )
            self.events.real_finger_material = EventTerm(
                func=mdp.randomize_rigid_body_material, mode="startup",
                params={"asset_cfg": SceneEntityCfg("robot", body_names=["arm_link7", "arm_link8"]),
                        "static_friction_range": (0.6, 1.2), "dynamic_friction_range": (0.5, 1.0),
                        "restitution_range": (0.0, 0.0), "num_buckets": 64, "make_consistent": True},
            )
            self.events.real_object_mass = EventTerm(
                func=mdp.randomize_rigid_body_mass, mode="startup",
                params={"asset_cfg": SceneEntityCfg("object"),
                        "mass_distribution_params": (0.7, 1.3), "operation": "scale"},
            )
            self.events.real_arm_gains = EventTerm(
                func=mdp.randomize_actuator_gains, mode="reset",
                params={"asset_cfg": SceneEntityCfg("robot", joint_names=["arm_joint[1-6]"]),
                        "stiffness_distribution_params": (0.9, 1.1),
                        "damping_distribution_params": (0.9, 1.1),
                        "operation": "scale", "distribution": "uniform"},
            )
            print("[GRASPENV] real-world noise layer ON (obs corruption + dynamics rand)", flush=True)
