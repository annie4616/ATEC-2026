"""Configuration for the Unitree A2 Pro quadruped + AgileX Piper X arm.

The robot changed from B2+Piper to **A2 Pro + Piper X** (2026-08). Assets were built by
converting the two source URDFs to USD (IsaacLab convert_urdf) and rigidly fixing the
arm base to the A2 base_link, giving ONE articulation:
  - 12 leg DoF : FL/FR/RL/RR x {hip, thigh, calf}  (foot is a fixed link)
  - 8 arm DoF  : joint1..joint6 (revolute) + joint7/joint8 (prismatic gripper)
Mirrors b2.py so downstream task/env code needs only the CFG swap.

A2 is much lighter than B2 (base_link mass ~19.65 kg vs B2 ~60 kg). Leg URDF efforts are
120 N·m (hip/thigh) / 180 N·m (calf); we keep B2-style implicit PD (stiffness 160 / damping
5) as a starting point — revisit gains once parkour training bootstraps (see the B2 parkour
run notes: too-weak gains made the heavy robot sit down; A2 lighter so 160/5 should be ample).
"""

import os
import numpy as np
from copy import deepcopy
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from atec_rl_lab.assets.robots.cfg import ATECArticulationCfg
from isaaclab.sensors import CameraCfg
from atec_rl_lab.assets import ATEC_ASSETS_MODEL_DIR
from scipy.spatial.transform import Rotation as R

# A2-only USD (legs) and the composed A2+PiperX USD (legs + arm).
A2_USD_PATH = os.path.join(ATEC_ASSETS_MODEL_DIR, "robot/a2/a2_usd/a2.usd")
A2_PIPER_USD_PATH = os.path.join(ATEC_ASSETS_MODEL_DIR, "robot/a2/a2_piper.usda")

UNITREE_A2_CFG = ATECArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{A2_USD_PATH}",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ATECArticulationCfg.InitialStateCfg(
        # A2 stands a touch lower than B2; 0.50 keeps the folded legs clear of the ground.
        pos=(0.0, 0.0, 0.50),
        joint_pos={
            ".*R_hip_joint": -0.1,
            ".*L_hip_joint": 0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # hip + thigh (A2 URDF effort 120, vel 22)
        "legs_hip_thigh": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_.*", ".*_thigh_.*"],
            effort_limit_sim=120,
            velocity_limit_sim=22,
            stiffness=160.0,
            damping=5.0,
            friction=0.01,
            armature=0.01,
        ),
        # calf (A2 URDF effort 180, vel ~14.7)
        "legs_calf": ImplicitActuatorCfg(
            joint_names_expr=[".*_calf_.*"],
            effort_limit_sim=180,
            velocity_limit_sim=14,
            stiffness=160.0,
            damping=5.0,
            friction=0.01,
            armature=0.01,
        ),
    },
    soft_joint_pos_limit_factor=0.9,
    # Head camera on the A2 base_link. Reuse the B2 forward/down mount as a starting pose;
    # refine against the A2 body geometry when wiring up Task A perception.
    head_camera_offset=CameraCfg.OffsetCfg(
        pos=(0.4216099977493286, 0.02500000037252903, 0.06185099855065346),
        rot=tuple(float(x) for x in R.from_euler("xyz", [0.0, np.pi / 6, 0.0]).as_quat(scalar_first=True)),
        convention="world",
    ),
)

UNITREE_A2_CFG.leg_joint_names = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
UNITREE_A2_CFG.joint_names = list(UNITREE_A2_CFG.leg_joint_names)


# --- A2 + Piper X (legs + arm) ------------------------------------------------------------
UNITREE_A2_PIPER_CFG = deepcopy(UNITREE_A2_CFG)
UNITREE_A2_PIPER_CFG.spawn.articulation_props.enabled_self_collisions = False
UNITREE_A2_PIPER_CFG.spawn.usd_path = str(A2_PIPER_USD_PATH)

# Arm init inside joint limits (joint2 in [0, 3.14], joint3 in [-2.97, 0]); a slightly
# tucked pose keeps the arm off the ground during locomotion.
UNITREE_A2_PIPER_CFG.init_state.joint_pos.update({
    "joint1": 0.0,
    "joint2": 0.8,
    "joint3": -0.8,
    "joint4": 0.0,
    "joint5": 0.0,
    "joint6": 0.0,
    "joint7": 0.0,   # gripper prismatic in [0, 0.05]
    "joint8": 0.0,   # gripper prismatic in [-0.05, 0]
})

# Piper X arm actuators (URDF effort 100). joint7/8 are the prismatic gripper.
UNITREE_A2_PIPER_CFG.actuators["arms"] = ImplicitActuatorCfg(
    joint_names_expr=["joint[1-6]"],
    effort_limit_sim=100.0,
    velocity_limit_sim=100.0,
    stiffness=80.0,
    damping=4.0,
    friction=0.01,
    armature=0.01,
)
UNITREE_A2_PIPER_CFG.actuators["gripper"] = ImplicitActuatorCfg(
    joint_names_expr=["joint[78]"],
    effort_limit_sim=100.0,
    velocity_limit_sim=100.0,
    stiffness=80.0,
    damping=4.0,
    friction=0.01,
    armature=0.01,
)

# Piper X gripper base link is "Link6" (carries gripper_base mesh); the fingers are Link7/8.
# In the COMPOSED a2_piper.usda the arm is referenced under a "/arm" Xform (see the FixedJoint
# body1 = /a2_piperx/arm/base_link), so at runtime the arm links live at /Robot/arm/LinkN — the
# ee-camera link name must include that "arm/" prefix or the Task A env's ee_camera spawn fails
# with "Unable to find source prim path '/Robot/Link6'".
UNITREE_A2_PIPER_CFG.ee_camera_link_name = "arm/Link6"
UNITREE_A2_PIPER_CFG.ee_camera_offset = CameraCfg.OffsetCfg(
    pos=(-0.05, 0.0, 0.06),
    rot=tuple(float(x) for x in R.from_euler("xyz", [0.0, 0.0, -np.pi / 2]).as_quat(scalar_first=True)),
    convention="ros",
)

UNITREE_A2_PIPER_CFG.leg_joint_names = list(UNITREE_A2_CFG.leg_joint_names)
UNITREE_A2_PIPER_CFG.arm_joint_names = [
    "joint1", "joint2", "joint3", "joint4",
    "joint5", "joint6", "joint7", "joint8",
]
UNITREE_A2_PIPER_CFG.joint_names = UNITREE_A2_PIPER_CFG.leg_joint_names + UNITREE_A2_PIPER_CFG.arm_joint_names
