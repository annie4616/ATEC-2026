"""Unitree A2 Pro articulation config — mirrors UNITREE_B2_CFG, scaled to the A2.

From a2.urdf: 40.1 kg total, legs FL/FR/RL/RR_{hip,thigh,calf} (+fixed *_foot),
thigh/calf links 0.275 m each (B2: 0.35), limits hip +-1.01, thigh front [-2.34,3.15] /
hind [-1.56,3.94], calf [-2.77,-0.54], efforts 120/120/180 Nm, velocities 22/22/14.7.
Standing pose (thigh 0.8/1.0, calf -1.5) -> base height ~0.43 m (B2: ~0.55).
Head camera at URDF camera_joint (0.338, 0.034, 0.053); front lidar on base_link too —
matches the RGB+LiDAR deployment assumption.
"""

import os

import numpy as np
from scipy.spatial.transform import Rotation as R

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import CameraCfg

from atec_rl_lab.assets.robots.cfg import ATECArticulationCfg

A2_USD_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "a2.usd")

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
        pos=(0.0, 0.0, 0.45),
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
        # gains scaled from B2 (160/5 at ~60 kg) by the ~2/3 mass ratio
        "hip_thigh": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_.*", ".*_thigh_.*"],
            effort_limit_sim=120,
            velocity_limit_sim=22,
            stiffness=120.0,
            damping=4.0,
            friction=0.01,
            armature=0.01,
        ),
        "calf": ImplicitActuatorCfg(
            joint_names_expr=[".*_calf_.*"],
            effort_limit_sim=180,
            velocity_limit_sim=14,
            stiffness=120.0,
            damping=4.0,
            friction=0.01,
            armature=0.01,
        ),
    },
    soft_joint_pos_limit_factor=0.9,
    head_camera_offset=CameraCfg.OffsetCfg(
        pos=(0.3381, 0.0336, 0.0525),
        rot=tuple(float(x) for x in R.from_euler("xyz", [0.0, 0.0, 0.0]).as_quat(scalar_first=True)),
        convention="world",
    ),
)
