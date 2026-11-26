"""A2 Pro + Piper arm articulation cfg (OFFICIAL PiPER-X merged at URDF level (100 mm gripper), so all
arm joint/body names match the B2 grasp stack). Camera at URDF (0.338, 0.034, 0.053)."""

from copy import deepcopy
import os

from isaaclab.actuators import ImplicitActuatorCfg

from .a2 import UNITREE_A2_CFG

A2_PIPER_USD_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "a2_piperx.usd")

UNITREE_A2_PIPER_CFG = deepcopy(UNITREE_A2_CFG)
UNITREE_A2_PIPER_CFG.spawn.usd_path = str(A2_PIPER_USD_PATH)
UNITREE_A2_PIPER_CFG.spawn.articulation_props.enabled_self_collisions = False
UNITREE_A2_PIPER_CFG.actuators["arms"] = ImplicitActuatorCfg(
    joint_names_expr="arm_joint.*",
    effort_limit_sim=100.0,
    velocity_limit_sim=100.0,
    stiffness=80.0,
    damping=4.0,
    friction=0.01,
    armature=0.01,
)
UNITREE_A2_PIPER_CFG.init_state.joint_pos = {
    **UNITREE_A2_PIPER_CFG.init_state.joint_pos,
    "arm_joint[1-6]": 0.0,
    "arm_joint7": 0.05,
    "arm_joint8": -0.05,
}
