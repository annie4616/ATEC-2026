"""A2 Pro flat velocity env — the B2 flat cfg with the robot swapped and height-scale
parameters adapted to the smaller platform (legs 0.55 m vs B2 0.7 m). Everything else
(observation layout 45/48, actions 12, rewards, DR, terminations, PPO) stays stock, so
policies/checkpoints remain interface-compatible with the whole B2 tooling."""

from isaaclab.utils import configclass

from atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2.flat_env_cfg import (
    UnitreeB2FlatEnvCfg,
)

from .a2 import UNITREE_A2_CFG


@configclass
class UnitreeA2FlatEnvCfg(UnitreeB2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_A2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # height-scale overrides (A2 stands ~0.43 vs B2 ~0.55)
        self.rewards.base_height_l2.params["target_height"] = 0.43
        self.rewards.feet_height_body.params["target_height"] = -0.32
        if self.__class__.__name__ == "UnitreeA2FlatEnvCfg":
            self.disable_zero_weight_rewards()
