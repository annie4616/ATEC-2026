"""PPO runner configs for A2 Pro — the B2 runners with new experiment names."""

from isaaclab.utils import configclass

from atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2.agents.rsl_rl_ppo_cfg import (
    UnitreeB2FlatPPORunnerCfg,
)


@configclass
class UnitreeA2FlatPPORunnerCfg(UnitreeB2FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_a2_flat"


@configclass
class UnitreeA2FlatWTWPPORunnerCfg(UnitreeA2FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_a2_flat_wtw"
