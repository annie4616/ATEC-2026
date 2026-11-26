"""PPO runner cfgs for the Eunha lineage. Network arch stays [512,256,128] ELU so the
partial actor warm-start from model_14999.pt maps 1:1 (critic trains fresh)."""

from __future__ import annotations

from isaaclab.utils import configclass
from atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2.agents.rsl_rl_ppo_cfg import (
    UnitreeB2FlatPPORunnerCfg,
)


@configclass
class EunhaCrouchPPORunnerCfg(UnitreeB2FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "eunha_crouch"
        self.max_iterations = 3000


@configclass
class EunhaFlatPPORunnerCfg(UnitreeB2FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "eunha_flat"
