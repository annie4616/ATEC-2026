"""PPO runner cfg for ATEC-Crouch-B2-v0.

Same hyperparameters / network architecture as the flat-locomotion donor
(UnitreeB2FlatPPORunnerCfg) — this MUST stay identical (hidden [512,256,128] ELU,
normalization off) so the warm-start ``runner.load`` of the donor checkpoint matches.
Only the experiment_name and iteration budget differ so logs/checkpoints land in a new dir.
"""

from __future__ import annotations

from isaaclab.utils import configclass
from atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2.agents.rsl_rl_ppo_cfg import (
    UnitreeB2FlatPPORunnerCfg,
)


@configclass
class B2CrouchPPORunnerCfg(UnitreeB2FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "b2_crouch"
        self.max_iterations = 3000
        # Keep policy arch identical to the donor (do NOT change actor/critic dims or init_noise_std).
