"""PPO runner config for B2 Task D RL (same hyperparameters as B2 rough)."""

from isaaclab.utils import configclass

from atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2.agents.rsl_rl_ppo_cfg import (
    UnitreeB2RoughPPORunnerCfg,
)


@configclass
class UnitreeB2TaskDPPORunnerCfg(UnitreeB2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 10000
        self.experiment_name = "taskd_b2"


@configclass
class UnitreeB2FlatPretrainPPORunnerCfg(UnitreeB2RoughPPORunnerCfg):
    """Flat-ground walking pretrain. Same net/hparams as Task D so the checkpoint
    loads directly into `taskd_b2` for warm-starting."""

    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 1500
        self.experiment_name = "taskd_b2_flat_pretrain"
