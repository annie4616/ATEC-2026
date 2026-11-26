"""Task D RL: WALK-ACROSS crossing WITH dense far-rim climb shaping.

Extends UnitreeB2TaskDCrossWalkEnvCfg. Every prior CrossWalk run plateaued with
the far-rim milestone (taskd_m_farrim) stuck at 0: the robot learns to step DOWN
onto the box top (nearrim ~0.20, onbox ~0.31) but never CLIMBS the ~0.4 m far rim
back onto flat ground (the +20 / x>2.0 blocker). Mean reward then drifts negative
because stand_still + joint costs accumulate while the robot is stuck on the box.

Root cause (diagnosed from the 88-demo z-profile, root_pos_seq.npy):
  approach   x[-1.1,-0.6)  z~0.55  (flat)
  onbox      x[-0.1,+0.4)  z 0.30->0.54 (step down onto box top)
  far-rim    x[+0.4,+1.1)  z 0.16 -> 0.55  (the CLIMB: lowest 0.163 against the
                                            far wall, then up to flat 0.55)
The sparse far-rim milestone gives NO gradient until the robot is already up, and
progress_x goes ~0 against the vertical step. So there was no signal telling the
policy HOW to lift its body up the rim.

Fixes here:
  1. farrim_climb_band (NEW dense term): normalized base-height fraction in the
     far-rim x-band [0.40,1.10], paying more the higher the base climbs from
     z~0.16 toward z~0.55. This is the missing dense vertical gradient.
  2. Soften stand_still (-5 -> -1): the robot legitimately pauses to set up the
     climb; the harsh penalty was collapsing reward and discouraging commitment.
  3. Keep the milestone carrots but the climb band now bridges onbox -> farrim.

Warm-starts from cwk_rough_ws/model_1999.pt (same obs/action interface).
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import atec_rl_lab.train.locomotion.taskd.mdp as taskd_mdp
from atec_rl_lab.train.locomotion.taskd.config.quadruped.unitree_b2.taskd_crosswalk_env_cfg import (
    UnitreeB2TaskDCrossWalkEnvCfg,
)


@configclass
class UnitreeB2TaskDCrossWalkClimbEnvCfg(UnitreeB2TaskDCrossWalkEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ---- NEW: dense far-rim climb gradient (the missing signal) ----------
        # Pays the normalized base-height fraction (0 at z=0.16 on the box top,
        # 1 at z=0.55 back on flat) inside the far-rim x-band. Weight is large so
        # the cumulative dense reward over the climb (~tens of steps) outweighs the
        # joint/stand-still costs that previously made staying on the box optimal.
        self.rewards.taskd_farrim_climb = RewTerm(
            func=taskd_mdp.farrim_climb_band,
            weight=8.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "x_lo": 0.40, "x_hi": 1.10,
                "z_lo": 0.16, "z_hi": 0.55,
            },
        )

        # ---- Soften stand_still: it was collapsing reward on the box top -----
        # The robot needs a beat to plant its feet before stepping up the rim;
        # a -5 penalty per step made "never commit to the pit" the safe optimum.
        self.rewards.stand_still.weight = -1.0

        # NOTE: do NOT call disable_zero_weight_rewards() again here — the parent
        # CrossWalk __post_init__ already ran it, setting zero-weight terms to None.
        # A second pass iterates those None attrs and crashes (NoneType has no
        # .weight). Our additions (climb band +8, stand_still -1) are both nonzero.
