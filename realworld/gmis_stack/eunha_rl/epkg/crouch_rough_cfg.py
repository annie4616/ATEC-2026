"""GENTLE-rough crouch env for the Eunha lineage.

Competition-realistic roughness (not parkour): random heightfield noise 1-4 cm plus
mild pyramid slopes (<= ~8.5 deg). Amplitudes are chosen so the origin-z-based height
rewards/gates stay valid (noise << the 3 cm depth gate + 6 cm reward std). Terrain
curriculum stays off (inherited None from the flat chain).
"""

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass

from epkg.crouch_cfg import A2CrouchEunhaEnvCfg

GENTLE_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=10.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.7, noise_range=(0.01, 0.04), noise_step=0.01, border_width=0.25,
        ),
        "gentle_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.0, 0.15), platform_width=2.0, border_width=0.25,
        ),
        "gentle_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.0, 0.15), platform_width=2.0, border_width=0.25,
        ),
    },
)


@configclass
class A2CrouchEunhaRoughEnvCfg(A2CrouchEunhaEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=GENTLE_TERRAINS_CFG,
            max_init_terrain_level=None,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            debug_vis=False,
        )
