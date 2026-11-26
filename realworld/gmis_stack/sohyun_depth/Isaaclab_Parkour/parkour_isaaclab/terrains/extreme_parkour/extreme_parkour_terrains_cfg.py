from isaaclab.utils import configclass
from ..parkour_terrain_generator_cfg import ParkourSubTerrainBaseCfg
from . import extreme_parkour_terrians

@configclass
class ExtremeParkourRoughTerrainCfg(ParkourSubTerrainBaseCfg):
    apply_roughness: bool = True 
    apply_flat: bool = False 
    downsampled_scale: float | None = 0.075
    noise_range: tuple[float,float] = (0.02, 0.06)
    noise_step: float = 0.005
    x_range: tuple[float, float] = (0.8, 1.5)
    y_range: tuple[float, float] = (-0.4, 0.4)
    half_valid_width: tuple[float, float] = (0.6, 1.2)
    pad_width: float = 0.1 
    pad_height: float = 0.0

@configclass
class ExtremeParkourGapTerrainCfg(ExtremeParkourRoughTerrainCfg):
    function = extreme_parkour_terrians.parkour_gap_terrain
    gap_size: str = '0.1 + 0.7*difficulty'
    gap_depth: tuple[float, float] = (0.2, 1)

@configclass
class ExtremeParkourDoubleStepUpTerrainCfg(ExtremeParkourRoughTerrainCfg):
    """Consecutive UP-only steps (climb-only staircase). Default: two 0.6 m steps =
    1.2 m total climb, to test repeated step-up above the trained 0.45 m height."""
    function = extreme_parkour_terrians.parkour_double_step_up_terrain
    step_height: float = 0.6     # height of each step
    num_steps: int = 2           # number of consecutive up-steps
    step_len: float = 0.8        # tread length (x) of each step
    platform_len: float = 2.0    # flat run before the first riser

@configclass
class ExtremeParkourStaircaseTerrainCfg(ExtremeParkourRoughTerrainCfg):
    """A multi-step STAIRCASE matching the competition course (steps up, <=0.17 m each,
    ~0.35 m tread). step_height is a difficulty-scaled expression so the curriculum climbs from
    shallow to the course height limit. Uses the up-only staircase builder."""
    function = extreme_parkour_terrians.parkour_double_step_up_terrain
    step_height: str = '0.07 + 0.10*difficulty'   # 0.07 m (easy) -> 0.17 m (max, course limit)
    num_steps: int = 5                            # up to 5 consecutive steps
    step_len: float = 0.35                        # tread length (x) of each step (course value)
    platform_len: float = 1.5                     # flat run before the first riser

@configclass
class ExtremeParkourBoxPlatformTerrainCfg(ExtremeParkourRoughTerrainCfg):
    """Task D box-as-stepping-stone PLATFORM climb (post box-push), as parkour terrain.
    flat -> BOX step (box_h, box_len tread) -> PLATFORM top (platform_h) forward.
    Matches the deployment scene the head_camera sees during the climb."""
    function = extreme_parkour_terrians.parkour_box_platform_terrain
    box_h: float = 0.6           # box top height (the pushed 0.6 m box)
    box_len: float = 0.8         # box tread length in x (box is 0.8 m deep)
    box_width: float = 1.0       # box width in y (finite, like the submission box)
    platform_h: float = 1.0      # platform top height

@configclass
class ExtremeParkourTaskDPitTerrainCfg(ExtremeParkourRoughTerrainCfg):
    """Static reproduction of the Task D box-as-stepping-stone pit (post box-push).
    Measured geometry: pit x-width 0.8 m, depth 1.0 m, stone top 0.45 m below the rim
    (the verified parkour step-clearing height)."""
    function = extreme_parkour_terrians.parkour_taskd_pit_terrain
    pit_width: float = 0.8          # along travel (x)
    pit_depth: float = 1.0          # pit floor below the rim
    step_drop: float = 0.45         # stone top below the rim = the step to clear
    stone_len: float = 0.5          # stone size along x (box footprint)
    stone_half_width: float = 0.5   # stone half-size along y

@configclass
class ExtremeParkourHurdleTerrainCfg(ExtremeParkourRoughTerrainCfg):
    function = extreme_parkour_terrians.parkour_hurdle_terrain
    stone_len: str = '0.1 + 0.3 * difficulty'
    hurdle_height_range: str = '0.1 + 0.1 * difficulty, 0.15 + 0.15 * difficulty'

@configclass
class ExtremeParkourStepTerrainCfg(ExtremeParkourRoughTerrainCfg):
    function = extreme_parkour_terrians.parkour_step_terrain
    step_height: str = '0.1 + 0.35*difficulty'
    # Scale ONLY the first step's tread length (x, travel direction) by this factor.
    # 1.0 = unchanged. >1 gives the robot a longer first step to land/stabilize on
    # before the next up-step (used in the StepPlay env to make the first step easier).
    first_step_tread_scale: float = 1.0

@configclass
class ExtremeParkourTerrainCfg(ExtremeParkourRoughTerrainCfg):
    function = extreme_parkour_terrians.parkour_terrain
    pit_depth: tuple[float, float] = (0.2, 1)
    stone_width: float = 1.0
    last_stone_len: float =1.6
    x_range: str = '-0.1, 0.1+0.3*difficulty'
    y_range: str = '0.2, 0.3+0.1*difficulty'
    stone_len: str = '0.9 - 0.3*difficulty, 1 - 0.2*difficulty'
    incline_height: str = '0.25*difficulty'
    last_incline_height: str = 'incline_height + 0.1 - 0.1*difficulty'

@configclass
class ExtremeParkourDemoTerrainCfg(ExtremeParkourRoughTerrainCfg):
    function = extreme_parkour_terrians.parkour_demo_terrain
