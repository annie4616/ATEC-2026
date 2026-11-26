from parkour_isaaclab.terrains.parkour_terrain_generator_cfg import ParkourTerrainGeneratorCfg
from parkour_isaaclab.terrains.extreme_parkour import * 

EXTREME_PARKOUR_TERRAINS_CFG = ParkourTerrainGeneratorCfg(
    size=(16.0, 4.0),
    border_width=20.0,
    num_rows=10,
    num_cols=40,
    horizontal_scale=0.08, ## original scale is 0.05, But Computing issue in IsaacLab see this issue in https://github.com/isaac-sim/IsaacLab/issues/2187
    vertical_scale=0.005,
    slope_threshold=1.5,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum= True,
    sub_terrains={
        "parkour_gap": ExtremeParkourGapTerrainCfg(
                        proportion=0.15,
                        apply_roughness=True,
                        x_range = (0.8, 1.5),
                        half_valid_width = (0.6, 1.2),
                        gap_size = '0.1 + 0.7*difficulty'
                        ),
        "parkour_hurdle": ExtremeParkourHurdleTerrainCfg(
                        proportion=0.15,
                        apply_roughness=True,
                        x_range = (1.2, 2.2),
                        half_valid_width = (0.4,0.8),
                        hurdle_height_range= '0.1+0.1*difficulty, 0.15+0.25*difficulty'
                        ),
        "parkour_flat": ExtremeParkourHurdleTerrainCfg(
                        proportion=0.15,
                        apply_roughness=True,
                        apply_flat=True,
                        x_range = (1.2, 2.2),
                        half_valid_width = (0.4,0.8),
                        hurdle_height_range= '0.1+0.1*difficulty, 0.15+0.15*difficulty'
                        ),
        "parkour_step": ExtremeParkourStepTerrainCfg(
                        proportion=0.15,
                        apply_roughness=True,
                        x_range = (0.3,1.5),
                        half_valid_width = (0.5, 1),
                        step_height = '0.1 + 0.35*difficulty'
                        ),
        # multi-step STAIRCASE matching the competition course (4 x ~0.15 m steps). Added so the
        # teacher learns consecutive stairs (its old terrains only had SINGLE hurdles/steps, so it
        # -- and the distilled student -- stumbled on the course's 4-step staircase).
        "parkour_staircase": ExtremeParkourStaircaseTerrainCfg(
                        proportion=0.20,
                        apply_roughness=True,
                        half_valid_width = (0.5, 1.0),
                        step_height = '0.07 + 0.10*difficulty',   # 0.07 m (easy) -> 0.17 m (max, course limit)
                        num_steps = 5,                            # up to 5 consecutive steps
                        step_len = 0.35,
                        platform_len = 1.5,
                        ),
        "parkour": ExtremeParkourTerrainCfg(
                        proportion=0.2,
                        apply_roughness=True,
                        x_range  = '-0.1, 0.1+0.3*difficulty',
                        y_range  = '0.2, 0.3+0.1*difficulty',
                        stone_len  = '0.9 - 0.3*difficulty, 1 - 0.2*difficulty',
                        incline_height = '0.25*difficulty',
                        last_incline_height = 'incline_height + 0.1 - 0.1*difficulty'
                        ),
        "parkour_demo": ExtremeParkourDemoTerrainCfg(
                        proportion=0.0,
                        apply_roughness=True,
                        ),

    },
)


# STAIRCASE-only terrain (5 consecutive up-steps, <=0.17 m each = the competition course
# staircase) so the teacher can be played/inspected on the 5-step staircase ALONE (the mixed
# EXTREME_PARKOUR play spreads a single staircase col among gap/hurdle/step, hard to see).
# step_height = '0.07 + 0.10*difficulty' -> 0.17 m at difficulty 1.0.
STAIRCASE_TERRAINS_CFG = ParkourTerrainGeneratorCfg(
    size=(16.0, 4.0),
    border_width=20.0,
    num_rows=10,
    num_cols=40,
    horizontal_scale=0.08,
    vertical_scale=0.005,
    slope_threshold=1.5,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "parkour_staircase": ExtremeParkourStaircaseTerrainCfg(
                        proportion=1.0,
                        apply_roughness=True,
                        half_valid_width=(0.5, 1.0),
                        step_height='0.07 + 0.10*difficulty',
                        num_steps=5,
                        step_len=0.35,
                        platform_len=1.5,
                        ),
    },
)


# Task D pit terrain: statically reproduces the Task D box-as-stepping-stone pit
# (flat -> 0.8m-wide / 1.0m-deep pit with a 0.45m-below-rim stone -> flat) = the REAL
# Task D walk-across (down into the box-filled pit, then climb the far rim back to flat).
# Used to RL-train the teacher on the down-then-up pit cross (STEP-only didn't transfer).
TASKD_PIT_TERRAINS_CFG = ParkourTerrainGeneratorCfg(
    size=(16.0, 4.0),
    border_width=20.0,
    num_rows=10,
    num_cols=40,
    horizontal_scale=0.08,
    vertical_scale=0.005,
    slope_threshold=1.5,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "taskd_pit": ExtremeParkourTaskDPitTerrainCfg(
                        proportion=1.0,
                        apply_roughness=True,
                        pit_width=0.8,
                        pit_depth=1.0,
                        step_drop=0.45,
                        stone_len=0.5,
                        stone_half_width=0.5,
                        ),
    },
)


# STEP-only terrain (single up-step), step_height = 0.1 + 0.5*difficulty so
# PARKOUR_PLAY_DIFF=0.6 -> 0.4m, 0.72 -> 0.46m (the teacher's mastered height),
# 1.0 -> 0.6m. Used to isolate/verify the teacher's single-step climbing.
STEP_04_TERRAINS_CFG = ParkourTerrainGeneratorCfg(
    size=(16.0, 4.0),
    border_width=20.0,
    num_rows=10,
    num_cols=40,
    horizontal_scale=0.08,
    vertical_scale=0.005,
    slope_threshold=1.5,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "parkour_step": ExtremeParkourStepTerrainCfg(
                        proportion=1.0,
                        apply_roughness=True,
                        x_range=(0.3, 1.5),
                        half_valid_width=(0.5, 1.0),
                        step_height='0.1 + 0.5*difficulty',
                        ),
    },
)


# TASKD_PIT matched to the REAL Task D deployment geometry (env_cfg.py): pit_width 1.35m
# (NOT 0.8), the box stepping-stone is 0.8m long (x) with top at -0.4 (0.6m box in 1.0m
# pit). The wider real pit (1.35) means the 0.8m box leaves a ~0.5m open gap on one side,
# which the narrower TASKD_PIT (0.8 pit / 0.5 stone) did NOT capture. Training on this
# matched geometry should transfer to deployment where the prior TASKD_PIT did not.
TASKD_PIT_REAL_TERRAINS_CFG = ParkourTerrainGeneratorCfg(
    size=(16.0, 4.0),
    border_width=20.0,
    num_rows=10,
    num_cols=40,
    horizontal_scale=0.08,
    vertical_scale=0.005,
    slope_threshold=1.5,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "taskd_pit": ExtremeParkourTaskDPitTerrainCfg(
                        proportion=1.0,
                        apply_roughness=True,
                        pit_width=1.35,        # REAL Task D pit width (was 0.8)
                        pit_depth=1.0,
                        step_drop=0.4,         # 0.6m box in 1.0m pit -> top 0.4m below rim
                        stone_len=0.8,         # REAL box footprint along x (was 0.5)
                        stone_half_width=0.5,  # box 1.0m wide (y) -> half 0.5
                        ),
    },
)