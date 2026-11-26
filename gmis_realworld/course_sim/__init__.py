"""course_sim — competition course TEST environment (mud/pebble/grass/stairs/slope/
mixed sections + red boundary strips + trash objects + 3 tapered bins)."""
# Best-effort registration, same reason as course_mission/__init__: terrain.py's course
# GEOMETRY (curve_map, SECTION_CENTERS, BIN_*, CHECKPOINTS) is plain numpy and the mission
# reads it on every step, but this import chain put gymnasium and the whole task registry
# in front of it — so on a machine without Isaac (i.e. the robot) nothing in course_sim
# could be read at all. Registration is for the simulator; geometry is for everyone.
try:
    import gymnasium as gym
except ModuleNotFoundError:
    gym = None

if gym is not None:
    # NOT inside the try above. Folding this import into the same except meant a
    # PYTHONPATH slip in a real simulator run — where gymnasium imports fine — silently
    # skipped all three registrations and surfaced much later as an unhelpful
    # "environment ATEC-Course-B2Piper doesn't exist". Absence of gymnasium is the
    # hardware case and is expected; absence of the task registry is a broken install.
    import atec_rl_lab.tasks.task_b  # noqa: F401  (base cfgs + asset roots register first)

_ENVS = (
    ("ATEC-Course-B2Piper", "course_sim.env_cfg:CourseEnvB2Cfg"),
    ("ATEC-Course-A2Piper", "course_sim.env_cfg:CourseEnvA2Cfg"),
    ("ATEC-Course-A2Bare", "course_sim.env_cfg:CourseEnvA2BareCfg"),
)

if gym is not None:
    for _id, _cfg in _ENVS:
        gym.register(
            id=_id,
            entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
            disable_env_checker=True,
            kwargs={"env_cfg_entry_point": _cfg},
        )
