"""Course mission package: full pick-classify-deposit run on the competition course."""

# gymnasium is a SIMULATOR dependency, not a mission one. Importing it unguarded here
# meant the detector, the object catalogue and the deployment model — none of which touch
# gym — could not be imported on a machine without Isaac, which is every robot. The
# detector is the single most portable thing in the tree; do not let this line own it.
try:
    import gymnasium as gym
except ModuleNotFoundError:                # hardware build: nothing to register
    gym = None

# The env config needs isaaclab.sim, which only imports inside a running SimulationApp.
# The detector and the object catalogue do not, and scripts/tune_detector.py scores the
# detector on recorded frames with no simulator at all — so registration is best-effort
# and its absence is only an error for code that actually asks for the env.
try:
    if gym is None:
        raise ModuleNotFoundError("gymnasium")
    from .env_cfg import CourseMissionEnvCfg

    gym.register(
        id="ATEC-Course-Mission",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": CourseMissionEnvCfg},
    )
except ModuleNotFoundError:
    pass
