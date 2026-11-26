"""Constants for Task D (obstacle traversal) demo collection with B2 + Piper.

Task D goal (see source/.../tasks/task_d/env_cfg.py and mdp/):
  - Robot starts at world x = -3 and must walk forward (+x) over a pit/platform,
    optionally nudging the box forward, until root x > 3.5 (success termination).
  - The walking expert is the pretrained B2 flat-locomotion PPO policy shipped at
    demo/policy.pt. We feed it a fixed forward velocity command and map its 12-D
    leg action onto the 20-D B2+Piper full-body action (arm held at default).

The obs/action layout mirrors demo/solution_rl.py exactly, because the same
policy + scaling must be reproduced here for the oracle to walk.
"""

# --- Action scaling (env side; matches ActionsCfg in envs_base_cfg.py) -------
# env_action = (joint_target - default_joint_pos) / ACTION_SCALE
ACTION_SCALE = 0.5

# --- Full-body joint layout for B2 + Piper (20 DoF) --------------------------
# Proprio joint ordering puts the 12 leg joints first, then the 8 arm joints,
# identical to demo/solution_rl.py (leg = 0..11, arm = 12..19).
LEG_ACTION_DIM = 12
ARM_ACTION_DIM = 8
FULL_ACTION_DIM = LEG_ACTION_DIM + ARM_ACTION_DIM

LEG_JOINT_INDICES = list(range(0, 12))
ARM_JOINT_INDICES = list(range(12, 20))

# --- Locomotion policy obs/action scaling (copied from demo/solution_rl.py) --
# Training-time leg action -> env-side leg action.
TRAIN_TO_ENV_ACTION_SCALE = [
    0.25, 0.5, 0.5,
    0.25, 0.5, 0.5,
    0.25, 0.5, 0.5,
    0.25, 0.5, 0.5,
]
# Env-side leg action -> training-time leg action (inverse of the above).
ENV_TO_TRAIN_ACTION_SCALE = [
    4.0, 2.0, 2.0,
    4.0, 2.0, 2.0,
    4.0, 2.0, 2.0,
    4.0, 2.0, 2.0,
]

# Fixed forward velocity command [vx, vy, wz] fed to the walking policy.
# Task D only needs straightforward locomotion in +x, so vy = wz = 0.
FORWARD_VELOCITY_COMMAND = [0.6, 0.0, 0.0]

# --- Success / episode bookkeeping -------------------------------------------
# Mirror of TaskDTerminationsCfg.x_reached (robot_x_greater_than, x_threshold=3.5).
SUCCESS_X_THRESHOLD = 3.5
# Cap an episode so a stuck/fallen robot does not spin forever. The push-into-pit
# maneuver (back up ~1.5 m, strafe ~1.6 m, then push the box ~2.3 m) needs a
# generous budget at the flat policy's modest speeds.
MAX_EPISODE_STEPS = 3000
# A few warm-up steps with the policy before recording (let the robot settle on
# the terrain after reset). Not recorded.
WARMUP_STEPS = 10
