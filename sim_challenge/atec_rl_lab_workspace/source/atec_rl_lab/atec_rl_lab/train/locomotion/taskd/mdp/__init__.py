"""MDP terms for the Task D RL environment.

Reuses all locomotion-velocity MDP terms (so the walking-quality rewards,
observations, events and curricula remain available) and adds Task D specific
forward-progress / goal terms on top.
"""

from atec_rl_lab.train.locomotion.velocity.mdp import *  # noqa: F401, F403

from .rewards import *  # noqa: F401, F403
from .terminations import *  # noqa: F401, F403
