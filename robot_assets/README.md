# robot_assets — Robot models

## a2pro_piperx

Platform change: **B2 → A2 Pro + Piper X**.

A 20-DoF articulation asset composed from URDF to USD. The parts that mattered were
removing the FixedJoint and the arm root API.

| Script | Purpose |
|---|---|
| `compose_a2_piperx.py` | Compose the two URDFs into one articulation |
| `verify_articulation.py` | Verify the joint structure |
| `check_arm_stow.py`, `sweep_arm_stow.py` | Find a stowed arm pose |
| `check_collision.py` | Self-collision check |
| `show_standing.py` | Visualise the standing pose |

## Joint mapping

**It is identity. Do not apply a permutation.** Doing so once produced a lateral
drift bug that took a while to track down.

Note: the actual USD/URDF asset files are excluded from this repository because of
their size.
