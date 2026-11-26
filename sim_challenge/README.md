# sim_challenge — Simulation Challenge (qualifiers)

The online qualifying round, built on IsaacLab. **10th of 93 teams**, advancing to the
real-world competition in Hong Kong.

| Directory | Contents |
|---|---|
| `atec_rl_lab_workspace/` | Official starter repo plus our extensions. Tasks A/D/E: environments, training, evaluation, submissions |
| `task_e/` | Task E (pick-and-place), self-contained. State machine + ACT |
| `taskd_final_controller/` | Task D locked final controller (reproduced 3/3) |

## Entry points

```
atec_rl_lab_workspace/scripts/
├── rsl_rl/             PPO training (train.py, play.py)
├── act/                Imitation learning (collect_demos_*, train_task_*, eval_*)
├── steering_head_wz/   Task A steering head, train/eval pairs
├── steering_ytrack/    Task A y-tracking and deployment
├── task_d/             Task D diagnostics, 21 diag_* probes
└── common/             Environment inspection and recording utilities
```

## Task D final solution

`taskd_final_controller/push_controller_cross_FINAL_LOCKED_box050_bail190_v230.py`

A six-state machine: `BACK_UP → ALIGN_Y → PUSH_Y → ALIGN_X → PUSH_X → CROSS`

The key insight: **pushing in +x jams the box against the platform wall, so it must
first be moved out of the platform band in −y.** Yaw is held at zero so that
body-frame velocity commands map directly onto world coordinates.

Locked values: `BOX_PUSH_DONE_X=0.5`, `PUSH_X_BAIL_RX=-1.90`, `speed=2.30`,
`y tolerance=0.18`

Three repeat runs all produced `max_x=+3.51 CROSSED, steps=1327` — fully deterministic.
