# ATEC 2026 Robotics Challenge

Source code and documentation for our work on the ATEC 2026 Simulation Challenge
and the real-world preliminary round.

**Result: 10th of 93 teams** in the online qualifiers, advancing to the real-world
competition in Hong Kong.

---

## About this repository

This is the **source-code release**. Large artifacts (videos, checkpoints, datasets,
3D assets) are excluded because of GitHub's 100 MB file limit.

```
sim_challenge/      Simulation challenge (qualifiers)
realworld/          Real-world preliminary stack (simulator + onboard mini-PC)
parkour_training/   Go2 parkour teacher/student training
robot_assets/       A2 Pro + Piper X robot model
```

Each directory has its own README. `README_server_structure.md` documents the
full working tree as it exists on our server, including the parts not published here.

### Task index

Directories are organised by **working context, not by task**. To find a task:

| Task | Code | Data / results (server only) |
|---|---|---|
| **Task A**<br>Off-road Navigation | `sim_challenge/atec_rl_lab_workspace/scripts/`<br>　`steering_head_wz/` steering head train/eval<br>　`steering_ytrack/` y-tracking, deployment<br>　`act/collect_demos_task_a.py` | `taska_exp/` 552 M experiments<br>`taska_pdheads/` 861 M PD iterations<br>`task_a_real_exp/` 107 M A2 prep<br>`3_taska_videos/` 680 M, 178 clips |
| **Task D**<br>Obstacle Traversal | `sim_challenge/atec_rl_lab_workspace/scripts/`<br>　`task_d/` 21 diagnostic probes<br>　`act/` collect, train, eval, experiments<br>`sim_challenge/taskd_final_controller/` ★ locked final | `4_taskd_sweeps/` 580 M sweeps<br>`datasets/atec_task_d_*` |
| **Task E**<br>Tabletop Manipulation | **`sim_challenge/task_e/`**<br>　`collector.py`, `state_machine.py`<br>　`solution_taskE.py`, 5 object configs<br>　`scripts/act/*_task_e.py` | `task_e/` 26 G<br>　`trajectory{,_filtered,_strict}.hdf5` 25 G<br>　`checkpoints/` 255 M |
| **Real-world**<br>(Tasks 1 & 2) | `realworld/gmis_stack/`<br>　`onboard/` mini-PC deployment<br>　`course_mission/` mission FSM | Task 1 **80/80**, Task 2 **120/160** |
| **parkour**<br>(shared by A & D) | `parkour_training/parkour_{depth,lidar}/` | `logs/rsl_rl/unitree_go2_parkour/` 68 runs |

> **Note on Task E:** the three HDF5 files (25 G) are three filtering stages of the
> same data. Since `collector.py` and `filter_demos*.py` are preserved, `_filtered`
> and `_strict` (15.9 G) can be regenerated from `trajectory.hdf5`.

---

## Competition overview

ATEC 2026 Simulation Challenge — a quadruped robotics competition built on IsaacLab.

Official tasks: Off-road Navigation (A), Tabletop Manipulation (E),
Garbage Collection (B), Obstacle Traversal (D).
We worked on **A, D and E**.

---

## Runtime environment

> ⚠️ **2026-11-26: all three conda environments were deleted** (112 GB reclaimed),
> as the competition had ended and no reruns were planned. Rebuilding is required
> to run any of this code.
>
> | env | size | purpose |
> |---|---|---|
> | `env_isaaclab` | 67 G | ATEC2026 main (Tasks A/D/E), IsaacLab 3.1.2 |
> | `parkour_il` | 38 G | parkour teacher/student training, rsl-rl-lib 2.3.3 |
> | `parkour` | 7 G | superseded |
>
> Rebuild hints: `IsaacLab/environment.yml`, each `submission_*/requirements.txt`,
> and `realworld/gmis_stack/docker/Dockerfile`. The absolute interpreter paths
> hardcoded in the runner `.sh` scripts must be updated to match.

The record below describes the setup as it was.

### Two separate environments

They were kept **strictly separate** — mixing them breaks on API drift.

| env | purpose |
|---|---|
| `env_isaaclab` | ATEC2026 main (Tasks A/D/E) |
| `parkour_il` | parkour training **only here** |

Scripts invoked interpreters by absolute path rather than activating:
`/home/sohyun/.conda/envs/env_isaaclab/bin/python`

### Required environment variables

```bash
export PYTHONPATH=/hdd0/sohyun            # resolves atec_2026_tesk_e.* imports
export TMPDIR=/hdd0/sohyun/tmp            # default /tmp/isaaclab is owned by another user
export CUDA_VISIBLE_DEVICES=2             # varies by job, see below
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nonexistent_x   # MPS bypass required on this host
```

### GPU assignment

| Job | GPU | Reason |
|---|---|---|
| IsaacLab, general | 7 | |
| Task D ACT collect/train/eval | 2 | avoids render contention on GPU 3 |
| parkour teacher playback | 2 | |

---

## Results by task

### Task A — Off-road Navigation (Hiking)

**Problem:** follow a red 40 cm boundary line along a course. The line exists in the
real venue but not in simulation.

**How the approach evolved:**

1. The walking policy itself converged (speed ceiling vx 2.63)
2. The real bottleneck turned out to be **steering**, not locomotion
3. Frozen DINOv2 CLS + MLP → wz head ("PD8")
4. LiDAR rail detection → rule-based parallel following
5. Depth encoder restored (latent 0.0257, outperforming LiDAR)

**Measured results** (`taska_exp/PRETRAINED_NAV_SURVEY.md`, 2026-07-31):

| Condition | Completion |
|---|---|
| Straight baseline (PD8 + kd 0.2) | 8/10 (80%) |
| Widened course (5, 6 m) | 0/5 |
| Curved path | 2/5 |
| Right-angle turn | 1/5 |

→ **Clear generalization failure.** Overfit to simulation.

The deployment bundle (`task_a_real_exp/PD8_deploy_for_colleague/`) is eval-legal —
it uses no privileged information.

### Task D — Obstacle Traversal

**Problem:** push a box, cross a pit, climb a 0.4 m step.

**Final solution:** hardcoded push + parkour teacher climb, switching between two
policies at runtime.

- PUSH stage: +14 points, successful
- CLIMB stage: the teacher failed out of distribution

**Locked configuration:** box050 / bail190 / v230 — 3/3 CROSSED (max_x 3.51).
All four active controller files verified byte-identical.

**Memory-replay approach:** replaying 88 memorised demos. Both open-loop and
closed-loop oracles missed the box because of a server start-state mismatch, stalling
at +2. **Final score: 16.**

### Task E — Tabletop Manipulation

Self-contained in `sim_challenge/task_e/`. State machine + ACT.
Per-object configs (`config_sugar.py`, `config_my_banana.py`, `config_my_mustard.py`).
Data pipeline in three stages: `trajectory.hdf5` → `_filtered` → `_strict`.

---

## parkour training

Upstream: [CAI23sbP/Isaaclab_Parkour](https://github.com/CAI23sbP/Isaaclab_Parkour)
(an IsaacLab port of Extreme-Parkour). Both `parkour_isaaclab/` and `parkour_tasks/`
are installed with `pip install -e`.

### Three phases across 68 runs

| Period | Runs | Character |
|---|---|---|
| 06-24 – 06-29 | 40 | timestamps only; early iteration |
| 08-05 | 4 | |
| 08-16 – 09-04 | 24 | descriptive suffixes; naming discipline established |

### Key runs

| Run | Meaning |
|---|---|
| `2026-06-24_10-58-09_학습성공_0.45m` | 0.45 m step climb succeeded (the Korean suffix reads "training succeeded") |
| `2026-08-05_01-38-20` | **parkour teacher** (RMA, no distillation), `model_4200.pt` |
| `2026-08-28_13-13-12_teacher_overspeed` | teacher with overspeed penalty |
| `2026-09-01_08-37-52_encoder_depth` | depth encoder (adopted) |
| `2026-09-03_07-16-35_encoder_lidar_overspeed` | LiDAR encoder + overspeed |
| `2026-09-04_06-50-41_encoder_lidar_front` | front mount (offset 0.338/0/0.081) |

Suffix vocabulary: `teacher_*`/`student_*` (distillation), `encoder_*` (perception
encoder line), `sanity_*` (checks preceding each encoder run), `_ft` (fine-tune
continuation), `_smoke` (smoke test).

### Teacher observation spec (753-dim)

```
proprio 53 + scandot 132 + priv_explicit 9 + priv_latent 29 (ignored) + history 530
```

The scandot grid is 12×11. It is **not a real sensor** — it must be synthesised by an
adapter at submission time.

---

## Robot platform change: B2 → A2 Pro + Piper X

`robot_assets/a2pro_piperx/` — a 20-DoF articulation asset composed from URDF to USD.
The key steps were removing the FixedJoint and the arm root API.

Scripts: `compose_a2_piperx.py`, `verify_articulation.py`, `check_arm_stow.py`,
`sweep_arm_stow.py`, `check_collision.py`, `show_standing.py`.

**Joint mapping is identity. Do not apply a permutation** — doing so caused a lateral
drift bug in the past.

---

## Pitfalls

1. **Two git repositories lost their contents on 2026-11-13.**
   Both `ATEC2026_Simulation_Challenge/.git` and the real-world stack's `.git` were
   reduced to empty directory skeletons (no HEAD, config or objects; `du` = 0 bytes),
   timestamped within minutes of each other. **History, remote and branch names are
   unrecoverable**; the code itself is intact. Only the names `perf` (branch) and
   `origin` (remote) survived, recovered from leftover path names.
2. **`yewon_1/` was byte-identical to `yewon/task_d_demo/`** — a copy made 27 minutes
   later. Removed.
3. **`Isaaclab_Parkour_lidar/logs/` is empty.** The LiDAR results live in the main
   parkour tree. Its README is an unmodified copy and says nothing about LiDAR.
4. **`IsaacLab/` is only 61 MB** — far smaller than a normal checkout, so it is likely
   a partial or asset-stripped copy. Verify before relying on it.
5. **Unset TMPDIR causes PermissionError** — the default `/tmp/isaaclab` is owned by
   another user.
6. **At most 3 processes per GPU.** Four or more is not safe; limit `OMP_NUM_THREADS`
   alongside.
7. **Use at most half the CPUs** — `taskset -c 0-191` on the 384-core host.
8. **Copy before making large edits.** Repeatedly editing `course_student_a2.py`
   in place made a working 8/28 configuration impossible to reproduce.
9. **Never reuse an `exp_name`** — ACT training overwrites results.
10. **`Isaaclab_Parkour` was vendored in six places**, differing in only six files
    between the depth and LiDAR variants. See `parkour_training/README.md`.

---

## Disk cleanup, 2026-11-26

Reduced the working tree from **191 GB to 74 GB** (117 GB reclaimed), then removed the
three conda environments for a further 112 GB — **229 GB in total**.

| Category | Size |
|---|---|
| npz caches and rollout logs | 67.5 GB |
| intermediate checkpoints | 37.8 GB |
| duplicate HDF5 | 9.4 GB |
| training rollout videos | 1.9 GB |

Preserved: every `best_loss.pt` and `final.pt`, all 38 `.onnx` exports, all four
submission bundles, the Task D locked controller, and the 88-demo RGB-D dataset.

Full file lists, the cleanup script and the reasoning behind each decision are kept
on the server under `_cleanup_records/`.
