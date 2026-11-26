# `onboard/` — A2 Pro + PiPER-X, on the mini-PC

*(한국어: [README.ko.md](README.ko.md))*

The deployable stack: what actually runs on the robot's Beelink (Intel N150, 4 E-cores,
12 GB, no GPU). Self-contained — `rsync onboard/ robot:` plus the two sibling packages it
reuses, and nothing here needs Isaac Sim, IsaacLab, gymnasium or CUDA.

The rest of this repository proves the *algorithms* work (Task 1 80/80, Task 2 120/160 under
honest navigation and object detection). None of it runs on a robot: every actuator write
goes through `robot.set_joint_position_target` into PhysX, every state comes from
`env.step()`, and there is no wall-clock scheduler anywhere. This directory is the part that
does run on a robot.

## Quick start

```bash
# 1. Acceptance gate — no simulator, no GPU, no absolute paths
python onboard/tools/offsim_gate.py

# 2. Whole stack against a mock robot, on this machine
python -m onboard.run --backend mock --gate timer --minutes 1

# 3. Safety contract
for f in lowstate_stall inverted estop joint_fault; do
  python onboard/tools/replay_run.py --inject $f --quiet; done

# 4. Corridor following, closed loop
python onboard/tools/check_nav.py

# 5. Task 2: score the tracker offline, then run the collector on recorded frames
python onboard/tools/check_collector.py --frames <dataset_hk>
python -m onboard.run --backend mock --with-collector --frames <dataset_hk> --minutes 2

# 6. On the mini-PC: install, then find out what is still missing
bash onboard/bin/setup_minipc.sh --check     # report only
bash onboard/bin/setup_minipc.sh             # install
python onboard/tools/doctor.py               # readiness

# 7. Arm alone / legs alone (separate buses, so each tests independently)
python -m onboard.tools.arm_bench                    # mock
python -m onboard.tools.arm_bench --backend real     # CAN only: link, enable, feedback
ARM=0 NIC=enp2s0 bash onboard/bin/run.sh             # legs only, no arm

# 8. On the robot
NIC=enp2s0 bash onboard/bin/run.sh
```

## The arm and the body attach to the mini-PC separately

There is no data link between them. The arm sits on the body **mechanically only**
(`kinematics.mount_xyz`); the frame transform is software, inside the mini-PC.

| | Body (A2 Pro, 12 joints) | Arm (PiPER-X, 6 joints + gripper) |
|---|---|---|
| Transport | ethernet / DDS | CAN 1 Mbps |
| Interface | a NIC, e.g. `enp2s0` | `can0` |
| Driver | `oio/a2_dds.py` | `oio/piper_can.py` |
| Config | `configs/a2pro.yaml` | `configs/piperx.yaml` |
| Rate | 500 Hz republish thread | 50 Hz commands |

The two drivers do not import each other; they meet in exactly one place, `run.py`'s
`build()`. So either side can be tested with only its own cable plugged in, and it **should**
be: a check that cannot separate "both are down" from "one is down" spends the five-minute
preparation window finding out which.

* **Legs only**: `ARM=0 bash onboard/bin/run.sh` (the existing `--no-arm`).
* **Arm only**: `onboard/tools/arm_bench.py`. `onboard.run` refuses to start without
  LowState — prep's first stage is `legs.wait_for_state` — so with only the arm connected
  there is no way in through `run.py`. Hence a separate entry point.

  It is **read-only by default**: link, state rate, joint limits and gripper feedback, with
  nothing commanded. The first question on check-in day is whether the CAN link is real, not
  whether the arm reaches, and answering the first must not require clearing the space around
  a machine nobody has commanded yet. Motion is opt-in: `--move` (home ramp, then one joint
  at a time) and `--grip` (open, then close).

## Layout

| | |
|---|---|
| `oio/` | the hardware boundary. Legs over DDS, arm over CAN, sensors, remote, E-Stop, and a mock of every one of them |
| `rt/` | drift-free 50 Hz cadence, age-stamped mailboxes |
| `control/` | RMA leg policy, arm IK, limiters, the safety state machine |
| `estimate/` | attitude, leg odometry, the map-free corridor filter |
| `perceive/` | camera rectification, depth synthesis, LiDAR rails, the perception process |
| `mission/` | hiking FSM, wall-clock budget |
| `bringup/` | five-minute prep, start gate, go/no-go self-test |
| `tools/` | the venue tools and every check listed above |
| `artifacts/` | the five policy files plus a sha256 manifest |

Three rules hold the design together, enforced by `tools/offsim_gate.py`:

1. Nothing under `onboard/` imports a simulator.
2. No absolute paths.
3. Hardware access never leaves `oio/` — so everything above it runs against mocks.

Rule 3 is the important one. The A2 Pro is shared equipment we do not touch until check-in
day; if the only way to run the control loop, the state machine and the mission is to have
the robot, then all of it arrives untested on the single day there is no time to debug.

## Reused, not rewritten

`course_mission/{mathx,navfuse,armkin,perception,catalogue,deploy}.py` and
`eunha_rl/epkg/rma.py` are imported as-is — they are already Isaac-free and already
validated. `eunha_rl/epkg/__init__.py` imports gymnasium, so `rma.py` is loaded by path
(`vendor/loadmod.py`), the same trick `scripts/check_mathx.py` uses. IsaacLab's
`DifferentialIKController` is copied verbatim into `vendor/diffik.py`: it is pure torch
(123 µs per solve on a CPU core) and it is the exact controller the grasp policy was trained
through.

## The hardware contract

Cross-checked against `a2pkg/rough_env_cfg.py`, `epkg/rma.py` and `course_mission/mission.py`,
and stated once in `configs/a2pro.yaml` rather than as constants in code.

```
obs45 = [ w_b*0.25 (3) | g_b (3) | cmd (3) | q-q_def (12) | qd*0.05 (12) | a_{t-1} (12) ]
history 50 frames (oldest first) -> AdaptationNet (ReLU) -> latent 8
actor([obs45, latent]) : 53 -> 512 -> 256 -> 128 -> 12   (ELU)
q_target = q_def + [0.125 hip | 0.25 thigh, calf] * a          at 50 Hz, kp 120, kd 4
joint order = FR, FL, RR, RL x (hip, thigh, calf)
q_def = [-0.1, 0.8, -1.5,  +0.1, 0.8, -1.5,  -0.1, 1.0, -1.5,  +0.1, 1.0, -1.5]
```

Four things here are easy to get wrong and expensive to get wrong:

* `a_{t-1}` is the policy's **raw** output. The simulator reaches the joint target in two
  hops (`a * scale/0.5`, then the action term's `default + 0.5*a`); on hardware they collapse
  into the one line above. Applying the 0.5 as well halves the gait.
* The adaptation net's activation is **not** in the checkpoint. It must be ReLU — ELU
  collapses velocity tracking on turns (measured 0/160).
* The gripper is two mirrored prismatic joints in simulation, so a sim "width" is one
  finger and the real opening is **twice** it. One conversion, in `piper_can.py`.
* Anything driving the trained grasp policy must use
  `armkin.jacobian_training_convention()`, which reproduces an IsaacLab frame quirk on
  purpose. Using the correct Jacobian scored 0/5 against a 2–3/6 baseline.

## Sensors

One forward camera plus IMU and encoders is the whole perception input. Every tuned constant
in `perception.py` was fitted at 640×360, fx = fy = 488.67, centre principal point, mounted
at (0.3381, 0.0336, 0.0525) and pitched 22° down. So `perceive/camadapter.py` rectifies
whatever the real lens is **into that virtual pinhole**, and `perception.py` is not touched.
`tools/check_camadapter.py` proves the hardware pose path equals `perception.cam_pose_body`
to 2e-15 m.

### The open hardware questions, and what each costs

| | if it goes the wrong way |
|---|---|
| Does the built-in camera return depth? | Task 1 unaffected (`detect_belt` uses no depth). Task 2 is hit hard — see below |
| What is the real K, distortion and FOV? | `camadapter` remaps; needs a checkerboard calibration |
| Is the mount pose really 22° down, and is video/pose sync tight? | 2° of pitch error is 0.31 m of range error at 2 m |
| Is the LiDAR reachable through the SDK? | navigation falls back to camera-only, which is the validated path |

**Measured, on 80 real recorded course frames**: if the depth channel is absent and object
detection has to run on synthesised depth. Reproduce with
`python onboard/tools/bench.py --frames <dataset_hk> --limit 80 --depth-study`.

| depth source | detections retained |
|---|---|
| true depth (control) | 100 % |
| ground plane only | 32 % |
| LiDAR, 16 lines | 49 % |
| LiDAR, 32 lines | 57 % |
| LiDAR, 64 lines | 76 % |
| LiDAR, 128 lines | 84 % |
| LiDAR + ground-plane hole fill | **19 %** |

The last row is the one worth remembering. Filling the LiDAR's holes with the ground plane
sounds strictly better and is more than twice as bad as leaving them empty:
`detect_arrays` discriminates on *height above ground*, so a plane-filled pixel is by
construction at zero height and flattens the object being looked for. A NaN is honest; a
plane sample is a confident wrong answer. `depthsynth.synth()` therefore does not fill.

## Timing

Measured on one desktop core, single-threaded, with the N150's 4–6× E-core penalty
applied. Medians of nine blocks of 400 on a *shared* machine, so treat the absolute
numbers as ±10 % — the ratio between them, and the N150 column, are what matter:

| | this core | N150 |
|---|---|---|
| leg control step (obs + adaptation + actor) | 146 µs | 0.6–0.9 ms of the 20 ms budget |
| `detect_belt` | 5.3 ms | 21–32 ms |
| `detect_bins` | 7.1 ms | 29–43 ms |
| `detect_arrays` | 96 ms | 384–576 ms → **1.7–2.6 Hz** |
| ground-plane depth | 1.9 ms | 8–11 ms |

Navigation perception is comfortably real-time; object detection is what eats the budget,
which is why `mission_hk.yaml` runs it at 2 Hz and belt detection at 10 Hz.

**Perception runs in its own process, not a thread.** As a thread it looked fine — the loop
held 50.00 Hz — until a single 213 ms control tick appeared, with the tick's own work
measuring 0.25 ms. The time went to not being scheduled: the detector's connected-components
and blob routines are pure-Python loops that hold the GIL for as long as they run. In a
thread, a 200 ms detector frame is a 200 ms hole in the motor command stream. With the
detector in a child process the worst observed tick is 32 ms.

## Safety

`ZERO_TORQUE → RAMP → STAND → RUN → (DAMP | FAULT)`, with `RECOVER` off `RUN`.

* DAMP and FAULT are **sticky**. Ordinary mission logic — budget expired, run finished — may
  request STAND, and that request is refused from DAMP. Leaving DAMP is an explicit re-arm
  through RAMP; leaving FAULT additionally needs a human, which is exactly the accounting
  the rules use for an assisted reset.
* Damping is a state, not a packet. A 500 Hz writer thread republishes the latched command,
  so no single dropped DDS frame can leave the robot holding a mid-stride position target.
* Aborts: stale LowState (>100 ms), body past the orientation limit, controller abort
  button, hardware E-Stop line. All four verified by injection in `tools/replay_run.py`.
* Self-righting is a **scoring feature**: a human touching the robot is −10, a robot getting
  up on its own is 0.
* Rate and joint-limit clamps on every outgoing target, absent from both reference stacks.
  The leg cap is derived per joint from the velocity rating (`auto` in `a2pro.yaml`), not a
  flat number: measured over 2399 steps of the real policy the per-step target change is
  p50 0.002, p99 0.033, max 0.43 rad, so a flat 0.2 cap — the value the arm needs — was
  reshaping the gait at exactly the transients the policy uses to catch itself.

## Before the whistle

The rules give five minutes, once. `bringup/prep.py` runs
`link → sensors → motors → stand → estimator origin → self-test → READY` and times each
stage; it refuses to proceed on a FAIL. Perception is started **before** the control loop
exists, because `spawn` costs 1.2 s and that would otherwise be 1.2 s of stale commands.

`bringup/startgate.py` latches the run and is irreversible. Nothing commands a velocity until
it fires — a false start voids the attempt and three mean disqualification, so that is
structural rather than a matter of care.

## First hour at the venue

1. `bash onboard/bin/estop_test.sh` — before anything else moves.
2. `python onboard/tools/dump_motor_map.py --nic <iface>` — **do not skip this.** The policy's
   joint order is character-for-character the Unitree SDK order used on Go2, so the map is
   very probably the identity. A2 Pro is newer than anything either reference stack covers,
   and a wrong map does not crash: the robot stands, passes a smoke test, and folds the first
   time it tries to turn.
3. `python -m onboard.tools.arm_bench --backend real` — the arm's CAN link. Independent of
   the legs, so it does not wait its turn, and a failure here is the piper_sdk CAN bring-up,
   not the robot. Once there is space around it, `--move --grip` for real motion.
4. `python onboard/tools/calib_extrinsics.py` — checkerboard, target < 0.3° of pitch error.
5. `NIC=... bash onboard/bin/prep.sh` — time the five-minute sequence for real.

## Task 2: the collector

`mission/collector.py` fills the `attach_collector()` seat: pick a target, walk to it, grasp
it, deliver it. Run it with `--with-collector`. Object identity comes from
`perceive/objtrack.py` rather than a simulator registry, grasp success from finger width,
current and lift rather than the object's true pose, and the bins from where the robot last
*saw* them, carried on raw odometry (`estimate/binmem.py`).

**Read this before expecting points.** Measured over 18 simulator runs under honest
navigation *and* honest object detection — the deployment condition — the pipeline scored:

| | |
|---|---|
| grasp attempts | 265 |
| grasps | 5 (1.9 %), **all cans** |
| into the correct bin | 1 |
| Task 2 best | 20/160, with 17 of 18 runs at zero |

The class pattern is not bad luck. The shipping grasp checkpoint was fine-tuned on objects
spanning 45–70 × 45–130 × 30–120 mm; only the can (66 × 66 × 115) is inside that box, and
rind (200 × 50 × 20) and lunchbox (150 × 110 × 60) are not. Their only path is the scripted
pinch. Separately, a third of attempts aimed more than a metre from the article — the aim
error is bimodal, 68 % under 100 mm and 32 % over a metre with nothing between, which is the
signature of targeting the wrong object rather than of aiming imprecisely at the right one.

So half of this module is `mission/ledger.py`, which records for every attempt what it had to
work with and where it died — track views, view spread, bank drift, the TCP residual at
closure, finger width, current and lift, the bin sighting's age and travel. The field that
carries the most is `reobs_delta`: the disagreement between the estimate the robot approached
on and the one it re-measures from the crouch. Ground truth is unavailable at a venue;
disagreement between two independent looks is not.

`tools/check_collector.py --frames <dir>` scores the tracker offline against recorded frames,
separating *wrong class* (a detection with no article of that class within a metre — the
detector's precision is 45–59 %) from *localisation error* among the rest. Those want
opposite fixes, and one number for both is how a precision problem gets mistaken for a
tracker problem.
