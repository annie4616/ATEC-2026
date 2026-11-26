# realworld — Real-world preliminary stack

For the Hong Kong real-world round (A2 Pro + PiPER). This is the only one of our three
working areas that is a coherent, documented codebase.

It deliberately holds **two stacks in one tree**:

| Stack | Role | Dependencies |
|---|---|---|
| Simulator | Proves the algorithms are correct | IsaacLab + GPU |
| **Onboard** (`gmis_stack/onboard/`) | What **actually runs on the robot** | No Isaac, no CUDA |

The onboard target is a Beelink mini-PC: Intel N150, 4 E-cores, 12 GB, **no GPU**.

## Results

Task 1 **80/80**, Task 2 **120/160** (under honest navigation and object detection).

## Entry points

- `gmis_stack/onboard/run.py` — the mission entry point
  - `--backend mock` — desktop rehearsal
  - `--backend real --nic enp2s0` — the actual robot
  - The backend flag is the *only* difference between the two. The robot is shared
    equipment we could not touch until check-in day, so everything above the hardware
    boundary is byte-identical in both paths.
- `gmis_stack/onboard/tools/offsim_gate.py` — acceptance gate: no simulator, no GPU,
  no absolute paths
- `gmis_stack/deploy/run_mission.sh` — full venue mission with swappable policies

## Do not remove the braces in run_mission.sh

The entire script body is wrapped in a single `{ }` group. This is deliberate.

Bash reads a script lazily by byte offset. Editing the file while it was running
shifted that offset, and the wrapper resumed mid-stream and **launched a second
unplanned mission** (incident `0903_2222_svO`). The braces force bash to parse the
whole file before executing any of it.

## Mini-PC bottleneck

See `gmis_stack/GRASP_PERCEPTION_SPEEDUP_PLAN.md`.

The grasp policy is innocent — a 42→7 MLP running in microseconds. **Perception is the
culprit.** Hotspots: `perception.py:1130 _blobs` (connected components in pure Python
with union-find, holding the GIL) and `perception.py:208 label_map` (230k pixels ×
prototypes, full-frame float).

Detection rate went from 1.2 Hz to 5 Hz.
