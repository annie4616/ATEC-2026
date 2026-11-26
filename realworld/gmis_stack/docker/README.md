# `docker/` — the on-board runtime as an image

*(한국어: [README.ko.md](README.ko.md))*

What actually runs on the mini-PC, packaged. Same code as [`onboard/`](../onboard/); this
directory is how it gets onto the robot and how it stays the same afterwards.

## Why containerise a single machine we control

Because the rules lock the code, parameters and system configuration at inspection and
require the code to be escrowed — and "it worked on the machine we set up in August" cannot
be re-created from a git tag. An image digest can. Re-flashing the mini-PC then takes
minutes instead of a morning of `setup_minipc.sh`, and a backup unit runs the identical
bytes rather than a second install that is *probably* the same.

## Use

```bash
bash docker/build.sh                       # build; tagged with the short git rev
bash docker/run.sh doctor                  # what is installed, configured, assumed
bash docker/run.sh identity                # git rev + config/artifact digests, for inspection
bash docker/run.sh gate                    # the acceptance gate, inside the image
NIC=enp2s0 bash docker/run.sh estop        # FIRST, before anything can move
NIC=enp2s0 bash docker/run.sh motormap --nic enp2s0
NIC=enp2s0 bash docker/run.sh run --backend real --nic enp2s0 --with-collector
```

`build.sh` prints the image **digest**. Cite that in the escrow submission, not the tag: a
tag can be moved, a digest cannot.

## The flags, and what breaks without them

`run.sh` and `compose.yml` carry these. Each one, dropped, produces a failure that looks
like something else:

| flag | why |
|---|---|
| `--network host` | DDS is not a TCP client. CycloneDDS discovers the robot by multicast on `192.168.123.x`, and SocketCAN's `can0` is a network interface too. On a bridge network the robot is simply not there, and the symptom is "no LowState" with no error |
| `--ipc host` | perception runs in a child process; the default 64 MB `/dev/shm` is a classic opaque multiprocessing failure under load. Note that `--ipc host` uses the host's `/dev/shm` as-is, which makes `--shm-size` **inert** — it used to be set here alongside it and read as protection it never provided |
| `--cap-add SYS_NICE` | the control thread asks for `SCHED_FIFO`. Without it, it silently runs at normal priority — survivable, but every perception burst becomes jitter on a 50 Hz motor stream |
| `--ulimit rtprio=99` | the capability alone is not enough; the limit has to permit it. **On the host this stack could not take real-time priority at all; in the container it can** |
| `--device /dev/videoN` | the camera, when it is a UVC device rather than an SDK channel |

## What is in the image, and what is not

13 MB of source in 90 files — `onboard/` plus the seven `course_mission` modules the runtime
imports and `eunha_rl/epkg/rma.py`, which is loaded by path. 12 MB of that is the policy
weights. The **root** `.dockerignore` whitelists exactly that set.

That file has to sit at the context root, and until 2026-09-09 it did not: it lived in
`docker/`, where Docker never looks (BuildKit's only other candidate is
`docker/Dockerfile.dockerignore`). The whitelist was therefore inert, the context was the
whole 485 MB repository including 154 MB of `.git`, and `COPY onboard/` baked untracked
`__pycache__` into an image whose entire purpose is to have a digest that means something.
Measured after the move: 13.04 MB, 90 files, no `__pycache__`, no logs.

Not in the image: the simulator, the training trees, the repository README, the recorded
datasets. `check_docs.py` detects a runtime image and skips the checks that reach for a
checkout around them. (`doctor.py` has no such check and needs none — nothing it inspects
depends on the checkout.)

The build **fails** if `onboard/tools/offsim_gate.py` does not pass inside it. A deployment
image that cannot pass its own acceptance gate should not exist.

## Dependency pinning

`unitree_sdk2py` depends on `opencv-python` — unpinned, and the GUI build. Installing it
left two OpenCVs in the environment, `import cv2` resolved to whichever landed last, and the
version floated between builds. For code the rules lock at inspection, a transitive
dependency that changes version on its own is a defect: the image you tested is not the
image you ship. The build now force-reinstalls the pinned headless build and then **verifies
every pin, failing the build on drift**.

CycloneDDS at the version `unitree_sdk2py` requires is source-only — there is no 0.10.2
wheel, and pip's error ("Could not locate cyclonedds") describes the symptom, not the cause.
The C library is built first and `CYCLONEDDS_HOME` is set before pip runs. Getting that
order wrong is the most common way this install fails.

## Parity: is the container computing the same thing?

Containerising moves the stack onto a different Python, a different torch build and a
different BLAS. `onboard/tools/parity.py` fingerprints every quantity on the control path
from fixed inputs and compares:

```bash
python onboard/tools/parity.py --out host.json
LOGS=$PWD bash docker/run.sh parity --out /logs/container.json   # /logs -> /app/onboard/logs
python onboard/tools/parity.py --compare host.json container.json
```

Re-measured 2026-09-09, host (python 3.11.15, torch 2.7.0+cu128, numpy 1.26.0) against the
image (python 3.11.16, torch 2.5.1+cpu, numpy 2.1.3):

| | |
|---|---|
| policy weights | 6/6 sha256 identical |
| every fingerprinted quantity on the control path | **20/20 bit-identical** |
| worst difference | 0.0 |

That is stronger than the earlier measurement on this page (13/20 bit-identical, torch items
differing by up to 3.8e-6, worst case 76 nanometres of commanded TCP motion against a 100 mm
gripper opening) and it is not a code change — it is what happens once both sides pin BLAS to
a single thread, so a dot product is summed in the same order in both builds. Treat 20/20 as
the current reading of this machine pair, not as a guarantee: a different host BLAS can put
the torch items back to a few parts in 1e-6, which was already far below anything the
hardware can express.

**No GPU was used or needed.** The runtime is CPU-only by design, so equivalence is a
CPU-only question.

## Runbook: from a robot on a table to a run

**Read the time budget before the steps.** This runbook used to be laid out as a continuous
0:00→12:00 day, against "the 12 hours of equipment testing the organisers provide". That
figure is real but it is *pre-competition* time at a **separate testing venue, by online
reservation** (0907 guide §VII.8; Hong Kong's window was 10 Aug – 4 Sep 2026 with booking
closing 3 Sep). It is not time at the competition site, and a plan that reaches its first
course walk at 5:30 cannot be run in the window you actually get.

What you actually get on the day (0907 §VII.8, "Testing and Debugging of Standardized
Equipment During the Competition Period"):

| when | how long | what it is for |
|---|---|---|
| pre-competition | by reservation, separate venue | forfeited entirely if not booked in the window |
| **check-in day** | **2 hours** | equipment verification, installing team-owned equipment, algorithm deployment, testing and debugging. Advance booking required |
| **competition day** | **≥ 1 hour** before your slot | verification, installing team equipment, power-on, program deployment, testing |

Then the run itself is 5 minutes of preparation and 30 minutes of task, and the clock never
stops (§IV.3).

Two steps below are **gates** — everything after them is meaningless if they are wrong, and
one of them fails silently. The 2-hour column is what the whole check-in day has to fit
into, so the order matters more than it used to.

Measured values go into a directory you mount over the image's configs, so a calibration
does not cost an image rebuild:

```bash
cp -r onboard/configs ./venue-configs      # once, at the start of the day
export CONFIGS=$PWD/venue-configs NIC=<iface>
```

### Host prerequisites (the mini-PC, before any of this)

Nothing below works without these, and none of them is installed by `setup_minipc.sh`:

```bash
docker --version                       # `run.sh` refuses early if this is missing
docker load < atec-onboard.tar.gz      # the image, carried in on a stick
bash ~/src/piper_sdk/can_activate.sh can0 1000000   # the arm's CAN interface
```

`can_activate.sh` ships with the PiPER SDK; clone it to `~/src/piper_sdk` on the mini-PC.
The container refuses to start a real-backend run while `can0` is down and says so in one
line — set `ARM=0` to run legs-only instead.

### Before the day (do all of it now, not on the day)

```bash
bash docker/build.sh                                  # tag + digest
bash docker/run.sh identity                           # git rev, config + artifact digests
python onboard/tools/parity.py --out host.json
LOGS=$PWD bash docker/run.sh parity --out /logs/container.json
python onboard/tools/parity.py --compare host.json container.json
docker save atec-onboard:latest | gzip > atec-onboard.tar.gz   # carry it on a stick
```

On the mini-PC: `docker load < atec-onboard.tar.gz`, then `bash docker/run.sh doctor`.
Everything except the robot-dependent rows should already be green. **Every minute spent
here is a minute you do not have on check-in day.**

### Check-in day — the 2 hours

#### 0:00–0:10 — Mount and E-Stop, nothing powered

Mount the mini-PC using **only the designated holes**, with an anti-detachment retainer and
strain relief on the network cable — the rules require the declared solution and inspect it.
An independent, inspected safety E-Stop link is the one permitted physical connection to
off-board equipment (§IV.4). Then, before anything can move:

```bash
NIC=$NIC bash docker/run.sh estop
```

**GATE 1. Do not skip this and do not do it later.** Triggering the E-Stop during a run
counts as a manual assistance (−10, three per run); an E-Stop that turns out not to work
costs the robot.

#### 0:10–0:25 — Link and the motor map, robot on blocks with the feet clear

Turn the robot's built-in high-level controller **off** (`L2+R2` on the controller while in
damping, or through the app). While `sport_mode` is running it fights every LowCmd this
stack sends, and the symptom is a robot that twitches and then does nothing.

```bash
sudo ip addr add 192.168.123.99/24 dev $NIC && sudo ip link set $NIC up
ping -c1 192.168.123.161
CONFIGS=$CONFIGS NIC=$NIC bash docker/run.sh motormap --nic $NIC --watch   # transport check
CONFIGS=$CONFIGS NIC=$NIC bash docker/run.sh motormap --nic $NIC           # the map itself
CONFIGS=$CONFIGS bash docker/run.sh doctor    # motor map should read `ok  measured: [...]`
```

`--watch` streams every motor it can see. If nothing appears the message type is wrong: set
`msg_type: hg` in `venue-configs/a2pro.yaml` and retry. Move one joint at a time when
prompted, then paste the printed yaml into `venue-configs/a2pro.yaml`.

**GATE 2, and the dangerous one.** A wrong map does not crash. The robot stands, passes a
smoke test, and folds the first time it turns. Do not let it walk until that row is `ok`.

Note that the `ip addr` line does **not** survive a reboot; re-run it after any power cycle.

#### 0:25–0:35 — First stand

Off the blocks, on the ground, with a person holding the E-Stop.

```bash
CONFIGS=$CONFIGS NIC=$NIC bash docker/run.sh prep
```

Expect `ZERO_TORQUE → RAMP` over two seconds → `STAND`, then it holds. If any joint drives
the wrong way, stop immediately — that is `joint_sign`, not a gain problem.

#### 0:35–0:55 — Camera

```bash
CONFIGS=$CONFIGS bash docker/run.sh calib --capture      # ~20 checkerboard views
CONFIGS=$CONFIGS bash docker/run.sh calib --intrinsics   # -> paste into sensors.yaml
CONFIGS=$CONFIGS bash docker/run.sh calib --extrinsics <image>   # board flat on the ground
```

These write to `onboard/logs/calib/`, which is the bind mount — so the views survive the
container and the next command can read them. (They did not until 2026-09-09: the mount was
at `/logs` while the tool wrote to `/app/onboard/logs`, so `--capture` filled a directory
that was deleted with the container and `--intrinsics` then found nothing.)

Target under **0.3°** of pitch error: 2° is 0.31 m of range error at 2 m, larger than the
whole grasp tolerance. Also settle **whether the camera returns depth at all** and set
`camera.depth.source` accordingly.

#### 0:55–1:05 — Arm

```bash
bash ~/src/piper_sdk/can_activate.sh can0 1000000
CONFIGS=$CONFIGS bash docker/run.sh doctor       # CAN interface should be `ok`
```

Then characterise the gripper: close on each object class and record width and current,
which is what `HoldDetector` thresholds on. The bands in `piperx.yaml` were fitted in
simulation.

#### 1:05–1:45 — Walk the course

```bash
CONFIGS=$CONFIGS NIC=$NIC bash docker/run.sh run --backend real --nic $NIC
```

Task 1 first, and alone. It is 80 points, it is the prerequisite for the other 160, and it
is the part with a measured record. Each run writes
`onboard/logs/<stamp>/{report.json,ledger.json,run.log}` to the host.

If — and only if — Task 1 is solid and time remains, add `--with-collector`. Read
`onboard/logs/<stamp>/ledger.json` after every attempt: it says which stage each attempt
died at, and `reobs_delta` over 300 mm means the robot walked to the wrong article.

#### 1:45–2:00 — Freeze

Copy the measured configs back over the image's own, rebuild once, and record the identity:

```bash
cp venue-configs/*.yaml onboard/configs/ && bash docker/build.sh
bash docker/run.sh identity
docker image inspect atec-onboard:latest --format '{{.Id}}'
```

The digest of that build is what ran and what should be cited. **A run made with a mounted
config override is not reproducible from the image alone** — `identity` prints a sha256 over
the *effective* configs precisely so that an override is a recorded fact rather than an
invisible one.

### Competition day — the ≥1 hour before your slot

Rule §IV.5: once inspection is complete the code, algorithm parameters and system
configuration are **locked**. No modifications, no parameter adjustment, no flashing, no
writing through a connected debugger — violation voids the run and can disqualify the team.
A backup unit must run the same code and configuration. So this hour is verification only:

```bash
docker load < atec-onboard.tar.gz      # only if this is a fresh/backup mini-PC
bash docker/run.sh identity            # must match what was escrowed
NIC=$NIC bash docker/run.sh estop      # the E-Stop, again
sudo ip addr add 192.168.123.99/24 dev $NIC && sudo ip link set $NIC up
CONFIGS=$CONFIGS NIC=$NIC bash docker/run.sh prep
```

Then, in the 5-minute preparation window before the whistle: start the run and let it wait
at the gate. **Any robot movement before the whistle invalidates the attempt, and three
accumulated false starts is disqualification** (§IV.3) — which is why `--gate manual` holds
until the controller combination, and why `--gate timer` is a rehearsal setting only.

### When it goes wrong

| symptom | first thing to check |
|---|---|
| no LowState at all | `sport_mode` still running; then `msg_type: hg` |
| stands, folds on the first turn | the motor map — go back to 0:10–0:25 |
| a joint drives the wrong way | `joint_sign` in `a2pro.yaml` |
| detections in the wrong place | camera pitch; then whether depth is real |
| gripper closes on nothing | that is the known 97 % failure — read the ledger, not the code |
| control loop late/overrun | `--cap-add SYS_NICE` and `--ulimit rtprio` missing from the run |
| `ConnectionError: ['can0' Interface __init__ ERROR]` | `can0` is down — `can_activate.sh`, or `ARM=0` for legs-only. The container checks this first and says so in one line |
| `crc_amd64.so: cannot open shared object file` | an image built before 2026-09-09. `unitree_sdk2py`'s `setup.py` never installed its prebuilt CRC libraries; the build now restores them and asserts they load |
| `NIC: set NIC ...` from a verb that used to work | pass `NIC=<iface>`; `run.sh` forwards it into the container (it did not before 2026-09-09) |
| ledger.json / report.json missing after a run | they are written under `onboard/logs/<stamp>/` on the host; a run with zero attempts still writes a ledger saying so |

## Still not run-ready

The image removes the *installation* problem. It does not remove the *measurement* problem —
`doctor.py` inside the container still reports the motor map, the message type, the camera
mount and the camera backend as **ASSUMED**, and the intrinsics and E-Stop line as
**UNCONFIGURED**. Those need the robot:

```
1. NIC=<iface> bash docker/run.sh estop              # before anything moves
2. NIC=<iface> bash docker/run.sh motormap --nic <iface>   # first, and do not skip it
3. bash docker/run.sh calib --capture
4. NIC=<iface> bash docker/run.sh prep
```

A wrong motor map does not crash. The robot stands, passes a smoke test, and folds the first
time it turns.
