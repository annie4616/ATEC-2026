"""Real-deployment model: sensor noise and on-board compute limits.

Split out of mission.py with no Isaac imports so both the simulator-side drivers and
the offline detector tooling can share one definition of "what the robot actually
sees and how fast it can think".
"""

from __future__ import annotations

import os

import torch

# ---- real-deployment model -----------------------------------------------------------
# Target hardware: ASUS NUC14MNK15 (Intel N150, 4 E-cores, 6 W) bolted to the robot.
# MISSION_REAL_NOISE=0 turns the sensor noise off, MISSION_COMPUTE=ideal removes the
# compute limits; both are ON by default because that is what the robot will actually see.
REAL_NOISE = os.environ.get("MISSION_REAL_NOISE", "1") == "1"
COMPUTE = os.environ.get("MISSION_COMPUTE", "n150")

# Proprioceptive noise EXACTLY as the walk/crouch policies were trained with
# (task_base/envs_base_cfg.py ProprioObservationsCfg). IsaacLab adds noise to the raw
# term and scales afterwards, so these are pre-scaling magnitudes.
LEG_NOISE = {"ang_vel": 0.2, "gravity": 0.05, "joint_pos": 0.01, "joint_vel": 1.5}
# Grasp-policy noise as trained under GRASP_REAL_NOISE=1 (grasp_env_cfg.py)
GRASP_NOISE = {"ee_pose": 0.004, "width": 0.002, "base_state": 0.02, "axes": 0.01}
# Camera: small global-shutter RGB-D module — photon/read noise on colour, and a depth
# error that grows with range (structured light / stereo both do).
CAM_RGB_SIGMA = 4.0        # LSB out of 255
CAM_DEPTH_REL = 0.008      # 0.8 % of range, 1-sigma
CAM_DEPTH_ABS = 0.004      # metres, 1-sigma floor
# N150 budget. The two loops on this machine cost very different amounts:
#   perception — a 230k-pixel unprojection + HSV + blob labelling in numpy is ~20 ms on a
#     desktop core and 4-6x that on an N150 E-core, so ~5 Hz is what it can sustain;
#   locomotion/grasp — a 3-layer MLP is well under 1 ms even here, so the 50 Hz loop is
#     dominated by the sensor read and CAN/EtherCAT round trip, i.e. one control period.
# action_delay_steps is in 20 ms control steps and is the honest knob for that round trip;
# MISSION_ACTION_DELAY overrides it, which is how scripts/run_latency_sweep.sh measures
# how much the crouch policy actually tolerates (see README: it falls over at 2 steps).
COMPUTE_LIMITS = {
    "n150":  {"detect_hz": 5.0, "action_delay_steps": 1},
    "ideal": {"detect_hz": 25.0, "action_delay_steps": 0},
}


def _u(shape, amp, device):
    """Uniform noise in [-amp, amp] (IsaacLab's AdditiveUniformNoise)."""
    return (torch.rand(shape, device=device) * 2.0 - 1.0) * amp


# rgbl perception-noise model (identical to graspenv/mdp/observations.py)
PN_BIAS = (0.012, 0.012, 0.010)
PN_JITTER = 0.004
PN_DIM_REL = 0.12
PN_HOLD = 5


# ======================================================================================
# course geometry helpers
# ======================================================================================
