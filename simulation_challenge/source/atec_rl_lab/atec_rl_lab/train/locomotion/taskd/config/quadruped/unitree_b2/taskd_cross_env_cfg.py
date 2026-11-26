"""Hierarchical Task D RL: CROSSING ONLY (box already seated in the pit).

The box-pushing oracle (scripts/act/task_d/push_controller.py) already seats the
box in the pit with 100% success. So instead of asking RL to learn pushing AND
crossing (the integrated env stalled — see RL_POLICY_DESIGN.md / memory), this env
STARTS from the oracle's end state — box settled in the pit, robot just in front of
it — and asks RL to learn ONLY the traversal: step onto the settled box and climb
the 0.4 m far step to reach x > 3.5.

vs UnitreeB2TaskDEnvCfg (integrated):
  * box spawns SETTLED in the pit (not on the platform at (-3, 1.6, 0.5)),
  * robot spawns just in front of the pit (not back at (-3, 0)),
  * both are randomized within the pit geometry each reset (robustness to the
    oracle's slightly-varying final pose at deploy time),
  * box-pushing rewards are removed; only the crossing objective remains,
  * episode is short (crossing only).

Warm-starts from the integrated/flat checkpoint (same robot/obs/action interface).
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import atec_rl_lab.train.locomotion.velocity.mdp as vel_mdp
import atec_rl_lab.train.locomotion.taskd.mdp as taskd_mdp
from atec_rl_lab.train.locomotion.taskd.config.quadruped.unitree_b2.taskd_env_cfg import (
    UnitreeB2TaskDEnvCfg,
)


# ==========================================================================
# COORDINATE-FRAME FIX (2026-06-22). The previous poses were ~4.2 m off in x:
# they were written as if `init_state.pos` were WORLD frame, but in this
# (velocity-lineage) training env the reset event does
#     world_pos = init_state.pos + env_origins   (events.py:256)
# and env_origin = (-4.2, 0, 0). So init_state.pos=(0.2,...) actually spawned the
# box at WORLD x=-4.0 — 4.2 m BEHIND the pit (which is at world x in [-0.4,+0.4]).
# The robot likewise spawned at world x=-4.5. That is why the prior 1287-iter run
# never reached any milestone: the robot had to march +4.2 m on flat ground just to
# reach the near rim. (Confirmed empirically: diag_box_at_settle.py shows the pit at
# world x in [-0.4,+0.4], box settles z=-0.7; experiment_climb pins the box at
# org+4.4 = world +0.2 = pit center.)
#
# The reward thresholds (nearrim -0.15, enter 0.0, farrim 0.8, advance 2.0,
# reach_goal 3.5) and milestone re-arm (x < -0.25) are all in WORLD frame, so we
# must spawn at the matching world poses. init_state.pos = world_target - env_origin
# = world_target + 4.2.
_ENV_ORIGIN_X = -4.2  # this fixed single-tile env's origin (diag_taskd_layout.py)

# Desired WORLD poses (what the reward thresholds assume):
#   box  settled in the pit, top at z=-0.4 (center z=-0.7) -> world (0.2, -0.45, -0.7)
#   robot RIGHT AT the near pit edge (world x=-0.3) so it must confront the climb
#         immediately; facing +x. (Pit near edge ~ world x=-0.4.)
_BOX_WORLD = (0.2, -0.45, -0.7)
_ROBOT_WORLD = (-0.3, -0.40, 0.55)

# init_state.pos = world - env_origin  (only x is offset; env_origin y=z=0).
_BOX_SETTLED_POS = (_BOX_WORLD[0] - _ENV_ORIGIN_X, _BOX_WORLD[1], _BOX_WORLD[2])      # (4.4, -0.45, -0.7)
_ROBOT_FRONT_POS = (_ROBOT_WORLD[0] - _ENV_ORIGIN_X, _ROBOT_WORLD[1], _ROBOT_WORLD[2])  # (3.9, -0.40, 0.55)


# ==========================================================================
# BOX POSITION — single source of truth (2026-06-23). The box is the stepping
# stone the robot mounts to climb the pit's far step. Everything that depends on
# WHERE the box sits — its reset pose AND the jump-reward x-bands — is derived
# from this one world-x so moving the box only requires editing _BOX_CENTER_X_WORLD.
#
# Frame rules (do NOT mix them up):
#   * Reset events take an ENV-LOCAL x: world = env_local + env_origin, so
#     env_local = world - env_origin = world + 4.2  -> use _box_reset_x().
#   * Jump-reward x-bands are WORLD frame (rewards read root_pos_w) -> use the
#     world-x bands directly, NO +4.2.
_BOX_CENTER_X_WORLD = 1.5     # box center, WORLD x (user-set 2026-06-23: box against far wall)
_BOX_X_HALF = 0.4             # box half-length in x (CuboidCfg size x=0.8)

# Box faces in world x: front (-x, robot hits first) and back (+x, robot ends on top).
_BOX_FRONT_X = _BOX_CENTER_X_WORLD - _BOX_X_HALF   # 0.8
_BOX_BACK_X = _BOX_CENTER_X_WORLD + _BOX_X_HALF    # 1.6


def _box_reset_x() -> float:
    """Box reset x in ENV-LOCAL frame (what reset_root_state_uniform expects)."""
    return _BOX_CENTER_X_WORLD - _ENV_ORIGIN_X     # 1.2 + 4.2 = 5.4


# Jump-reward x-bands (WORLD frame), all relative to the box so they track it.
# The robot mounts the box from its front face and ends past its back face, so:
#   approach/takeoff:  from a bit BEFORE the front face to just past center
#   flight/base_height: from the front face to just past the back face (on top)
_JUMP_TAKEOFF_BAND = (_BOX_FRONT_X - 0.4, _BOX_CENTER_X_WORLD)        # (0.4, 1.2): launch in front of / onto the box
_JUMP_FLIGHT_BAND = (_BOX_FRONT_X - 0.4, _BOX_BACK_X + 0.1)           # (0.4, 1.7): airborne over the box
_JUMP_HEIGHT_BAND = (_BOX_FRONT_X, _BOX_BACK_X + 0.1)                 # (0.8, 1.7): body up on the box top
_JUMP_FOOTCLEAR_BAND = (_BOX_FRONT_X - 0.5, _BOX_CENTER_X_WORLD)      # (0.3, 1.2): lift feet approaching the front face


@configclass
class UnitreeB2TaskDCrossEnvCfg(UnitreeB2TaskDEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ------------------------------ Observations ------------------------
        # Warm-start from the generic walking checkpoint logs/rsl_rl/unitree_b2_rough
        # (user-chosen 2026-06-23). That policy was trained with NO height_scan in the
        # actor obs (actor in=45: base_lin_vel=None, height_scan=None — see
        # rough_env_cfg.py:38-39 and its saved params/env.yaml). The Task D lineage
        # keeps height_scan -> actor in=232, so the checkpoint would NOT load (shape
        # mismatch on actor.0). Drop height_scan here too so the actor obs is the same
        # 45-dim layout and the rough checkpoint loads cleanly. The jump rewards use
        # base z / foot z / contacts directly, so they still work blind (no terrain
        # scan) — the robot jumps by proprioception + the shaped rewards.
        self.observations.policy.height_scan = None

        # CRITIC obs must ALSO match the rough checkpoint (critic in=235), else
        # rsl_rl rejects the load (size mismatch critic.0: 235 vs 251). The 16-dim
        # gap is the arm: the velocity-lineage critic uses joint_pos/joint_vel over
        # ALL joints (joint_names=".*") -> 20 DoF on B2+Piper -> +16 vs the rough
        # checkpoint's 12 leg joints. Restrict the critic's joint obs to the same
        # 12 leg joints as the actor so critic in == 235 and the checkpoint loads.
        # (height_scan STAYS on the critic: the rough critic includes it; only the
        # actor dropped it. Asymmetric actor/critic obs is fine for PPO.)
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------ Reset randomization -----------------
        # Randomize both within the PIT GEOMETRY each reset so the policy is robust
        # to the oracle's slightly-varying final box/robot pose at deploy time.
        # Box x: settled center 0.2, move within pit (half-width 0.7 - box half 0.4
        #        = 0.3 of slack) -> +/-0.25. y along the (long) pit -> +/-0.20.
        #        Small yaw jitter so the policy doesn't overfit a perfectly square box.
        self.events.reset_box = EventTerm(
            func=vel_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("box"),
                "pose_range": {"x": (_box_reset_x(), _box_reset_x()), "y": (-0.20, 0.20), "yaw": (0, 0)},
                "velocity_range": {},
            },
        )
        # Robot x: in front of the pit, +/-0.20 (closer risks pitching in, farther
        # adds run-up). y tracks the box lane -> +/-0.15. Small yaw to vary heading.
        self.events.reset_robot_front = EventTerm(
            func=vel_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "pose_range": {"x": (-0.10+4.2, 0.10+4.2), "y": (-0.15, 0.15), "yaw": (-0.15, 0.15)},
                "velocity_range": {},
            },
        )
        # Disable the inherited base reset so it doesn't ALSO reset the robot each
        # episode (it and reset_robot_front both target the robot; whichever runs last
        # wins, making the start pose nondeterministic). reset_robot_front fully
        # replaces it for the crossing task.
        self.events.randomize_reset_base = None

        # ------------------------------ Rewards -----------------------------
        # Crossing only: drop the box-pushing terms (the box is already placed).
        self.rewards.taskd_box_progress_x.weight = 0.0
        self.rewards.taskd_box_in_target.weight = 0.0

        # FULLY REMOVE the terms that paid the robot to stand still or that penalized the
        # climbing motion (per user 2026-06-22). Re-weighting (upward 3.0->0.3 etc.) did
        # NOT break the "stand and shuffle in front of the pit" local optimum in the
        # 1287-iter run: upward's raw signal was so large it still reached episode reward
        # ~11.9 at weight 0.3. So zero them out entirely instead of fighting the weights:
        #   - upward (was 3.0): reward for being upright -> paid ~12/step for standing.
        #   - track_lin_vel_xy_exp / track_ang_vel_z_exp: per-step "follow the velocity
        #     command" mime reward -> paid ~110/episode for marching in place. A fixed-goal
        #     crossing task doesn't need command tracking at all.
        #   - lin_vel_z_l2 (was -2.0): penalized the vertical velocity of stepping up.
        #   - feet_height_body (was -5.0): penalized lifting the feet high to climb.
        # disable_zero_weight_rewards() (below) then drops them from the manager.
        self.rewards.upward.weight = 0.0
        self.rewards.track_lin_vel_xy_exp.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.feet_height_body.weight = 0.0

        # Drive: dense forward progress + a stand-still penalty so doing nothing is
        # strictly worse than advancing. (Uprightness is still enforced by the fall
        # termination below, so removing `upward` doesn't let it flop indefinitely.)
        self.rewards.taskd_progress_x.weight = 60.0          # was 40: push forward harder
        self.rewards.stand_still.weight = -5.0               # was -2: punish staying put

        # STAGED MILESTONES — now made BIG so crossing decisively beats standing, and the
        # crucial NEAR-RIM bonus (per user) is larger than any fall cost so the robot is
        # rewarded for committing past the pit edge even at some risk of falling in.
        self.rewards.taskd_m_nearrim = RewTerm(        # reached the near pit edge (commit!)
            func=taskd_mdp.reach_x_milestone, weight=120.0,
            params={"asset_cfg": SceneEntityCfg("robot"), "x_threshold": -0.15, "key": "nearrim"},
        )
        self.rewards.taskd_m_enter = RewTerm(          # stepped onto the box in the pit
            func=taskd_mdp.reach_x_milestone, weight=150.0,
            params={"asset_cfg": SceneEntityCfg("robot"), "x_threshold": 0.0, "key": "enter"},
        )
        self.rewards.taskd_m_farrim = RewTerm(         # climbed onto the far platform!
            func=taskd_mdp.reach_x_milestone, weight=250.0,
            params={"asset_cfg": SceneEntityCfg("robot"), "x_threshold": 0.8, "key": "farrim"},
        )
        self.rewards.taskd_m_advance = RewTerm(        # advancing across the far platform
            func=taskd_mdp.reach_x_milestone, weight=80.0,
            params={"asset_cfg": SceneEntityCfg("robot"), "x_threshold": 2.0, "key": "advance"},
        )

        # ---------------------- JUMP / dynamic-climb shaping ----------------
        # The walking prior can step up small ledges but never *jumps* the ~0.4 m
        # box-top -> platform-top residual (it shuffles in place; see memory
        # atec2026-taskd-climb-findings). These parkour-style terms pay the policy
        # for the explosive vertical takeoff + flight phase that mounts the step.
        # All x-bands are WORLD frame and DERIVED FROM THE BOX position
        # (_BOX_CENTER_X_WORLD at module top): _JUMP_*_BAND track the box front/back
        # faces, so moving the box auto-realigns every band. NO +4.2 here (rewards
        # read world-frame root_pos_w); the +4.2 lives only in the reset events.
        #
        # foot_link_name / base z come from the B2 binding (UnitreeB2TaskDEnvCfg):
        # base standing z ~0.55 on flat ground; platform top adds ~0.4 -> target ~0.95.
        _foot = self.foot_link_name  # ".*_foot"

        # (1) Explosive UP velocity right at the step face. The single most direct
        #     "jump now" signal. Capped so one spike can't dominate.
        self.rewards.taskd_takeoff = RewTerm(
            func=taskd_mdp.takeoff_impulse, weight=8.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "x_lo": _JUMP_TAKEOFF_BAND[0], "x_hi": _JUMP_TAKEOFF_BAND[1], "vz_cap": 2.5,
            },
        )
        # (2) All-four-feet-airborne flight phase — the defining feature of a jump,
        #     gated to the step band so it isn't farmed by pronking across the flat top.
        self.rewards.taskd_flight = RewTerm(
            func=taskd_mdp.flight_phase, weight=20.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_foot),
                "asset_cfg": SceneEntityCfg("robot"),
                "x_lo": _JUMP_FLIGHT_BAND[0], "x_hi": _JUMP_FLIGHT_BAND[1],
            },
        )
        # (3) Reach platform-mounting base height over the step band (Gaussian pull).
        #     The term that most reliably gets the BODY up to the right altitude.
        self.rewards.taskd_base_height = RewTerm(
            func=taskd_mdp.base_height_target_band, weight=15.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "target_height": 0.95, "std": 0.3, "x_lo": _JUMP_HEIGHT_BAND[0], "x_hi": _JUMP_HEIGHT_BAND[1],
            },
        )
        # (4) Lift the feet high while approaching the step face so they land ON the
        #     step instead of stubbing into the vertical wall.
        self.rewards.taskd_foot_clear = RewTerm(
            func=taskd_mdp.foot_clearance_band, weight=4.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=_foot),
                "target_clearance": 0.45, "x_lo": _JUMP_FOOTCLEAR_BAND[0], "x_hi": _JUMP_FOOTCLEAR_BAND[1],
            },
        )
        # (5) Penalize slamming landings (negative weight) so the jump ends absorbed.
        self.rewards.taskd_hard_landing = RewTerm(
            func=taskd_mdp.hard_landing_penalty, weight=-2e-3,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_foot),
                "force_threshold": 400.0,
            },
        )

        # ------------------------------ Terminations ------------------------
        # RELAX the fall cutoff to MATCH THE EVALUATION ENV (TaskDEnvB2Cfg uses -0.4):
        # climbing onto/over the box dips the base, and stepping down into the pit drops
        # it further; at minimum_height=0.25 those legitimate crossing motions terminate
        # the episode, so the policy learns to never leave the safe flat ground. -0.4
        # lets the robot dip into the pit / climb the step without being killed (and keeps
        # training consistent with scoring). A true fall to the pit floor is far lower.
        self.terminations.fall.params["minimum_height"] = -0.4

        # MATCH THE EVALUATION ENV: turn illegal_contact OFF entirely. The eval env
        # (tasks/task_d/env_cfg.py:136) sets `self.terminations.illegal_contact = None`,
        # keeping only `fall` (base z < -0.4) and the x>3.5 success term. Our training
        # env inherited illegal_contact from the velocity lineage (base_link contact
        # force > 1.0 N -> terminate), which is fine for flat walking but during the
        # 0.4 m step-up the base legitimately brushes the box / far wall — that fired on
        # ~79% of episodes at iter ~665, killing every climb attempt before it could
        # reach the far rim (enter milestone plateaued, farrim stayed 0). It is also a
        # train-eval gap: a behavior penalized in training that scoring never checks.
        # Removing it matches scoring exactly and lets the climb actually be explored.
        self.terminations.illegal_contact = None

        # ------------------------------ Curriculum --------------------------
        # The velocity-command curricula (command_levels_lin_vel / _ang_vel) tune the
        # `track_lin_vel_xy_exp` / `track_ang_vel_z_exp` reward terms — which we just
        # removed (weight 0 -> dropped by disable_zero_weight_rewards). Leaving the
        # curricula active would reference a non-existent reward term and crash. A
        # fixed-goal crossing task has no velocity command to curriculum anyway, so
        # disable both. (terrain_levels is already None in TaskDRoughEnvCfg.)
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

        # ------------------------------ Episode -----------------------------
        # Crossing from in front of the pit is quick; no long box-push phase.
        self.episode_length_s = 8.0

        # Drop the now-zero-weight box rewards so they don't reference box dynamics
        # we no longer shape (the box entity still exists as a stepping stone).
        if self.__class__.__name__ == "UnitreeB2TaskDCrossEnvCfg":
            self.disable_zero_weight_rewards()
