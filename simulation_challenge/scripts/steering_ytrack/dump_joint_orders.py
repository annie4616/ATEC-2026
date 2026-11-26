"""Dump the ACTUAL runtime joint orderings on both sides so the parkour-student <-> Task A
joint permutation can be built from measured data, not hand-copied name lists.

Side A (Task A A2 env):
  - the robot articulation's physics DOF order (asset.data.joint_names)
  - the proprio obs joint order  = action_manager / obs term resolved ids
  - the action joint order (joint_leg + joint_arm resolved ids)

Side B (parkour student): its action term used joint_names=['.*'], preserve_order=False, so
its obs/action order == the articulation DOF order of the SAME robot (a2_piper). Since both
envs spawn the same UNITREE_A2_PIPER_CFG, the student's order == this env's articulation DOF
order. So we only need to print, for the Task A env:
  (1) articulation DOF order         -> STUDENT basis
  (2) Task A proprio/action order    -> TASK A basis
and the permutation is name->index between them.
"""
import argparse
from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--task", type=str, default="ATEC-TaskA-A2Piper")
ap.add_argument("--out", type=str, required=True)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import atec_rl_lab.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

def main():
    out = open(args.out, "w")
    def w(m): out.write(str(m) + "\n"); out.flush()

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    # disable cameras (fast) — mirror the play driver
    for c in ("head_camera", "ee_camera", "ee_dual_camera"):
        if hasattr(env_cfg.scene, c):
            setattr(env_cfg.scene, c, None)
    if hasattr(env_cfg.observations, "image"):
        env_cfg.observations.image = None

    env = gym.make(args.task, cfg=env_cfg)
    u = env.unwrapped
    robot = u.scene["robot"]

    # (1) articulation DOF order (== parkour STUDENT obs/action order, since student used
    #     joint_names=['.*'], preserve_order=False on this same robot)
    artic = list(robot.data.joint_names)
    w("=== ARTICULATION DOF ORDER (STUDENT basis) ===")
    for i, n in enumerate(artic):
        w(f"  stu[{i:2d}] = {n}")

    # (2) Task A proprio joint order (obs term joint_pos resolved ids)
    try:
        term = u.observation_manager._group_obs_term_cfgs  # dict group->list
    except Exception:
        term = None
    # More robust: read the action terms' resolved joint order (joint_leg + joint_arm)
    am = u.action_manager
    w("\n=== ACTION TERMS resolved joint order (TASK A action basis) ===")
    taska_action_names = []
    for tname in am.active_terms:
        t = am.get_term(tname) if hasattr(am, "get_term") else None
    # action_manager stores terms in _terms
    for tname, tcfg in am._terms.items():
        ids = getattr(tcfg, "_joint_ids", None)
        names = None
        if ids is not None:
            if isinstance(ids, slice):
                names = artic  # slice(None) => all in articulation order
            else:
                names = [artic[i] for i in ids]
        w(f"  action term '{tname}': joint_names={names}")
        if names:
            taska_action_names.extend(names)
    w(f"\n  => TASK A action full order ({len(taska_action_names)}): {taska_action_names}")

    # (3) Task A proprio joint order via the obs term's asset_cfg.joint_ids
    w("\n=== PROPRIO obs joint order (TASK A obs basis) ===")
    try:
        om = u.observation_manager
        for grp, cfgs in om._group_obs_term_cfgs.items():
            names_grp = om._group_obs_term_names[grp]
            for cfg, nm in zip(cfgs, names_grp):
                if nm in ("joint_pos", "joint_vel"):
                    sc = cfg.params.get("asset_cfg")
                    jid = getattr(sc, "joint_ids", None)
                    if isinstance(jid, slice):
                        nn = artic
                    elif jid is not None:
                        nn = [artic[i] for i in jid]
                    else:
                        nn = None
                    w(f"  obs '{grp}.{nm}': {nn}")
    except Exception as e:
        w(f"  proprio dump failed: {e}")

    # (4) compute permutations directly from measured lists
    #     STUDENT basis = artic ; TASK A basis = taska_action_names
    if len(taska_action_names) == len(artic) == 20:
        STU_FROM_TASKA = [taska_action_names.index(n) for n in artic]
        TASKA_FROM_STU = [artic.index(n) for n in taska_action_names]
        w("\n=== MEASURED PERMUTATIONS ===")
        w(f"STU_FROM_TASKA = {STU_FROM_TASKA}")
        w(f"TASKA_FROM_STU = {TASKA_FROM_STU}")

    env.close()
    out.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
