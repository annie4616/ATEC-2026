"""Find the EXACT permutation from the proprio joint_pos vector to the oracle's
LEG_JOINT_NAMES (then ARM_JOINT_NAMES) order, so the submission can reconstruct the
oracle observation purely from obs['proprio'] (no robot.data access).

The proprio 'joint_pos' obs term uses SceneEntityCfg('robot', joint_names='.*',
preserve_order=True). We need the actual joint-name order that '.*' resolves to in
THIS env (it is NOT the articulation order, per diag: maxdiff 0.328). We get it by
reading the ObservationManager term's resolved joint_ids/names.

Writes the permutation arrays to a RESULT npz the submission will embed."""
import os, sys
from isaaclab.app import AppLauncher
ap = __import__("argparse").ArgumentParser(); AppLauncher.add_app_launcher_args(ap)
a = ap.parse_args(); a.enable_cameras = True; a.headless = True
app = AppLauncher(a); sim = app.app

import numpy as np, torch
from isaaclab.envs import ManagerBasedRLEnv
import atec_rl_lab.tasks  # noqa
from atec_rl_lab.tasks.task_d.env_cfg import TaskDEnvB2Cfg
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "act"))
from task_d.walking_oracle import LEG_JOINT_NAMES, ARM_JOINT_NAMES

OUT = "/hdd0/sohyun/tmp/diag_perm_result.txt"
fh = open(OUT, "w")
def w(*x):
    fh.write(" ".join(str(s) for s in x) + "\n"); fh.flush()

cfg = TaskDEnvB2Cfg(); cfg.scene.num_envs = 1
env = ManagerBasedRLEnv(cfg)
robot = env.unwrapped.scene["robot"]
obs, _ = env.reset()

art_names = list(robot.data.joint_names)
w("articulation order:", art_names)

# Resolve the proprio joint_pos term's joint order. The term uses asset_cfg with
# joint_names='.*' preserve_order=True. Find the resolved joint_ids -> names.
# We reconstruct it the same way IsaacLab does: SceneEntityCfg resolution.
from isaaclab.managers import SceneEntityCfg
sc = SceneEntityCfg("robot", joint_names=".*", preserve_order=True)
sc.resolve(env.unwrapped.scene)
proprio_joint_ids = list(sc.joint_ids) if not isinstance(sc.joint_ids, slice) else list(range(len(art_names)))
proprio_joint_names = [art_names[i] for i in proprio_joint_ids]
w("proprio joint_pos order (resolved '.*'):", proprio_joint_names)
w("proprio joint_ids:", proprio_joint_ids)

# Permutation: for each oracle LEG joint (in LEG_JOINT_NAMES order), its index within
# the proprio joint vector.
leg_perm = [proprio_joint_names.index(n) for n in LEG_JOINT_NAMES]
arm_perm = [proprio_joint_names.index(n) for n in ARM_JOINT_NAMES]
w("\nLEG_JOINT_NAMES:", LEG_JOINT_NAMES)
w("leg_perm (proprio idx for each oracle leg joint):", leg_perm)
w("ARM_JOINT_NAMES:", ARM_JOINT_NAMES)
w("arm_perm:", arm_perm)

# Also: the ACTION vector order. joint_leg action group uses the robot's leg joints;
# we need the inverse map: given oracle-order leg action (12), where does each go in
# the 20-D env action vector? The env action vector concatenates action terms
# [joint_leg(12), joint_arm(8)] (joint_wheel empty for B2Piper). Each term's internal
# order is the resolved joint order of that term's joint_names. The collector's oracle
# builds action_env as [:12]=legs(in LEG_JOINT_NAMES order), [12:]=arm -> and that
# matched seq exactly (diag maxdiff 0). So the ENV ACTION is leg-first in
# LEG_JOINT_NAMES order, arm next in ARM_JOINT_NAMES order. Confirm seq is consumed
# that way by checking action term joint orders.
try:
    am = env.unwrapped.action_manager
    for term_name in am.active_terms:
        term = am.get_term(term_name)
        jn = [art_names[i] for i in term._joint_ids] if hasattr(term, "_joint_ids") else "?"
        w(f"action term '{term_name}' joint order:", jn)
except Exception as e:
    w("action term introspection failed:", repr(e))

np.savez("/hdd0/sohyun/tmp/proprio_perm.npz",
         leg_perm=np.array(leg_perm, dtype=np.int64),
         arm_perm=np.array(arm_perm, dtype=np.int64),
         proprio_joint_names=np.array(proprio_joint_names))
w("\nsaved /hdd0/sohyun/tmp/proprio_perm.npz")
fh.close()
env.close(); sim.close()
