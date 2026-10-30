"""Arm poses a grasp or a place actually asks for, for stage-3 training.

Stage 2 sampled arm goals uniformly over a joint box whose shoulder roll ran
0.5 rad into the torso (the upper arm meets it at ~0.06 rad). The policy
learned to balance against jammed arms shoving the chest, which no real
reach does. This solves the runtime's own IK (Runtime.reach_joint_space:
same posture bias, same collision-aware roll limit) for hand targets across
the band the right hand can reach at a bench, and saves the joint poses.
The left arm uses the mirror image.

    python training/arm_goal_bank.py   # -> training/arm_goal_bank.npz
"""

import pathlib
import sys

import mujoco
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from controllers.diff_ik import DiffIK  # noqa: E402
from training.walk_stop_env import WalkStopEnv  # noqa: E402


def main():
    env = WalkStopEnv(seed=0)
    env.reset()
    m, d = env.model, env.data
    arm = env.body.cfg["arms"]["right"]
    joints = arm["joints"]
    ik = DiffIK(m, "right_ee", joints) if "right_ee" in [
        m.site(i).name for i in range(m.nsite)] else None
    if ik is None:
        from skills.runtime import EE_SITE
        ik = DiffIK(m, EE_SITE["right"], joints)
    qadr = np.array([m.joint(j).qposadr[0] for j in joints])
    lim = np.array([m.joint(j).range for j in joints])
    for j, l in (arm.get("ik_limits") or {}).items():
        lim[joints.index(j)] = l
    ready = np.asarray(arm["ready"], float)
    snap = d.qpos.copy()
    pelvis = d.xpos[env.root].copy()
    rng = np.random.default_rng(0)
    poses = []
    for _ in range(1500):
        # pelvis frame: ahead, right of the midline, bench to hover height
        target = pelvis + [rng.uniform(0.22, 0.40), rng.uniform(-0.40, -0.18),
                           rng.uniform(0.15, 0.40)]
        d.qpos[:] = snap
        d.qpos[qadr] = ready
        mujoco.mj_forward(m, d)
        for _ in range(600):
            err = target - d.site_xpos[ik.site_id]
            if np.linalg.norm(err) < 0.008:
                break
            qd = ik.solve(d, np.r_[3.0 * err, np.zeros(3)], posture=ready,
                          posture_gain=1.2)
            d.qpos[qadr] = np.clip(d.qpos[qadr] + qd * 0.01,
                                   lim[:, 0], lim[:, 1])
            mujoco.mj_forward(m, d)
        if np.linalg.norm(target - d.site_xpos[ik.site_id]) < 0.02:
            poses.append(d.qpos[qadr].copy())
    poses = np.array(poses)
    out = REPO / "training" / "arm_goal_bank.npz"
    np.savez(out, right=poses, ready=ready)
    print(f"{len(poses)} reachable poses of 1500 -> {out}")
    print("roll range", poses[:, 1].min().round(2), poses[:, 1].max().round(2))


if __name__ == "__main__":
    main()
