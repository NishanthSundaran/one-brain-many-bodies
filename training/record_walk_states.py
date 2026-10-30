"""Record the states the stance policy is handed: the walking policy steered
to a point, its command eased to zero, and the robot left to slow down.

Every control tick from the start of the slow-down to 3.5 s later is saved
(robot joints only, pelvis moved back to the origin), so the stance policy
can be trained to catch a robot at any moment of it. Seeds here are disjoint
from those eval/score_handover.py scores on.

    python training/record_walk_states.py [runs]
"""

import pathlib
import sys
from multiprocessing import Pool

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import mujoco  # noqa: E402

from bodies.body import Body  # noqa: E402
from controllers.locomotion import POLICY_JOINTS  # noqa: E402
from sim.world import World  # noqa: E402
from skills.runtime import Runtime  # noqa: E402

CFG = REPO / "bodies" / "g1_walk" / "config.yaml"
OUT = REPO / "training" / "walk_states.npz"
SLOWDOWN_S = 3.5


def robot_joints():
    b = Body(CFG)
    b.spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                              size=[8, 8, 0.1])
    m = b.spec.compile()
    return [m.joint(j).name for j in range(m.njnt)]


def run(seed):
    rng = np.random.default_rng(seed)
    start = (-2.5 + rng.uniform(-0.3, 0.3), rng.uniform(-0.4, 0.4))
    world = World(Body(CFG), start_xy=start)
    m, d = world.model, world.data
    for n in POLICY_JOINTS:
        d.qpos[m.joint(n).qposadr[0]] += rng.uniform(-0.02, 0.02)
    mujoco.mj_forward(m, d)
    rt = Runtime(world, start="walk")
    p = rng.uniform(-np.pi, np.pi)
    rt.loco.phase = np.array([p, np.fmod(p + 2 * np.pi, 2 * np.pi) - np.pi])
    names = robot_joints()
    qadr = [(m.joint(n).qposadr[0], m.joint(n).dofadr[0],
             4 if m.joint(n).type == mujoco.mjtJoint.mjJNT_FREE else 0)
            for n in names]
    target = np.array([-0.5, rng.uniform(-0.3, 0.3)])
    if not rt.gait_travel(target, arrive=0.30):
        return None
    tick = rt.loco.n_substeps * rt.dt
    v0 = rt.loco.command.copy()
    ramp = rng.uniform(0.3, 1.0)
    n_ramp = max(1, int(round(ramp / tick)))
    Q, V = [], []
    for k in range(int(SLOWDOWN_S / tick)):
        s = min(1.0, (k + 1) / n_ramp)
        rt.loco.set_command(*(v0 * (1.0 - s)))
        rt.step(tick)
        if rt.fallen():
            break
        q, v = [], []
        for qa, da, free in qadr:
            if free:
                pos = d.qpos[qa:qa + 7].copy()
                pos[:2] = 0.0
                q.extend(pos)
                v.extend(d.qvel[da:da + 6])
            else:
                q.append(d.qpos[qa])
                v.append(d.qvel[da])
        Q.append(q)
        V.append(v)
    return np.array(Q), np.array(V)


def main():
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    with Pool() as pool:
        out = [r for r in pool.map(run, [20000 + 7 * k for k in range(runs)])
               if r is not None and len(r[0])]
    Q = np.concatenate([q for q, _ in out])
    V = np.concatenate([v for _, v in out])
    np.savez_compressed(OUT, qpos=Q, qvel=V, joints=np.array(robot_joints()))
    print(f"{len(out)} runs, {len(Q)} states -> {OUT}")


if __name__ == "__main__":
    main()
