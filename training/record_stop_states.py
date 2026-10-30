"""Record the states the stance policy is handed in demo 13: the fine-tuned
walking policy (training/train_walkstop.py) walks 2 m, is told to stop, and
stops by itself. From the end of its slow-down to 1.5 s later, every control
tick is saved (robot joints only, pelvis moved back to the origin), so the
stance policy can be trained to take over at any moment of it.

    python training/record_stop_states.py [runs]
"""

import pathlib
import os
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
from training.record_walk_states import robot_joints  # noqa: E402

CFG = REPO / "bodies" / "g1_walk" / "config.yaml"
POLICY = os.environ.get("STOP_POLICY", "third_party/policies/g1_walk_arms.pt")
OUT = REPO / "training" / os.environ.get("STOP_OUT", "stop_states.npz")
HOLD_S = 1.5


def run(seed):
    rng = np.random.default_rng(seed)
    body = Body(CFG)
    body.cfg["locomotion"]["policy"] = POLICY
    body.cfg["locomotion"]["hold_with"] = "walk_policy"
    world = World(body, start_xy=(-2.5, rng.uniform(-0.3, 0.3)))
    m, d = world.model, world.data
    for n in POLICY_JOINTS:
        d.qpos[m.joint(n).qposadr[0]] += rng.uniform(-0.02, 0.02)
    mujoco.mj_forward(m, d)
    rt = Runtime(world, start="walk")
    p = rng.uniform(-np.pi, np.pi)
    rt.loco.phase = np.array([p, np.fmod(p + 2 * np.pi, 2 * np.pi) - np.pi])
    names = robot_joints()
    qadr = [(m.joint(n).qposadr[0], m.joint(n).dofadr[0],
             m.joint(n).type == mujoco.mjtJoint.mjJNT_FREE) for n in names]
    stop_x = rng.uniform(-1.0, -0.6)
    line = rt.base_xy[1]
    if not rt.gait_travel(lambda r: np.array([r.base_xy[0] + 1.0, line]),
                          arrive=0.05, until=lambda r: r.base_xy[0] >= stop_x):
        return None
    tick = rt.loco.n_substeps * rt.dt
    v0 = rt.loco.command.copy()
    for k in range(1, 31):                    # demo 13's 0.6 s ramp
        rt.loco.set_command(*(v0 * (1 - k / 30)))
        rt.step(tick)
    Q, V = [], []
    for _ in range(int(HOLD_S / tick)):
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
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    with Pool(10, maxtasksperchild=4) as pool:
        out = [r for r in pool.map(run, [30000 + 7 * k for k in range(runs)])
               if r is not None and len(r[0])]
    Q = np.concatenate([q for q, _ in out])
    V = np.concatenate([v for _, v in out])
    np.savez_compressed(OUT, qpos=Q, qvel=V, joints=np.array(robot_joints()))
    print(f"{len(out)} runs, {len(Q)} states -> {OUT}")


if __name__ == "__main__":
    main()
