"""Record the arm commands demo 13 actually sends while the robot stands
at the bench (ready swing, grasp, carry, place, lowering), so the stance
policy trains against the real disturbance rather than random poses.

Earlier from-scratch policies trained on random arm goals stood still in
training and fell in 16/20 demo grasps: the demo's arm motion was outside
what they had seen.

    python training/record_arm_traj.py 40   # -> training/arm_traj.npz
"""
import importlib.util
import pathlib
import sys
from multiprocessing import Pool

import mujoco
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from controllers.locomotion import ARMS  # noqa: E402

STANCE = "third_party/policies/g1_stance_humanA.pt"
WALK = "third_party/policies/g1_walk_arms.pt"
SMOOTH = True   # record the arm motion the demo now uses


def run(seed):
    spec = importlib.util.spec_from_file_location(
        "d13", REPO / "demos" / "13_walk_across_and_fetch.py")
    d13 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(d13)
    from skills.runtime import Runtime
    import skills.runtime as rtmod
    d13.HOLD = "stance"
    d13.STOP_DECIDED[0] = -0.37
    d13.SMOOTH_ARM = SMOOTH
    ctx, cmds = {}, []
    init = Runtime.__init__

    def patched(self, world, *a, **k):
        init(self, world, *a, **k)
        ctx["rt"], ctx["m"], ctx["d"] = self, world.model, world.data
        ctx["act"] = np.array([world.model.actuator(n).id for n in ARMS])
    Runtime.__init__ = patched
    step, n = mujoco.mj_step, [0]

    def logged(m, d, *a):
        step(m, d, *a)
        n[0] += 1
        rt = ctx.get("rt")
        if rt is not None and rt._standing and n[0] % 10 == 0:   # 50 Hz
            cmds.append(d.ctrl[ctx["act"]].copy())
    rtmod.mujoco.mj_step = logged
    r = d13.trial((seed, False, STANCE, WALK))
    return r["stage"], np.array(cmds, dtype=np.float32)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    with Pool(10, maxtasksperchild=1) as pool:
        rows = pool.map(run, [5000 + 7 * k for k in range(n)])
    cmd = [c for _, c in rows if len(c) > 50]
    start = np.cumsum([0] + [len(c) for c in cmd[:-1]])
    out = REPO / "training" / "arm_traj.npz"
    np.savez(out, cmd=np.concatenate(cmd), start=start,
             length=np.array([len(c) for c in cmd]))
    from collections import Counter
    print(Counter(s for s, _ in rows))
    print(f"wrote {out}: {len(cmd)} runs, "
          f"{sum(len(c) for c in cmd) * 0.02:.0f} s of arm commands")
