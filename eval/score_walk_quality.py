"""How human is the walk in demo 13? Measured over the walk to the bench
(from 1 s after the start to the stop decision), N trials:

speed, cadence (steps/s), stride, swing-foot clearance, pelvis bob, torso
lean, knee flexion at mid-stance, and how much of the leg motion is above
8 Hz (tremor). Human references are the LAFAN1 values measured in the
companion g1-motion-imitation repo, scaled to the G1 leg.

    python eval/score_walk_quality.py <walk_policy.pt> [N]
"""
import importlib.util
import pathlib
import sys
from multiprocessing import Pool

import mujoco
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
HUMAN = {"speed": 0.49, "cadence": 1.93, "bob": 0.073, "knee_max": 32.0}
LEGS = [f"{s}_{j}_joint" for s in ("left", "right")
        for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee",
                  "ankle_pitch", "ankle_roll")]


def run(args):
    seed, walk = args
    spec = importlib.util.spec_from_file_location(
        "d13", REPO / "demos" / "13_walk_across_and_fetch.py")
    d13 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(d13)
    from skills.runtime import Runtime
    import skills.runtime as rtmod
    d13.HOLD = "stance"
    d13.STOP_DECIDED[0] = -0.37
    d13.WALK_ONLY = True
    ctx, rows = {}, []
    init = Runtime.__init__

    def patched(self, world, *a, **k):
        init(self, world, *a, **k)
        m = world.model
        ctx.update(rt=self, torso=m.body("torso_link").id,
                   feet=[m.body(f"{s}_ankle_roll_link").id
                         for s in ("left", "right")],
                   lq=np.array([m.joint(n).qposadr[0] for n in LEGS]),
                   lv=np.array([m.joint(n).dofadr[0] for n in LEGS]))
    Runtime.__init__ = patched
    step, n = mujoco.mj_step, [0]

    def logged(m, d, *a):
        step(m, d, *a)
        n[0] += 1
        rt = ctx.get("rt")
        if rt is None or rt._standing or n[0] % 2:
            return
        c = ctx
        R = d.xmat[c["torso"]].reshape(3, 3)
        rows.append(np.concatenate([
            [d.time, d.xpos[rt.root_id][0], d.xpos[rt.root_id][2],
             np.degrees(np.arcsin(np.clip(R[0, 2], -1, 1)))],
            [d.xpos[f][2] for f in c["feet"]],
            d.qpos[c["lq"]], d.qvel[c["lv"]]]))
    rtmod.mujoco.mj_step = logged
    r = d13.trial((seed, False, "third_party/policies/g1_stance_human.pt",
                   walk))
    L = np.array(rows)
    L = L[L[:, 0] > 1.0]
    if len(L) < 300:
        return dict(stage=r["stage"])
    dt = 0.004
    t, x, z, lean = L[:, 0], L[:, 1], L[:, 2], L[:, 3]
    fz = L[:, 4:6]
    q, v = L[:, 6:18], L[:, 18:30]
    ground = np.percentile(fz, 5, axis=0)
    steps = 0
    for k in range(2):
        up = fz[:, k] > ground[k] + 0.02
        steps += int(np.sum(up[1:] & ~up[:-1]))
    dur = t[-1] - t[0]
    speed = (x[-1] - x[0]) / dur
    cadence = steps / dur
    P = np.abs(np.fft.rfft(v - v.mean(0), axis=0)) ** 2
    f = np.fft.rfftfreq(len(v), dt)
    hi = float(P[f >= 8].sum() / P[1:].sum())
    # pelvis bob: peak-to-peak within ~1 s windows, median
    w = int(1.0 / dt)
    bob = np.median([np.ptp(z[i:i + w]) for i in range(0, len(z) - w, w // 2)])
    knee = np.degrees(q[:, [3, 9]])
    return dict(stage=r["stage"], speed=speed, cadence=cadence,
                stride=speed / max(cadence, 1e-6) * 2,
                clearance=float(np.mean(np.max(fz, axis=0) - ground)),
                clear_p90=float(np.mean(np.percentile(fz, 90, axis=0) - ground)),
                bob=float(bob), lean=float(np.mean(lean)),
                knee_max=float(np.percentile(knee, 95)),
                leg_hf=hi)


def main():
    walk = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    with Pool(10, maxtasksperchild=1) as pool:
        res = [r for r in pool.map(run, [(9000 + 23 * k, walk)
                                         for k in range(n)]) if "speed" in r]
    print(walk, f"({len(res)}/{n} walks measured)")
    for k in ["speed", "cadence", "stride", "clear_p90", "bob", "lean",
              "knee_max", "leg_hf"]:
        v = [r[k] for r in res]
        h = HUMAN.get(k)
        print(f"  {k:10s} median {np.median(v):7.3f}"
              + (f"   human {h}" if h is not None else ""))


if __name__ == "__main__":
    main()
