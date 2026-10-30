"""How human does demo 13 look? A scorecard over N trials, measured while
the robot stands at the bench (after the walk stops), so stance policies
and arm changes are judged on the same numbers, not by eye alone.

    python eval/score_fetch_quality.py <stance.pt> [N] [smooth|plain]

Reports task success plus: bench contacts, torso and pelvis tilt, pelvis
height and knee bend (crouch), leg joint speed (fidgeting), foot lifts
and foot travel per second (stepping/shuffling), hand jerk (smoothness of
the arm), and time to finish.
"""
import importlib.util
import pathlib
import sys
from multiprocessing import Pool

import mujoco
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import os
WALK = os.environ.get("WALK_POLICY", "third_party/policies/g1_walk_arms.pt")
LEGS = [f"{s}_{j}_joint" for s in ("left", "right")
        for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee",
                  "ankle_pitch", "ankle_roll")]


def run(args):
    seed, stance, smooth = args
    spec = importlib.util.spec_from_file_location(
        "d13", REPO / "demos" / "13_walk_across_and_fetch.py")
    d13 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(d13)
    from skills.runtime import Runtime
    import skills.runtime as rtmod
    d13.HOLD = "stance"
    d13.STOP_DECIDED[0] = -0.37
    d13.SMOOTH_ARM = smooth
    ctx, rows = {}, []
    init = Runtime.__init__

    def patched(self, world, *a, **k):
        init(self, world, *a, **k)
        m = world.model
        ctx.update(rt=self, m=m, d=world.data,
                   torso=m.body("torso_link").id,
                   feet=[m.body(f"{s}_ankle_roll_link").id
                         for s in ("left", "right")],
                   lq=np.array([m.joint(n).qposadr[0] for n in LEGS]),
                   lv=np.array([m.joint(n).dofadr[0] for n in LEGS]),
                   knee=[LEGS.index("left_knee_joint"),
                         LEGS.index("right_knee_joint")])
    Runtime.__init__ = patched
    step, n = mujoco.mj_step, [0]

    def logged(m, d, *a):
        step(m, d, *a)
        n[0] += 1
        rt = ctx.get("rt")
        if rt is None or not rt._standing or n[0] % 10:
            return
        c = ctx
        tilt = lambda b: np.degrees(np.arccos(np.clip(d.xmat[b][8], -1, 1)))
        rows.append(np.concatenate([
            [d.time, tilt(c["torso"]), tilt(rt.root_id),
             d.xpos[rt.root_id][2],
             np.mean(d.qpos[c["lq"]][c["knee"]]),
             np.sqrt(np.mean(d.qvel[c["lv"]] ** 2)),
             np.linalg.norm(d.xpos[c["feet"][0]][:2]
                            - d.xpos[c["feet"][1]][:2])],
            [d.xpos[f][0] for f in c["feet"]], [d.xpos[f][1] for f in c["feet"]],
            [d.xpos[f][2] for f in c["feet"]], rt.ee_pos("right")]))
    rtmod.mujoco.mj_step = logged
    r = d13.trial((seed, False, stance, WALK))
    L = np.array(rows)
    if len(L) < 20:
        return dict(stage=r["stage"])
    dt = 0.02
    fxy = np.stack([L[:, 7:9], L[:, 9:11]], axis=-1)        # (T, foot, xy)
    travel = np.sum(np.linalg.norm(np.diff(fxy, axis=0), axis=2))
    fz = L[:, 11:13]
    lifts = 0
    for k in range(2):
        up = fz[:, k] > fz[:, k].min() + 0.01
        lifts += int(np.sum(up[1:] & ~up[:-1]))
    ee = L[:, 13:16]
    jerk = np.diff(ee, n=3, axis=0) / dt ** 3
    dur = L[-1, 0] - L[0, 0]
    return dict(stage=r["stage"], t=r["t"],
                bench=r["bench_contacts"], bench_f=r["bench_force_max"],
                torso_mean=L[:, 1].mean(), torso_max=L[:, 1].max(),
                pelvis_tilt=L[:, 2].mean(), pelvis_z=L[:, 3].mean(),
                knee=np.degrees(L[:, 4].mean()), leg_speed=L[:, 5].mean(),
                stance_w=L[:, 6].mean(), lifts_s=lifts / dur,
                travel_s=travel / dur,
                hand_jerk=float(np.sqrt(np.mean(np.sum(jerk ** 2, axis=1)))))


def main():
    stance = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    smooth = (sys.argv[3] if len(sys.argv) > 3 else "smooth") == "smooth"
    with Pool(10, maxtasksperchild=1) as pool:
        res = pool.map(run, [(9000 + 23 * k, stance, smooth)
                             for k in range(n)])
    ok = [r for r in res if r["stage"] == "done"]
    from collections import Counter
    print(stance, "smooth arm" if smooth else "plain arm")
    print("  stages:", dict(Counter(r["stage"] for r in res)))
    full = [r for r in res if "torso_mean" in r]
    keys = ["torso_mean", "torso_max", "pelvis_tilt", "pelvis_z", "knee",
            "leg_speed", "stance_w", "lifts_s", "travel_s", "hand_jerk"]
    print(f"  success {len(ok)}/{n}, bench-free successes "
          f"{sum(r['bench'] == 0 for r in ok)}/{len(ok)}, "
          f"time {np.median([r['t'] for r in ok]) if ok else float('nan'):.0f} s")
    for k in keys:
        v = [r[k] for r in full]
        print(f"  {k:12s} median {np.median(v):8.3f}   p90 {np.percentile(v, 90):8.3f}")


if __name__ == "__main__":
    main()
