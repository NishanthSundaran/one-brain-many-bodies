"""Does the walking policy stop when told to? Open floor, runtime plant.

Walk 2 m under gait_travel, far from the bench, ease the command to zero
over 0.6 s, then leave the walking policy running at zero for 4 s. Then
hand the legs to the stance policy (gait_stop, which now finds the robot
already still) and watch 3 s more.

Pass, fixed before the fine-tuned policy was first scored: no falls while
walking or stopping, body speed under 0.05 m/s 2 s after the stop in at
least 90% of trials, and at most 1 fall in 20 after the handover.

    python eval/score_walk_stop.py [--policy third_party/policies/x.pt]
"""

import argparse
import json
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
START, STOP_AT = (-6.0, 0.0), np.array([-4.0, 0.0])


def trial(args):
    seed, policy = args
    rng = np.random.default_rng(seed)
    body = Body(CFG)
    if policy:
        body.cfg["locomotion"]["policy"] = policy
    world = World(body, start_xy=START)
    m, d = world.model, world.data
    for n in POLICY_JOINTS:
        d.qpos[m.joint(n).qposadr[0]] += rng.uniform(-0.02, 0.02)
    mujoco.mj_forward(m, d)
    rt = Runtime(world, start="walk")
    p = rng.uniform(-np.pi, np.pi)
    rt.loco.phase = np.array([p, np.fmod(p + 2 * np.pi, 2 * np.pi) - np.pi])
    out = {"seed": seed}
    if not rt.gait_travel(STOP_AT, arrive=0.05):
        return {**out, "fell": "walk" if rt.fallen() else "timeout"}
    x0 = rt.base_xy.copy()
    v0 = rt.loco.command.copy()
    tick = rt.loco.n_substeps * rt.dt
    for k in range(1, 31):                       # 0.6 s ramp
        rt.loco.set_command(*(v0 * (1 - k / 30)))
        rt.step(tick)
    speeds = []
    for k in range(200):                         # 4 s at zero
        rt.step(tick)
        speeds.append(float(np.linalg.norm(d.qvel[:2])))
        if rt.fallen():
            return {**out, "fell": "stopping", "t": k * tick}
    out.update(speed_2s=float(np.mean(speeds[95:105])),
               speed_4s=float(np.mean(speeds[190:])),
               travel_after_stop=float(np.linalg.norm(rt.base_xy - x0)))
    x1 = rt.base_xy.copy()
    out["handover"] = rt.gait_stop(ramp=0.02, timed=True)
    for k in range(150):
        rt.step(0.02)
        if rt.fallen():
            return {**out, "fell": "after handover", "t": k * 0.02}
    out.update(fell=None, drift=float(np.linalg.norm(rt.base_xy - x1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=None)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = [700 + 13 * k for k in range(a.trials)]
    with Pool(10, maxtasksperchild=4) as pool:
        rows = pool.map(trial, [(s, a.policy) for s in seeds])
    n = len(rows)
    walk = sum(r["fell"] in ("walk", "timeout", "stopping") for r in rows)
    after = sum(r["fell"] == "after handover" for r in rows)
    ok = [r for r in rows if "speed_2s" in r]
    still = sum(r["speed_2s"] < 0.05 for r in ok)
    ms = lambda k: (f"{np.mean([r[k] for r in ok]):.3f} +/- "
                    f"{np.std([r[k] for r in ok]):.3f}") if ok else "n/a"
    print(f"policy {a.policy or 'g1_policy.onnx'}")
    print(f"fell walking or stopping {walk}/{n} | still (<0.05 m/s) at 2 s "
          f"{still}/{len(ok)} | speed 2 s {ms('speed_2s')} | speed 4 s "
          f"{ms('speed_4s')} | travel after stop {ms('travel_after_stop')} m "
          f"| fell after handover {after}/{len(ok)}")
    passed = walk == 0 and still >= 0.9 * n and after <= 1
    print("PASS" if passed else "fail")
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(rows, indent=1, default=str))


if __name__ == "__main__":
    main()
