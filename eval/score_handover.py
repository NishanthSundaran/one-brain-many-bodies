"""Walk-to-stance handover, measured on its own. Gate 3a of the walking plan.

The stance-to-walk handover has never worked. This measures the other
direction, which a fetch from across the room needs exactly once: walk 2 m
under the walking policy, ease the command to zero over T seconds, give the
legs to the stance policy, and watch for 3 s.

    coast  how far the robot travels between deciding to stop and the
           stance policy taking over (it creeps; zero does not stop it)
    drift  how far it moves in the 3 s after the stance policy takes over
    overrun  total travel past the point where stopping was decided; this
           is what has to fit between the stopping point and the bench

Foot lifts use eval/score_stance.py's detector (1 cm), the one behind the
README's 21.5 lifts/s for the stance policy, so the numbers compare.

Pass fixed before running: a condition with no falls after handover and
drift under 0.10 m in every trial.

First run (untimed, 5 trials per ramp T of 0.3, 0.6, 1.0 s): 9 of 15 fell
after handover, all within 1.6 s of it. Survivors were moving ~0.20 m/s at
handover, falls ~0.37 m/s, and handovers with a foot in the air fell. So
the conditions compared now are:

    untimed      hand over as soon as the 0.6 s ramp ends (the first run)
    timed        keep walking at zero command until the body is under
                 0.2 m/s with both feet down (cap 4 s), then hand over
    timed+lead   as timed, with the stance hold point 0.3 s ahead along
                 the velocity, where the robot would come to rest anyway

Ten trials per condition, same seeds for all three.

    python eval/score_handover.py
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
START = (-2.5, 0.0)
STOP_AT = np.array([-0.5, 0.0])     # 2 m walk, well short of the bench
CONDITIONS = {
    "untimed": dict(ramp=0.6, timed=False),
    "timed": dict(ramp=0.6, timed=True),
    "timed+lead": dict(ramp=0.6, timed=True, lead=0.3),
    "brake0.5": dict(ramp=0.6, timed=True, brake=0.5),
    "brake1.0": dict(ramp=0.6, timed=True, brake=1.0),
    "brake1.0 slow": dict(ramp=0.6, timed=True, brake=1.0, max_speed=0.1),
    "timed+still arms": dict(ramp=0.6, timed=True, still_arms=True),
    "untimed+still arms": dict(ramp=0.6, timed=False, still_arms=True),
}
WATCH_S = 3.0
JITTER = 0.02
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


def foot_lifts(hist, dt):
    H = np.array(hist)
    lifts = sum(int(np.sum(np.diff((H[:, k] > H[:, k].min() + 0.01)
                                   .astype(int)) == 1)) for k in (0, 1))
    return lifts / (len(H) * dt)


def trial(args):
    cond, seed, stance, catch = (*args, None, None)[:4]
    rng = np.random.default_rng(seed)
    body = Body(CFG)
    if stance:
        body.cfg["locomotion"]["stance_policy"] = stance
    if catch:
        body.cfg["locomotion"]["catch_policy"] = catch
    world = World(body, start_xy=START)
    m, d = world.model, world.data
    for n in POLICY_JOINTS:
        d.qpos[m.joint(n).qposadr[0]] += rng.uniform(-JITTER, JITTER)
    mujoco.mj_forward(m, d)
    rt = Runtime(world, start="walk")
    p = rng.uniform(-np.pi, np.pi)
    rt.loco.phase = np.array([p, np.fmod(p + 2 * np.pi, 2 * np.pi) - np.pi])
    out = {"cond": cond, "seed": seed}
    if rt.fallen():
        return {**out, "fell": "start"}
    if not rt.gait_travel(STOP_AT, arrive=0.30):
        return {**out, "fell": "travel" if rt.fallen() else None,
                "arrived": False}
    heading = np.array([np.cos(rt.base_yaw), np.sin(rt.base_yaw)])
    xy_decide = rt.base_xy.copy()
    info = rt.gait_stop(**CONDITIONS[cond])
    out.update(info)
    xy_hand = rt.base_xy.copy()
    if rt.fallen():
        return {**out, "fell": "ramp", "arrived": True}
    feet = [m.body(f).id for f in FEET]
    hist, fell = [], None
    tick = 0.02
    for i in range(int(WATCH_S / tick)):
        rt.step(tick)
        hist.append([d.xpos[f][2] for f in feet])
        if rt.fallen():
            fell = "stance"
            break
    return {**out, "arrived": True, "fell": fell,
            "coast": float((xy_hand - xy_decide) @ heading),
            "drift": float(np.linalg.norm(rt.base_xy - xy_hand)),
            "overrun": float((rt.base_xy - xy_decide) @ heading),
            "lifts_per_s": foot_lifts(hist, tick) if len(hist) > 10 else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--conds", default=",".join(CONDITIONS))
    ap.add_argument("--stance", default=None,
                    help="stance checkpoint, relative to the repo")
    ap.add_argument("--catch", default=None,
                    help="catch checkpoint, relative to the repo")
    a = ap.parse_args()
    conds = a.conds.split(",")
    seeds = [500 + 17 * k for k in range(a.trials)]
    jobs = [(c, sd, a.stance, a.catch) for c in conds for sd in seeds]
    with Pool() as pool:
        rows = pool.map(trial, jobs)
    print("| condition | fell walking | fell after handover | speed at handover "
          "| waited | drift (3 s) | overrun | foot lifts/s |")
    print("|---|---|---|---|---|---|---|---|")
    passing = []
    for c in conds:
        rs = [x for x in rows if x["cond"] == c]
        walk = sum(x.get("fell") in ("start", "travel") for x in rs)
        after = [x for x in rs if x.get("arrived")]
        fell_after = sum(x.get("fell") is not None for x in after)
        ok = [x for x in after if x.get("fell") is None]

        def ms(k, xs=ok):
            v = np.array([x[k] for x in xs if x.get(k) is not None])
            return f"{v.mean():.2f} +/- {v.std():.2f}" if len(v) else "n/a"
        drift = max((x["drift"] for x in ok), default=np.inf)
        if fell_after == 0 and len(ok) == len(after) and drift < 0.10:
            passing.append(c)
        print(f"| {c} | {walk}/{len(rs)} | {fell_after}/{len(after)} "
              f"| {ms('speed', after)} m/s | {ms('waited', after)} s "
              f"| {ms('drift')} m | {ms('overrun')} m | {ms('lifts_per_s')} |")
    print("\nGate 3a:", "PASS: " + ", ".join(passing) if passing else "fail")
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(rows, indent=1, default=str))


if __name__ == "__main__":
    main()
