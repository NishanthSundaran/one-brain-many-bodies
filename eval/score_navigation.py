"""Can the walking policy be steered to a point? Gate 2 of the walking plan.

Drives the borrowed walking policy with controllers/navigator.py on open
floor, five jittered trials per target. A trial ends when the pelvis comes
within ARRIVE of the target (arrived), the robot falls, or TIMEOUT passes.
Stopping is deliberately not part of this test: the policy cannot stop
itself (commanded zero it still walks ~0.3 m/s), so arrival hands over to
the stance policy, and that handover is measured separately.

Targets are placed relative to the robot's pose after a 1.5 s settle, so
"behind" means behind the robot, wherever its creep has carried it.

Pass thresholds were fixed before running: at least 4/5 arrivals and no
falls on every straight and 90 degree target, at least 3/5 on the target
behind. For comparison, the stance-policy navigation the project ships
arrives at 0.30 m and falls at 0.50 m and beyond (README).

    python eval/score_navigation.py
"""

import argparse
import importlib.util
import json
import pathlib
import sys
from multiprocessing import Pool

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import mujoco  # noqa: E402

from controllers.locomotion import POLICY_JOINTS, LocomotionPolicy  # noqa: E402
from controllers.navigator import GaitNavigator  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "score_gait_commands", REPO / "eval" / "score_gait_commands.py")
sg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sg)

TARGETS = {                      # robot frame after settling, metres
    "straight 1 m": (1.0, 0.0),
    "straight 2 m": (2.0, 0.0),
    "straight 3 m": (3.0, 0.0),
    "left 90, 2 m": (0.0, 2.0),
    "right 90, 2 m": (0.0, -2.0),
    "behind, 2 m": (-2.0, 0.0),
}
ARRIVE = 0.30
TIMEOUT = 30.0


def trial(args):
    name, seed = args
    rng = np.random.default_rng(seed)
    body, model, data = sg.build("A")
    body.reset(data)
    for n in POLICY_JOINTS:
        data.qpos[model.joint(n).qposadr[0]] += rng.uniform(-sg.JITTER, sg.JITTER)
    mujoco.mj_forward(model, data)
    cfg = body.cfg["locomotion"]
    pol = LocomotionPolicy(model, REPO / cfg["policy"], cfg["stand_angles"],
                           cfg["ctrl_dt"], cfg["action_scale"],
                           gait_freq=cfg["gait_freq"])
    p = rng.uniform(-np.pi, np.pi)
    pol.phase = np.array([p, np.fmod(p + 2 * np.pi, 2 * np.pi) - np.pi])
    dt = model.opt.timestep
    pelvis = model.body("pelvis").id

    def pose():
        R = data.xmat[pelvis].reshape(3, 3)
        return data.xpos[pelvis][:2].copy(), float(np.arctan2(R[1, 0], R[0, 0]))

    pol.set_command(0, 0, 0)
    for _ in range(int(sg.SETTLE_S / dt)):
        pol.step(data)
        mujoco.mj_step(model, data)
    xy, yaw = pose()
    off = np.array(TARGETS[name])
    c, s = np.cos(yaw), np.sin(yaw)
    target = xy + np.array([c * off[0] - s * off[1], s * off[0] + c * off[1]])

    nav = GaitNavigator()
    n_sub = pol.n_substeps
    fell_at, arrived_at, best, path = None, None, np.inf, 0.0
    prev = xy
    for i in range(int(TIMEOUT / dt)):
        if i % n_sub == 0:
            xy, yaw = pose()
            vx, wz = nav.command(xy, yaw, target, n_sub * dt)
            pol.set_command(vx, 0.0, wz)
        pol.step(data)
        mujoco.mj_step(model, data)
        xy = data.xpos[pelvis][:2]
        path += float(np.linalg.norm(xy - prev))
        prev = xy.copy()
        d = float(np.linalg.norm(target - xy))
        best = min(best, d)
        if sg.fallen(data, pelvis):
            fell_at = (i + 1) * dt
            break
        if d < ARRIVE:
            arrived_at = (i + 1) * dt
            break
    return {"target": name, "seed": seed, "arrived_at": arrived_at,
            "fell_at": fell_at, "closest": best,
            "path_over_direct": path / float(np.linalg.norm(off))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    jobs = [(n, 7000 + 13 * k + 101 * i)
            for i, n in enumerate(TARGETS) for k in range(a.trials)]
    with Pool() as pool:
        rows = pool.map(trial, jobs)
    print("| target | arrived | falls | time to arrive | closest | path / direct |")
    print("|---|---|---|---|---|---|")
    verdict = {}
    for n in TARGETS:
        rs = [r for r in rows if r["target"] == n]
        arr = [r for r in rs if r["arrived_at"] is not None]
        falls = sum(r["fell_at"] is not None for r in rs)
        t = np.array([r["arrived_at"] for r in arr]) if arr else np.array([np.nan])
        po = np.array([r["path_over_direct"] for r in arr]) if arr else np.array([np.nan])
        need = 3 if n.startswith("behind") else 4
        verdict[n] = len(arr) >= need and (falls == 0 or n.startswith("behind"))
        print(f"| {n} | {len(arr)}/{len(rs)} | {falls}/{len(rs)} "
              f"| {np.nanmean(t):.1f} +/- {np.nanstd(t):.1f} s "
              f"| {np.mean([r['closest'] for r in rs]):.2f} m "
              f"| {np.nanmean(po):.2f} |")
    print("\nGate 2:", "PASS" if all(verdict.values()) else "fail", verdict)
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps({"rows": rows, "gate": verdict},
                                                  indent=1, default=str))


if __name__ == "__main__":
    main()
