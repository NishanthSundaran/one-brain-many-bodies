"""Measure where each G1 arm can actually grasp a block on the bench: one block
at each grid point, grasp with each arm, record success. The task generator
only places objects (and handover spots) where the measurement says it works.

Usage: python posttrain/sweep_reach.py --out posttrain/reach_map.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

XS = np.round(np.arange(0.24, 0.401, 0.02), 3)
YS = np.round(np.arange(-0.30, 0.301, 0.03), 3)


def trial(args):
    x, y, arm, kind = args
    from posttrain.run_expert import build
    from skills.registry import execute
    size = [0.025, 0.025, 0.025] if kind == "box" else [0.03, 0.05]
    task = {"objects": [{"name": "probe", "kind": kind, "size": size,
                         "rgba": [1, 0, 0, 1], "pos": [float(x), float(y)]},
                        {"name": "bin", "kind": "bin", "size": [0.065, 0.065, 0.035],
                         "rgba": [0.5, 0.5, 0.5, 1], "pos": [0.32, 0.60]}]}  # bin out of the way
    world, rt = build(task)
    r = execute("grasp", rt, world, object="probe", arm=arm)
    return (float(x), float(y), arm, kind, bool(r.get("ok")), r.get("error", ""))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    jobs = [(x, y, arm, kind) for kind in ("box", "cylinder") for arm in ("left", "right")
            for x in XS for y in YS]
    with mp.Pool(a.workers) as pool:
        res = pool.map(trial, jobs, chunksize=4)
    json.dump(res, open(a.out, "w"))
    for kind in ("box", "cylinder"):
        for arm in ("left", "right"):
            print(f"\n{kind}, {arm} arm (rows x, cols y {YS[0]}..{YS[-1]}): # = grasp ok")
            ok = {(x, y) for x, y, ar, k, s, _ in res if ar == arm and k == kind and s}
            for x in XS:
                print(f"x={x:.2f} " + "".join("#" if (float(x), float(y)) in ok else "." for y in YS))


if __name__ == "__main__":
    main()
