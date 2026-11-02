"""Run the privileged expert on seeded tasks in physics and report its success
rate per family: the ceiling for any planner on this task set.

Usage: python posttrain/run_expert.py --seeds 0 40 [--out results.jsonl]
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import signal
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from bodies.body import Body  # noqa: E402
from posttrain.tasks import check, expert, sample_task, snapshot  # noqa: E402
from sim.world import World  # noqa: E402
from skills.registry import execute  # noqa: E402
from skills.runtime import Runtime  # noqa: E402

MAX_STEPS = 10
EPISODE_LIMIT_S = 90   # wall clock; a stuck servo loop must not stall a whole run


class EpisodeTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise EpisodeTimeout


def build(task: dict):
    body = Body(REPO / "bodies" / "g1" / "config.yaml")
    world = World(body, objects=task["objects"])
    return world, Runtime(world)


def run_episode(task: dict) -> dict:
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(EPISODE_LIMIT_S)
    t0 = time.time()
    try:
        return _run(task) | {"wall_s": round(time.time() - t0, 1)}
    except EpisodeTimeout:
        return {"seed": task["seed"], "family": task["family"], "command": task["command"],
                "success": False, "why": f"timeout after {EPISODE_LIMIT_S}s", "n_calls": -1,
                "calls": [], "wall_s": EPISODE_LIMIT_S}
    finally:
        signal.alarm(0)


def _run(task: dict) -> dict:
    world, rt = build(task)
    start = snapshot(world)
    calls = []
    for _ in range(MAX_STEPS):
        call = expert(task, world, rt)
        if call["tool"] == "done":
            calls.append({**call, "result": {"ok": True}})
            break
        result = execute(call["tool"], rt, world, **call["args"])
        calls.append({**call, "result": result})
    verdict = check(task, world, start, rt)
    return {"seed": task["seed"], "family": task["family"], "command": task["command"],
            **verdict, "n_calls": len(calls), "calls": calls}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs=2, default=[0, 40], help="half-open range")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    rows, t0 = [], time.time()
    for seed in range(*a.seeds):
        r = run_episode(sample_task(seed))
        rows.append(r)
        flag = "ok  " if r["success"] else "FAIL"
        print(f"{flag} seed {seed:4d} {r['family']:8s} {r['n_calls']} calls {r['wall_s']:5.1f}s  {r['command']}"
              + ("" if r["success"] else f"  -> {r['why']}"), flush=True)
    by = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        by[r["family"]][0] += r["success"]
        by[r["family"]][1] += 1
    print({f: f"{s}/{n}" for f, (s, n) in sorted(by.items())},
          f"total {sum(r['success'] for r in rows)}/{len(rows)}", f"{time.time() - t0:.0f}s")
    if a.out:
        with open(a.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
