"""Run the VLM planner (brain/agent_loop.py) on seeded tasks and score it in
physics with the same check as the expert.

Usage: python posttrain/run_agent.py --seeds 0 20 --out runs/base.jsonl [--model qwen2.5vl:3b]
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

from brain.agent_loop import Agent  # noqa: E402
from brain.vlm_client import OllamaVLM  # noqa: E402
from posttrain.run_expert import EpisodeTimeout, _alarm, build  # noqa: E402
from posttrain.tasks import check, sample_task, snapshot  # noqa: E402
from sim.cameras import CameraRig  # noqa: E402

MAX_STEPS = 10
EPISODE_LIMIT_S = 240


def run_episode(task: dict, vlm) -> dict:
    world, rt = build(task)
    rig = CameraRig(world.model, width=560, height=400)
    agent = Agent(rt, world, rig, vlm=vlm)
    start = snapshot(world)
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(EPISODE_LIMIT_S)
    t0 = time.time()
    why = None
    try:
        transcript = agent.run(task["command"], max_steps=MAX_STEPS)
    except EpisodeTimeout:
        transcript, why = agent.transcript, f"timeout after {EPISODE_LIMIT_S}s"
    finally:
        signal.alarm(0)
    verdict = check(task, world, start, rt) if why is None else {"success": False, "why": why}
    called_done = bool(transcript) and transcript[-1]["tool"] == "done"
    failed_calls = sum(1 for e in transcript if e["tool"] != "done"
                       and not (e.get("result") or {}).get("ok", False))
    return {"seed": task["seed"], "family": task["family"], "command": task["command"],
            **verdict, "called_done": called_done, "n_calls": len(transcript),
            "failed_calls": failed_calls, "wall_s": round(time.time() - t0, 1),
            "vlm_s": round(sum(e.get("vlm_seconds", 0) for e in transcript), 1),
            "transcript": transcript}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs=2, default=[0, 20], help="half-open range")
    p.add_argument("--model", default="qwen2.5vl:3b")
    p.add_argument("--host", default="http://localhost:11434", help="Ollama, or posttrain/serve_vlm.py")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    vlm = OllamaVLM(model=a.model, host=a.host)
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(out, "w") as f:
        for seed in range(*a.seeds):
            r = run_episode(sample_task(seed), vlm)
            rows.append(r)
            f.write(json.dumps(r, default=str) + "\n")
            f.flush()
            flag = "ok  " if r["success"] else "FAIL"
            calls = " ".join(f"{e['tool']}({json.dumps(e['args'], separators=(',', ':'))})" for e in r["transcript"])
            print(f"{flag} seed {seed:4d} {r['family']:8s} {r['wall_s']:5.1f}s  {r['command']}  ::  {calls}"
                  + ("" if r["success"] else f"  -> {r['why']}"), flush=True)
    by = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        by[r["family"]][0] += r["success"]
        by[r["family"]][1] += 1
    print({k: f"{s}/{n}" for k, (s, n) in sorted(by.items())},
          f"total {sum(r['success'] for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
