"""DAgger round: the fine-tuned planner drives (through serve_vlm.py), the expert
labels every state it reaches. SFT data only shows states on the expert's own
path; once the planner slips (e.g. after a handover) it sees states it was never
taught. Here those states get the expert's answer as the training target, while
the planner's own call is what is executed.

Steps where the expert would call done although the check fails, or where the
check is already unrecoverable (an object off the table), are dropped.

Usage: python posttrain/collect_dagger.py --seeds 30000 30200 --host http://localhost:11500 \
           --out ~/ml_data/obmb_post/dagger
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import pathlib
import signal
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from brain.agent_loop import Agent  # noqa: E402
from brain.vlm_client import OllamaVLM  # noqa: E402
from posttrain.collect_sft import to_rows  # noqa: E402
from posttrain.run_expert import EpisodeTimeout, _alarm, build  # noqa: E402
from posttrain.tasks import check, expert, sample_task, snapshot  # noqa: E402
from sim.cameras import CameraRig  # noqa: E402

EPISODE_LIMIT_S = 300


class LabelledPolicy:
    """The policy answers; the expert's call for the same state is recorded as the target."""

    def __init__(self, policy, task, world, rt, start):
        self.policy, self.task, self.world, self.rt, self.start = policy, task, world, rt, start
        self.steps: list[dict] = []
        self.disagree = 0

    def chat(self, messages, schema=None, timeout=None):
        call = expert(self.task, self.world, self.rt)
        target = json.dumps({"thought": call["thought"], "tool": call["tool"], "args": call["args"]})
        content, stats = self.policy.chat(messages, schema=schema)
        verdict = check(self.task, self.world, self.start, self.rt)
        bad_done = call["tool"] == "done" and not verdict["success"]
        if not bad_done:
            self.steps.append({"messages": copy.deepcopy(messages), "target": target})
        try:
            mine = json.loads(content)
            same = (mine.get("tool"), mine.get("args")) == (call["tool"], call["args"])
        except json.JSONDecodeError:
            same = False
        self.disagree += not same
        return content, stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs=2, required=True)
    p.add_argument("--host", default="http://localhost:11500")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = pathlib.Path(a.out).expanduser()
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    policy = OllamaVLM(host=a.host)
    stats = collections.Counter()
    with open(out / f"dagger_{a.seeds[0]}_{a.seeds[1]}.jsonl", "w") as f:
        for seed in range(*a.seeds):
            task = sample_task(seed)
            world, rt = build(task)
            rig = CameraRig(world.model, width=560, height=400)
            start = snapshot(world)
            vlm = LabelledPolicy(policy, task, world, rt, start)
            agent = Agent(rt, world, rig, vlm=vlm, context="last")
            signal.signal(signal.SIGALRM, _alarm)
            signal.alarm(EPISODE_LIMIT_S)
            try:
                agent.run(task["command"], max_steps=10)
                ok = check(task, world, start, rt)["success"]
            except EpisodeTimeout:
                ok = False
            finally:
                signal.alarm(0)
            stats[f"{task['family']}_{'ok' if ok else 'fail'}"] += 1
            stats["steps"] += len(vlm.steps)
            stats["disagree"] += vlm.disagree
            for r in to_rows(vlm.steps, img_dir, f"dagger{seed}"):
                f.write(json.dumps({"seed": seed, "family": task["family"], "policy_ok": ok, **r}) + "\n")
            f.flush()
            print(f"seed {seed} {task['family']:8s} policy {'ok  ' if ok else 'FAIL'} "
                  f"{len(vlm.steps)} labelled, {vlm.disagree} disagreements", flush=True)
    print(dict(stats))


if __name__ == "__main__":
    main()
