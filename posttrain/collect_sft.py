"""Collect supervised fine-tuning data: run the real agent loop with the
privileged expert answering in place of the VLM, so every example is exactly
the prompt the planner sees at that step (system prompt, camera image with
markers, marker list, action history, last result) paired with the expert's
call. Only episodes that succeed in physics are kept.

Evaluation uses seeds 0-99; training data comes from seeds 10000 and up.

Usage: python posttrain/collect_sft.py --seeds 10000 10400 --out ~/ml_data/obmb_post/sft
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
from posttrain.run_expert import EpisodeTimeout, _alarm, build  # noqa: E402
from posttrain.tasks import check, expert, sample_task, snapshot  # noqa: E402
from sim.cameras import CameraRig  # noqa: E402

EPISODE_LIMIT_S = 120


class ExpertVLM:
    """Stands in for OllamaVLM: answers with the expert's call and records the prompt."""

    def __init__(self, task, world, rt):
        self.task, self.world, self.rt = task, world, rt
        self.steps: list[dict] = []

    def chat(self, messages, schema=None, timeout=None):
        call = expert(self.task, self.world, self.rt)
        content = json.dumps({"thought": call["thought"], "tool": call["tool"], "args": call["args"]})
        self.steps.append({"messages": copy.deepcopy(messages), "target": content})
        return content, {"seconds": 0.0}


def to_rows(steps: list[dict], img_dir: pathlib.Path, stem: str) -> list[dict]:
    rows = []
    for k, st in enumerate(steps):
        msgs = []
        image = None
        for m in st["messages"]:
            if m.get("images"):
                image = img_dir / f"{stem}_s{k}.png"
                image.write_bytes(m["images"][0])
                msgs.append({"role": m["role"], "content": [{"type": "image"},
                                                            {"type": "text", "text": m["content"]}]})
            else:
                msgs.append({"role": m["role"], "content": [{"type": "text", "text": m["content"]}]})
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": st["target"]}]})
        rows.append({"messages": msgs, "image": str(image), "step": k})
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs=2, required=True, help="half-open range")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = pathlib.Path(a.out).expanduser()
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    stats = collections.Counter()
    with open(out / f"sft_{a.seeds[0]}_{a.seeds[1]}.jsonl", "w") as f:
        for seed in range(*a.seeds):
            task = sample_task(seed)
            world, rt = build(task)
            rig = CameraRig(world.model, width=560, height=400)
            vlm = ExpertVLM(task, world, rt)
            agent = Agent(rt, world, rig, vlm=vlm, context="last")
            start = snapshot(world)
            signal.signal(signal.SIGALRM, _alarm)
            signal.alarm(EPISODE_LIMIT_S)
            try:
                agent.run(task["command"], max_steps=10)
                ok = check(task, world, start, rt)["success"]
            except EpisodeTimeout:
                ok = False
            finally:
                signal.alarm(0)
            stats[(task["family"], ok)] += 1
            if not ok:
                print(f"skip seed {seed} {task['family']}: expert failed in physics", flush=True)
                continue
            for r in to_rows(vlm.steps, img_dir, f"seed{seed}"):
                f.write(json.dumps({"seed": seed, "family": task["family"], **r}) + "\n")
            stats["steps"] += len(vlm.steps)
            print(f"seed {seed} {task['family']}: {len(vlm.steps)} steps", flush=True)
    print(dict(stats))


if __name__ == "__main__":
    main()
