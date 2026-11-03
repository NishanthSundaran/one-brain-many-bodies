"""Planner post-training, side by side: the stock Qwen2.5-VL-3B planner and the
fine-tuned one (SFT + DAgger) on the same held-out task. Both are served by
posttrain/serve_vlm.py (greedy), so they differ only in their weights.

Record each planner (one server at a time fits a 6 GB GPU), then compose:
  python posttrain/serve_vlm.py --greedy                      # base, then:
  python demos/14_finetuned_planner.py record --label base --out /tmp/base.npz
  python posttrain/serve_vlm.py --greedy --adapter CKPT/dagger_r1/final
  python demos/14_finetuned_planner.py record --label tuned --out /tmp/tuned.npz
  python demos/14_finetuned_planner.py compose /tmp/base.npz /tmp/tuned.npz
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from brain.agent_loop import Agent  # noqa: E402
from brain.vlm_client import OllamaVLM  # noqa: E402
from posttrain.run_expert import build  # noqa: E402
from posttrain.tasks import check, sample_task, snapshot  # noqa: E402
from sim.cameras import CameraRig  # noqa: E402

RECORD_FPS = 10
PLAY_FPS = 20
W, H = 560, 400   # the eval camera size: the planner must see exactly what it saw in eval


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def caption(frame, title, lines, verdict=None):
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 26], fill=(0, 0, 0))
    d.text((8, 5), title, fill=(120, 220, 255), font=_font(15))
    h = 12 + 17 * max(1, len(lines))
    d.rectangle([0, H - h, W, H], fill=(0, 0, 0))
    for i, (text, color) in enumerate(lines):
        d.text((8, H - h + 5 + 17 * i), text[:72], fill=color, font=_font(13))
    if verdict:
        ok = verdict.startswith("success")
        d.rectangle([0, 30, W, 58], fill=(20, 110, 40) if ok else (150, 30, 30))
        d.text((8, 36), verdict, fill=(255, 255, 255), font=_font(15))
    return np.asarray(img)


def record(a) -> None:
    task = sample_task(a.seed)
    world, rt = build(task)
    rig = CameraRig(world.model, width=W, height=H)
    agent = Agent(rt, world, rig, vlm=OllamaVLM(host=a.host), context="last")
    start = snapshot(world)
    title = f"{a.label}: {task['command']}"
    st = {"frames": [], "next_t": 0.0, "lines": []}

    def on_frame(t):
        if t >= st["next_t"]:
            f = rig.third_person(world.data, azimuth=-160, distance=1.5,
                                 lookat=(0.3, 0.0, 0.78), elevation=-30)
            st["frames"].append(caption(f, title, st["lines"][-3:]))
            st["next_t"] += 1.0 / RECORD_FPS

    def on_step(entry):
        args = entry["args"]
        if entry["tool"] == "done":
            text = "done"
        else:
            tgt = args.get("target")
            tgt = "middle" if isinstance(tgt, list) and abs(tgt[1]) < 0.03 else tgt
            parts = [str(p) for p in (args.get("object") or tgt,) if p]
            parts += [f"{args['arm']} arm"] if args.get("arm") else ["no arm given"]
            text = f"{entry['tool']}({', '.join(parts)})"
        st["lines"].append((f"[{len(st['lines']) + 1}] {text}", (255, 255, 255)))

    rt.on_frame = on_frame
    transcript = agent.run(task["command"], max_steps=10, on_step=on_step)
    rt.on_frame = None
    rig_frame = rig.third_person(world.data, azimuth=-160, distance=1.5,
                                 lookat=(0.3, 0.0, 0.78), elevation=-30)
    v = check(task, world, start, rt)
    fails = sum(1 for e in transcript if e["tool"] != "done" and not (e.get("result") or {}).get("ok"))
    verdict = ("success" if v["success"] else f"failed: {v['why']}") + f"  ({fails} failed calls)"
    # mark failed calls in the caption history, then hold the verdict for 2 s
    for i, e in enumerate(transcript[: len(st["lines"])]):
        if e["tool"] != "done" and not (e.get("result") or {}).get("ok"):
            st["lines"][i] = (st["lines"][i][0] + "  FAILED", (255, 120, 120))
    st["frames"] += [caption(rig_frame, title, st["lines"][-3:], verdict)] * (2 * PLAY_FPS)
    rig.close()
    np.savez_compressed(a.out, frames=np.stack(st["frames"]))
    print(verdict, len(st["frames"]), "frames")
    print(json.dumps([(e["tool"], e["args"]) for e in transcript]))


def compose(a) -> None:
    left, right = (np.load(p)["frames"] for p in (a.left, a.right))
    n = max(len(left), len(right))
    pad = lambda x: np.concatenate([x, np.repeat(x[-1:], n - len(x), 0)]) if len(x) < n else x
    left, right = pad(left), pad(right)
    gap = np.full((H, 6, 3), 255, np.uint8)
    # every second frame: plays at 4x real time; 760 px wide keeps the GIF light enough for a README
    frames = [np.asarray(Image.fromarray(np.concatenate([l, gap, r], 1)).resize((760, 270), Image.LANCZOS))
              for l, r in zip(left[::2], right[::2])]
    out = REPO / "demos" / "media" / "14_finetuned_planner.gif"
    imageio.mimsave(out, frames, fps=PLAY_FPS, loop=0)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(frames)} frames)")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--seed", type=int, default=3)
    r.add_argument("--host", default="http://localhost:11500")
    r.add_argument("--label", required=True)
    r.add_argument("--out", required=True)
    c = sub.add_parser("compose")
    c.add_argument("left")
    c.add_argument("right")
    a = ap.parse_args()
    record(a) if a.cmd == "record" else compose(a)


if __name__ == "__main__":
    main()
