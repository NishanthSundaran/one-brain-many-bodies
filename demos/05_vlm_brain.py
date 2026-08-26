"""Milestone 4: the local VLM brain. A typed natural-language command is
executed by qwen2.5vl:3b (running on-device via Ollama) calling skills on the
G1. Records a GIF with the live tool-call captions and prints the transcript.

Usage: python demos/05_vlm_brain.py ["your command"]
"""

import json
import pathlib
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from bodies.body import Body
from brain.agent_loop import Agent
from sim.cameras import CameraRig
from sim.world import World
from skills.runtime import Runtime

RECORD_FPS = 10
PLAY_FPS = 20
DEFAULT_CMD = ("Put the green block in the bin, then put the red block "
               "next to the blue cup.")


def caption(frame, lines):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=14)
    except TypeError:
        font = ImageFont.load_default()
    h = 20 + 18 * len(lines)
    draw.rectangle([0, frame.shape[0] - h, frame.shape[1], frame.shape[0]],
                   fill=(0, 0, 0))
    for i, (text, color) in enumerate(lines):
        draw.text((8, frame.shape[0] - h + 6 + 18 * i), text,
                  fill=color, font=font)
    return np.asarray(img)


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CMD
    body = Body(REPO / "bodies" / "g1" / "config.yaml")
    world = World(body)
    rt = Runtime(world)
    rig = CameraRig(world.model, width=560, height=400)
    agent = Agent(rt, world, rig)

    state = {"frames": [], "next_t": 0.0,
             "lines": [(f'> {command}', (120, 220, 255))]}

    def on_frame(t):
        if t >= state["next_t"]:
            frame = rig.third_person(world.data, azimuth=-155, distance=1.7,
                                     lookat=(0.3, 0.0, 0.8), elevation=-18)
            state["frames"].append(caption(frame, state["lines"][-3:]))
            state["next_t"] += 1.0 / RECORD_FPS

    def on_step(entry):
        call = f"[brain] {entry['tool']}({json.dumps(entry['args'])})"
        state["lines"].append((call, (255, 255, 255)))

    rt.on_frame = on_frame
    transcript = agent.run(command, on_step=on_step)
    on_frame(state["next_t"])  # flush a final frame

    print(f"\ncommand: {command}\n")
    for e in transcript:
        print(f"[{e['step']}] ({e['vlm_seconds']:5.2f}s) {e['tool']}"
              f"({json.dumps(e['args'])}) -> {json.dumps(e.get('result'))}")

    out = REPO / "demos" / "media" / "05_vlm_brain.gif"
    imageio.mimsave(out, state["frames"], fps=PLAY_FPS, loop=0)
    print(f"\nwrote {out} ({out.stat().st_size / 1e6:.1f} MB, "
          f"{len(state['frames'])} frames)")


if __name__ == "__main__":
    main()
