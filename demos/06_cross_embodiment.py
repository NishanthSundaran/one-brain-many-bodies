"""Milestone 5 — the headline: ONE brain, MANY bodies.

The identical brain (same local VLM, same system prompt, same tool schemas,
same agent loop) executes the same natural-language command on two different
humanoids: the Unitree G1 (29 DoF, position actuators, 7-DoF wristed arms)
and the Unitree H1 (19 DoF, torque motors normalized by the body layer,
4-DoF wrist-less arms, taller workspace). Only bodies/<name>/config.yaml
differs. Produces a split-screen GIF.
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
COMMAND = "Put the green block in the bin."
W, H = 480, 380


def caption(frame, lines):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=14)
    except TypeError:
        font = ImageFont.load_default()
    h = 10 + 18 * len(lines)
    draw.rectangle([0, frame.shape[0] - h, frame.shape[1], frame.shape[0]],
                   fill=(0, 0, 0))
    for i, (text, color) in enumerate(lines):
        draw.text((8, frame.shape[0] - h + 4 + 18 * i), text,
                  fill=color, font=font)
    return np.asarray(img)


def run_body(name: str) -> tuple[list, list]:
    body = Body(REPO / "bodies" / name / "config.yaml")
    world = World(body)
    rt = Runtime(world)
    rig = CameraRig(world.model, width=W, height=H)
    agent = Agent(rt, world, rig)

    lookat_z = world.table_top + 0.15
    state = {"frames": [], "next_t": 0.0,
             "lines": [(f"Unitree {name.upper()}", (120, 220, 255))]}

    def on_frame(t):
        if t >= state["next_t"]:
            frame = rig.third_person(world.data, azimuth=-150, distance=1.9,
                                     lookat=(0.3, 0.0, lookat_z), elevation=-15)
            state["frames"].append(caption(frame, state["lines"][-2:]))
            state["next_t"] += 1.0 / RECORD_FPS

    def on_step(entry):
        state["lines"].append(
            (f"{entry['tool']}({json.dumps(entry['args'])})"[:64],
             (255, 255, 255)))

    rt.on_frame = on_frame
    transcript = agent.run(COMMAND, on_step=on_step)
    on_frame(state["next_t"])

    bb = world.object_pos("obj_bin")
    gp = world.object_pos("obj_green_block")
    success = bool(abs(gp[0] - bb[0]) < 0.07 and abs(gp[1] - bb[1]) < 0.07
                   and bb[2] < gp[2] < bb[2] + 0.12)
    print(f"[{name}] steps: {len(transcript)}, success: {success}")
    for e in transcript:
        print(f"  [{e['step']}] {e['tool']}({json.dumps(e['args'])}) -> "
              f"{json.dumps(e.get('result'))}")
    rig.close()
    return state["frames"], success


def main() -> None:
    frames_g1, ok_g1 = run_body("g1")
    frames_h1, ok_h1 = run_body("h1")

    n = max(len(frames_g1), len(frames_h1))
    frames_g1 += [frames_g1[-1]] * (n - len(frames_g1))
    frames_h1 += [frames_h1[-1]] * (n - len(frames_h1))
    combined = [np.hstack([a, b]) for a, b in zip(frames_g1, frames_h1)]

    out = REPO / "demos" / "media" / "06_cross_embodiment.gif"
    imageio.mimsave(out, combined, fps=PLAY_FPS, loop=0)
    print(f"\ncommand: {COMMAND}")
    print(f"G1 success: {ok_g1} | H1 success: {ok_h1}")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {n} frames)")


if __name__ == "__main__":
    main()
