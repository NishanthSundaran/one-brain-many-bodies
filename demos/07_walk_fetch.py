"""Milestone 6: walk-and-fetch. The object the brain is asked for sits on a
second table across the room, so the plan must include locomotion: walk over,
pick it up, walk back, drop it in the bin.

Locomotion here is a kinematic base glide with a joint-space gait (see
`Runtime.walk_to`) — honest v1 stand-in for a learned policy.
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
from sim.world import DEFAULT_OBJECTS, World
from skills.runtime import Runtime

RECORD_FPS = 10
PLAY_FPS = 20
COMMAND = "Fetch the yellow cup from the side table and put it in the bin."

SIDE_TABLE = {"name": "side table", "xy": [0.10, -1.25], "half": [0.22, 0.3]}
YELLOW_CUP = {"name": "yellow cup", "kind": "cylinder", "size": [0.03, 0.05],
              "rgba": [0.9, 0.75, 0.1, 1], "pos": [-0.06, -1.05], "z": 0.72}


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


def main() -> None:
    body = Body(REPO / "bodies" / "g1" / "config.yaml")
    world = World(body, objects=list(DEFAULT_OBJECTS) + [YELLOW_CUP],
                  extra_tables=[SIDE_TABLE])
    rt = Runtime(world)
    rig = CameraRig(world.model, width=600, height=420)
    agent = Agent(rt, world, rig)

    state = {"frames": [], "next_t": 0.0,
             "lines": [(f"> {COMMAND}", (120, 220, 255))]}

    def on_frame(t):
        if t >= state["next_t"]:
            # follow the robot so it stays in shot while it walks
            cx, cy = rt.base_xy
            frame = rig.third_person(
                world.data, azimuth=150, distance=3.0,
                lookat=(0.25 + 0.4 * cx, 0.5 * cy - 0.35, 0.8), elevation=-16)
            state["frames"].append(caption(frame, state["lines"][-2:]))
            state["next_t"] += 1.0 / RECORD_FPS

    def on_step(entry):
        state["lines"].append(
            (f"[brain] {entry['tool']}({json.dumps(entry['args'])})"[:78],
             (255, 255, 255)))

    rt.on_frame = on_frame
    transcript = agent.run(COMMAND, max_steps=8, on_step=on_step)
    on_frame(state["next_t"])

    bb = world.object_pos("obj_bin")
    yc = world.object_pos("obj_yellow_cup")
    ok = bool(abs(yc[0] - bb[0]) < 0.08 and abs(yc[1] - bb[1]) < 0.08
              and yc[2] > bb[2])
    print(f"\ncommand: {COMMAND}")
    for e in transcript:
        print(f"[{e['step']}] ({e['vlm_seconds']:5.2f}s) {e['tool']}"
              f"({json.dumps(e['args'])}) -> {json.dumps(e.get('result'))}")
    out = REPO / "demos" / "media" / "07_walk_fetch.gif"
    imageio.mimsave(out, state["frames"], fps=PLAY_FPS, loop=0)
    print(f"\nfetch success: {ok}")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, "
          f"{len(state['frames'])} frames)")


if __name__ == "__main__":
    main()
