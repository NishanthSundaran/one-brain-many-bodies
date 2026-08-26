"""Milestone 3: deterministic skills. Scripted sequence (no brain yet):
right arm drops the green block into the bin, left arm parks the red block
next to it, then both arms home. Records a 2x-speed GIF.
"""

import pathlib
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from bodies.body import Body
from sim.cameras import CameraRig
from sim.world import World
from skills.registry import execute
from skills.runtime import Runtime

RECORD_FPS = 10   # captured sim frames per second
PLAY_FPS = 20     # playback speed (2x)


class Recorder:
    def __init__(self, rig):
        self.rig, self.frames, self.next_t, self.label = rig, [], 0.0, ""

    def __call__(self, t):
        if t >= self.next_t:
            frame = self.rig.third_person(self.data, azimuth=-155, distance=1.7,
                                          lookat=(0.3, 0.0, 0.8), elevation=-18)
            self.frames.append(caption(frame, self.label))
            self.next_t += 1.0 / RECORD_FPS


def caption(frame, text):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=16)
    except TypeError:
        font = ImageFont.load_default()
    draw.rectangle([0, frame.shape[0] - 26, frame.shape[1], frame.shape[0]],
                   fill=(0, 0, 0))
    draw.text((8, frame.shape[0] - 22), text, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def main() -> None:
    body = Body(REPO / "bodies" / "g1" / "config.yaml")
    world = World(body)
    rt = Runtime(world)
    rig = CameraRig(world.model, width=560, height=400)

    rec = Recorder(rig)
    rec.data = world.data
    rt.on_frame = rec

    script = [
        ("grasp", {"object": "green block"}, 'grasp("green block")'),
        ("place", {"target": "bin"}, 'place("bin")'),
        ("grasp", {"object": "red block"}, 'grasp("red block")'),
        ("place", {"target": [0.30, 0.24, 0.75]}, "place([0.30, 0.24, 0.75])"),
        ("home", {}, "home()"),
    ]
    for skill, args, label in script:
        rec.label = label
        res = execute(skill, rt, world, **args)
        print(label, "->", res)

    bb = world.object_pos("obj_bin")
    gp = world.object_pos("obj_green_block")
    in_bin = bool(abs(gp[0] - bb[0]) < 0.07 and abs(gp[1] - bb[1]) < 0.07
                  and 0.72 < gp[2] < 0.82)
    out = REPO / "demos" / "media" / "04_pick_place.gif"
    imageio.mimsave(out, rec.frames, fps=PLAY_FPS, loop=0)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, "
          f"{len(rec.frames)} frames), green block in bin: {in_bin}")


if __name__ == "__main__":
    main()
