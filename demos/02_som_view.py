"""Milestone 1: tabletop world + head camera + Set-of-Mark markers.
Records a side-by-side GIF (third-person | annotated head-cam) and prints
the marker map the brain will receive.
"""

import json
import pathlib
import sys

import imageio.v2 as imageio
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from bodies.body import Body
from brain.som import annotate
from sim.cameras import CameraRig
from sim.world import World

FPS = 20
DURATION_S = 4.0


def main() -> None:
    body = Body(REPO / "bodies" / "g1" / "config.yaml")
    world = World(body)
    rig = CameraRig(world.model, width=480, height=360)

    frames = []
    markers = {}
    for _ in range(int(DURATION_S * FPS)):
        world.step(1.0 / FPS)
        rgb = rig.rgb(world.data)
        seg = rig.segmentation(world.data)
        annotated, markers = annotate(rgb, seg, world.model, world.data, world.objects)
        frames.append(np.hstack([rig.third_person(world.data), annotated]))

    out = REPO / "demos" / "media" / "02_som_view.gif"
    imageio.mimsave(out, frames, fps=FPS, loop=0)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print("marker map:", json.dumps(markers, indent=2))


if __name__ == "__main__":
    main()
