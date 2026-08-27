"""Milestone 6, completed: walk over and fetch it, balancing the whole way.

The robot stands on its own two legs throughout. Nothing is welded to the
floor, no base motion is scripted, and every step is the result of a learned
policy reacting to real contact and IMU feedback. The block starts at the far
end of the bench, out of the arm's reach, so getting it requires walking.

Two learned controllers share the legs:

  walking  : a pretrained joystick policy (MuJoCo Playground, MJX/PPO) —
             the good gait, shown travelling in demos/07
  stance   : trained in this repo (training/train_stance.py) to hold the
             pelvis on a mark *while the arms reach*, which the walking
             policy cannot do. Moving its hold point a stride at a time also
             makes it a precise short-range positioner — it lands within a
             couple of centimetres, where the walking policy stops a few
             tenths of a metre out, and that difference is what decides
             whether an object is inside the arm's envelope.
"""

import pathlib
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from bodies.body import Body  # noqa: E402
from sim.cameras import CameraRig  # noqa: E402
from sim.world import World  # noqa: E402
from skills.registry import execute  # noqa: E402
from skills.runtime import Runtime  # noqa: E402

RECORD_FPS = 10
PLAY_FPS = 20
WORK_SPOT = np.array([0.0, -0.30])     # where the block can be reached from
HOME_SPOT = np.array([0.0, 0.0])


def caption(frame, lines):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=15)
    except TypeError:
        font = ImageFont.load_default()
    h = 8 + 19 * len(lines)
    draw.rectangle([0, frame.shape[0] - h, frame.shape[1], frame.shape[0]],
                   fill=(0, 0, 0))
    for i, (text, color) in enumerate(lines):
        draw.text((8, frame.shape[0] - h + 3 + 19 * i), text, fill=color,
                  font=font)
    return np.asarray(img)


def main() -> None:
    body = Body(REPO / "bodies" / "g1_walk" / "config.yaml")
    world = World(body)
    rt = Runtime(world)
    rig = CameraRig(world.model, width=620, height=440)
    start = world.object_pos("obj_green_block").copy()

    state = {"frames": [], "next_t": 0.0, "label": "standing"}

    def on_frame(t):
        if t >= state["next_t"]:
            z = world.data.xpos[rt.root_id][2]
            frame = rig.third_person(world.data, azimuth=125, distance=2.3,
                                     lookat=(0.28, -0.22, 0.80), elevation=-12)
            state["frames"].append(caption(frame, [
                (state["label"], (120, 220, 255)),
                (f"balancing on two legs — pelvis {z:.2f} m", (225, 225, 225)),
            ]))
            state["next_t"] += 1.0 / RECORD_FPS

    rt.on_frame = on_frame
    reach0 = rt.to_base(start)
    print(f"block starts {reach0[0]:.2f} m ahead, {abs(reach0[1]):.2f} m to the "
          f"side — beyond the {0.43:.2f} m the arm can reach")

    state["label"] = "walking along the bench to the block"
    walked = rt.step_to(WORK_SPOT, yaw=0.0, tol=0.06)
    print(f"walk to the block : {walked}  base {rt.base_xy.round(3)}")

    state["label"] = "picking it up"
    got = execute("grasp", rt, world, object="green block")
    print(f"grasp             : {got}")

    state["label"] = "carrying it back"
    back = rt.step_to(HOME_SPOT, yaw=0.0, tol=0.06)
    print(f"walk back         : {back}  base {rt.base_xy.round(3)}")

    state["label"] = "putting it down"
    put = execute("place", rt, world,
                  target=[0.33, -0.20, world.table_top + 0.03])
    print(f"place             : {put}")

    state["label"] = "done"
    rt.step(1.5)

    end = world.object_pos("obj_green_block")
    carried = float(np.linalg.norm(end[:2] - start[:2]))
    upright = bool(world.data.xpos[rt.root_id][2] > 0.55)
    fetched = bool(got.get("ok") and put.get("ok") and end[2] > 0.80 and upright)

    out = REPO / "demos" / "media" / "08_walk_and_fetch.gif"
    imageio.mimsave(out, state["frames"], fps=PLAY_FPS, loop=0)
    print(f"\nblock moved {carried:.2f} m along the bench")
    print(f"fetched and set down: {fetched}")
    print(f"still standing:       {upright}")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, "
          f"{len(state['frames'])} frames)")


if __name__ == "__main__":
    main()
