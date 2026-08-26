"""Milestone 6: physics-based locomotion.

The G1 walks on its own two legs. A pretrained RL policy (MJX/PPO, from
MuJoCo Playground) runs in closed loop at 50 Hz on the real IMU and joint
state, and a pursuit controller feeds it velocity commands to reach each
waypoint. Nothing about the base is scripted: no glide, no animation, no
weld. Every step is contact and balance, and a bad command makes it fall.

Also shown: station keeping. The policy is a *locomotion* policy — told to
hold still it creeps forward — so standing on a spot is its own closed-loop
problem, handled by a PI controller on the base pose.
"""

import pathlib
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import mujoco  # noqa: E402

from bodies.body import Body  # noqa: E402
from sim.cameras import CameraRig  # noqa: E402
from skills.runtime import Runtime  # noqa: E402

RECORD_FPS = 10
PLAY_FPS = 20

WAYPOINTS = [
    ((1.6, 0.0), "walk forward"),
    ((1.6, 1.0), "turn and walk left"),
    ((0.0, 0.6), "walk back"),
]


class OpenWorld:
    """Bare floor: this demo is about locomotion, not tabletops."""

    def __init__(self, body):
        self.body = body
        w = body.spec.worldbody
        w.add_light(pos=[2, -2, 3], dir=[-0.4, 0.4, -1], diffuse=[0.65] * 3,
                    ambient=[0.35] * 3)
        for i in range(-3, 4):   # stripes so the motion is visible
            w.add_geom(name=f"stripe{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                       size=[0.02, 4.0, 0.001], pos=[i * 0.5, 0, 0.001],
                       rgba=[0.42, 0.46, 0.52, 1])
        w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                   size=[8, 8, 0.1], rgba=[0.32, 0.36, 0.42, 1])
        self.model = body.spec.compile()
        self.data = mujoco.MjData(self.model)
        body.bind(self.model)
        body.reset(self.data)
        mujoco.mj_forward(self.model, self.data)

    def step(self, seconds):
        for _ in range(int(round(seconds / self.model.opt.timestep))):
            mujoco.mj_step(self.model, self.data)


def caption(frame, lines):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=15)
    except TypeError:
        font = ImageFont.load_default()
    h = 10 + 19 * len(lines)
    draw.rectangle([0, frame.shape[0] - h, frame.shape[1], frame.shape[0]],
                   fill=(0, 0, 0))
    for i, (text, color) in enumerate(lines):
        draw.text((8, frame.shape[0] - h + 4 + 19 * i), text, fill=color,
                  font=font)
    return np.asarray(img)


def main() -> None:
    body = Body(REPO / "bodies" / "g1_walk" / "config.yaml")
    world = OpenWorld(body)
    rt = Runtime(world)
    rig = CameraRig(world.model, width=600, height=420)

    state = {"frames": [], "next_t": 0.0, "label": "standing (station keeping)"}

    def on_frame(t):
        if t >= state["next_t"]:
            x, y = rt.base_xy
            z = world.data.xpos[rt.root_id][2]
            frame = rig.third_person(world.data, azimuth=138, distance=3.4,
                                     lookat=(x, y, 0.75), elevation=-14)
            state["frames"].append(caption(frame, [
                (state["label"], (120, 220, 255)),
                (f"base ({x:+.2f}, {y:+.2f}) m   pelvis {z:.2f} m",
                 (235, 235, 235)),
            ]))
            state["next_t"] += 1.0 / RECORD_FPS

    rt.on_frame = on_frame
    rt.step(1.5)

    travelled, t0 = 0.0, rt._t
    prev = rt.base_xy.copy()
    for (x, y), label in WAYPOINTS:
        state["label"] = label
        rt.walk_to(x, y)
        travelled += float(np.linalg.norm(rt.base_xy - prev))
        prev = rt.base_xy.copy()
        err = float(np.linalg.norm(rt.base_xy - np.array([x, y])))
        print(f"{label:22s} -> base {rt.base_xy.round(2)}  "
              f"(waypoint error {err:.2f} m), pelvis "
              f"{world.data.xpos[rt.root_id][2]:.3f} m")

    state["label"] = "standing (station keeping)"
    rt.step(2.0)                       # let the hold converge
    hold_start = rt.base_xy.copy()
    rt.step(3.0)
    drift = float(np.linalg.norm(rt.base_xy - hold_start))
    upright = world.data.xpos[rt.root_id][2] > 0.6

    out = REPO / "demos" / "media" / "07_physics_walk.gif"
    imageio.mimsave(out, state["frames"], fps=PLAY_FPS, loop=0)
    print(f"\ndistance walked: {travelled:.2f} m in {rt._t - t0:.1f} s "
          f"({travelled / max(rt._t - t0, 1e-6):.2f} m/s average)")
    print(f"drift while standing for 3 s: {drift:.3f} m")
    print(f"upright at the end: {upright}")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, "
          f"{len(state['frames'])} frames)")


if __name__ == "__main__":
    main()
