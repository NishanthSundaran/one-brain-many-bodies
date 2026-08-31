"""One command, 0.6 m: the shipped stance policy against the travel policy.

Commanded straight to a goal 0.6 m away the shipped policy falls in about a
second. A policy trained with the goal as a *command* reaches it, to within
18 cm, and then falls too, at about 24 s.

This clip runs until each robot is down rather than for a fixed window. An
earlier version recorded exactly 12 s, which the travel policy survives, and
captioned it "upright": the fall was twelve seconds past the end of the clip.
The recording length was doing the arguing.
"""

import pathlib
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from bodies.body import Body                                     # noqa: E402
from controllers.stance import StancePolicy                      # noqa: E402
from sim.cameras import CameraRig                                # noqa: E402
from sim.world import World                                      # noqa: E402
from skills.runtime import Runtime                               # noqa: E402

RECORD_FPS = 10
PLAY_FPS = 20
GOAL = 0.6
MAX_SECONDS = 32.0


def caption(frame, lines):
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=15)
    except TypeError:
        font = ImageFont.load_default()
    h = 8 + 19 * len(lines)
    d.rectangle([0, frame.shape[0] - h, frame.shape[1], frame.shape[0]],
                fill=(0, 0, 0))
    for i, (text, colour) in enumerate(lines):
        d.text((8, frame.shape[0] - h + 3 + 19 * i), text, fill=colour,
               font=font)
    return np.asarray(img)


def run(checkpoint, label, colour):
    world = World(Body(REPO / "bodies" / "g1_walk" / "config.yaml"))
    rt = Runtime(world)
    rt.stance = StancePolicy(world.model,
                             REPO / "third_party" / "policies" / checkpoint,
                             rt.body.cfg["locomotion"]["stand_angles"])
    rt.stand_still()
    rig = CameraRig(world.model, width=520, height=420)
    goal = rt.base_xy + np.array([0.0, -GOAL])
    frames, state = [], {"next": 0.0, "fell_at": None}

    def on_frame(t):
        if t >= state["next"]:
            gap = float(np.linalg.norm(rt.base_xy - goal))
            z = float(world.data.xpos[rt.root_id][2])
            frames.append(caption(
                rig.third_person(world.data, azimuth=118, distance=2.6,
                                 lookat=(0.15, -0.35, 0.75), elevation=-14),
                [(label, colour),
                 (f"t={t:4.1f}s   {gap:.2f} m from the goal"
                  + ("" if state["fell_at"] is None
                     else f"   FELL at {state['fell_at']:.0f}s"),
                  (235, 235, 235))]))
            state["next"] += 1.0 / RECORD_FPS

    rt.on_frame = on_frame
    rt.stance.hold(goal, rt.base_yaw)          # ONE command
    pel = world.model.body("pelvis").id
    fell_at = None
    while rt._t < MAX_SECONDS:
        rt.step(0.25)
        R = world.data.xmat[pel].reshape(3, 3)
        z = float(world.data.xpos[pel][2])
        # height AND orientation: a robot pitched onto its face keeps its
        # pelvis high, and height alone has misreported a fall here before
        if fell_at is None and (z < 0.55 or R[2, 2] < 0.7):
            fell_at = rt._t
            state["fell_at"] = fell_at
            rt.step(1.5)                    # a moment on the floor, then stop
            break
    rig.close()
    gap = float(np.linalg.norm(rt.base_xy - goal))
    print(f"{label:34} reached {gap:.2f} m from goal, "
          + (f"fell at {fell_at:.0f} s" if fell_at else "upright at 32 s"))
    return frames, gap, fell_at


def main():
    a, ga, fa = run("g1_stance.pt", "shipped stance: falls in ~1 s",
                    (255, 150, 130))
    b, gb, fb = run("g1_travel_v1.pt", "goal-commanded: reaches it, falls at ~24 s",
                    (255, 210, 130))
    # pad the shorter run with its last frame so both panels stay in step
    n = max(len(a), len(b))
    a += [a[-1]] * (n - len(a))
    b += [b[-1]] * (n - len(b))
    out = REPO / "demos" / "media" / "12_travel_policy.gif"
    imageio.mimsave(out, [np.hstack([a[i], b[i]]) for i in range(n)],
                    fps=PLAY_FPS, loop=0)
    print(f"\nwrote {out} ({out.stat().st_size / 1e6:.1f} MB, {n} frames)")


if __name__ == "__main__":
    main()
