"""Milestone 6: physics-based walking, tuned to look human.

The G1 walks on its own two legs under a pretrained RL policy (MJX/PPO, from
MuJoCo Playground) closing the loop at 50 Hz on real IMU and joint feedback.
Nothing about the base is scripted — no glide, no animation, no weld.

This demo runs the same walk twice to show what "human-like" cost:

  before : the policy's default 1.5 Hz gait with the arms held still
  after  : 1.3 Hz gait, arms left under the policy, faster commanded walk

and prints the gait measurements next to human references. Cadence is
compared after Froude scaling — a 0.65 m leg does not walk at a human's
1.9 steps/s, it walks at about 2.2-2.5.
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
from controllers.locomotion import LocomotionPolicy  # noqa: E402
from sim.cameras import CameraRig  # noqa: E402

RECORD_FPS = 12
PLAY_FPS = 24
WALK_SECONDS = 7.0


def build_world(body):
    w = body.spec.worldbody
    w.add_light(pos=[2, -2, 3], dir=[-0.4, 0.4, -1], diffuse=[0.7] * 3,
                ambient=[0.32] * 3)
    for i in range(-2, 14):   # ground markings, so the stride is visible
        w.add_geom(name=f"stripe{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[0.012, 3.0, 0.001], pos=[i * 0.5, 0, 0.001],
                   rgba=[0.44, 0.48, 0.54, 1])
    w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
               size=[12, 12, 0.1], rgba=[0.32, 0.36, 0.42, 1])
    model = body.spec.compile()
    data = mujoco.MjData(model)
    body.bind(model)
    body.reset(data)
    mujoco.mj_forward(model, data)
    return model, data


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


def run(label, gait_freq, speed, freeze_arms, color):
    body = Body(REPO / "bodies" / "g1_walk" / "config.yaml")
    model, data = build_world(body)
    cfg = body.cfg["locomotion"]
    pol = LocomotionPolicy(model, REPO / cfg["policy"], cfg["stand_angles"],
                           cfg["ctrl_dt"], cfg["action_scale"],
                           gait_freq=gait_freq)
    pol.controlled = "leg_waist" if freeze_arms else None
    rig = CameraRig(model, width=560, height=420)

    pelvis = model.body("pelvis").id
    lfoot = model.body("left_ankle_roll_link").id
    shoulder = model.joint("left_shoulder_pitch_joint").qposadr[0]

    pol.set_command(0, 0, 0)
    for _ in range(int(1.5 / model.opt.timestep)):
        pol.step(data)
        mujoco.mj_step(model, data)

    frames, lz, pz, sh = [], [], [], []
    start = data.xpos[pelvis][:2].copy()
    pol.set_command(speed, 0, 0)
    next_frame = 0.0
    for i in range(int(WALK_SECONDS / model.opt.timestep)):
        pol.step(data)
        mujoco.mj_step(model, data)
        lz.append(data.xpos[lfoot][2])
        pz.append(data.xpos[pelvis][2])
        sh.append(data.qpos[shoulder])
        t = i * model.opt.timestep
        if t >= next_frame:
            x, y = data.xpos[pelvis][:2]
            frames.append(rig.third_person(data, azimuth=90, distance=2.5,
                                           lookat=(x, y, 0.70), elevation=-4))
            next_frame += 1.0 / RECORD_FPS

    lz = np.array(lz)
    dist = float(np.linalg.norm(data.xpos[pelvis][:2] - start))
    lifted = lz > lz.min() + 0.03
    steps = max(1, int(np.sum(np.diff(lifted.astype(int)) == 1)) * 2)
    m = {"speed": dist / WALK_SECONDS, "cadence": steps / WALK_SECONDS,
         "stride": dist / steps, "clearance": float(lz.max() - lz.min()),
         "bob": float(np.ptp(pz)), "arm_swing": float(np.ptp(sh)),
         "upright": bool(data.xpos[pelvis][2] > 0.6)}
    deg = chr(176)
    captioned = [caption(f, [
        (label, color),
        (f"{m['speed']:.2f} m/s   {m['cadence']:.1f} steps/s   "
         f"stride {m['stride']:.2f} m   arm swing "
         f"{np.degrees(m['arm_swing']):.0f}{deg}", (235, 235, 235)),
    ]) for f in frames]
    rig.close()
    return captioned, m


def main() -> None:
    before, mb = run("BEFORE: default 1.5 Hz gait, arms held still",
                     1.5, 0.5, True, (255, 170, 120))
    after, ma = run("AFTER: 1.3 Hz gait, arms free, longer stride",
                    1.3, 1.0, False, (120, 230, 160))

    n = min(len(before), len(after))
    combined = before[:n] + after[:n]
    out = REPO / "demos" / "media" / "07_physics_walk.gif"
    imageio.mimsave(out, combined, fps=PLAY_FPS, loop=0)

    print(f"{'metric':17} {'before':>8} {'after':>8}   human reference")
    rows = [("speed m/s", "speed", "1.2-1.4 (0.9-1.2 scaled)"),
            ("cadence steps/s", "cadence", "1.8-2.0 (2.2-2.5 scaled)"),
            ("stride m", "stride", "~0.4 x leg length = 0.26"),
            ("foot clearance m", "clearance", "0.02-0.05"),
            ("pelvis bob m", "bob", "0.04-0.05"),
            ("arm swing rad", "arm_swing", "0.4-0.6")]
    for name, key, ref in rows:
        print(f"{name:17} {mb[key]:8.2f} {ma[key]:8.2f}   {ref}")
    print(f"{'upright':17} {str(mb['upright']):>8} {str(ma['upright']):>8}")
    print(f"\nwrote {out} ({out.stat().st_size / 1e6:.1f} MB, "
          f"{len(combined)} frames)")


if __name__ == "__main__":
    main()
