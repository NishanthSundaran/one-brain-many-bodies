"""Two ways to put a hand on a target: solved, and learned.

Left  — deterministic: measured IK, solved offline, then moved in joint
        space. Only the arm moves.
Right — learned: a policy trained here (training/train_reach.py) that drives
        the arm, waist and legs together (22 joints), so it can lean and
        shift its stance to get the hand there.

Same targets, same robot, same physics. The interesting part is where they
differ: the learned policy is roughly twice as accurate where it works and
handles a low target the IK path topples on, while the IK path degrades
gracefully at distance — it misses, but it stays on its feet.
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
from controllers.reach_policy import ReachPolicy  # noqa: E402
from sim.cameras import CameraRig  # noqa: E402
from skills.runtime import Runtime  # noqa: E402

RECORD_FPS = 10
PLAY_FPS = 20
TARGETS = [(0.25, -0.18, 0.12), (0.32, -0.20, 0.16), (0.30, -0.30, 0.10)]


def build():
    body = Body(REPO / "bodies" / "g1_walk" / "config.yaml")
    w = body.spec.worldbody
    w.add_light(pos=[2, -2, 3], dir=[-0.4, 0.4, -1], diffuse=[0.7] * 3,
                ambient=[0.32] * 3)
    w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
               size=[8, 8, 0.1], rgba=[0.32, 0.36, 0.42, 1])
    marker = w.add_body(name="target", pos=[0, 0, 0.5])
    marker.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.028],
                    rgba=[1.0, 0.55, 0.1, 0.85], contype=0, conaffinity=0)

    class World:
        pass
    world = World()
    world.body = body
    world.model = body.spec.compile()
    world.data = mujoco.MjData(world.model)
    body.bind(world.model)
    body.reset(world.data)
    mujoco.mj_forward(world.model, world.data)
    return world


def caption(frame, lines):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=14)
    except TypeError:
        font = ImageFont.load_default()
    h = 8 + 18 * len(lines)
    draw.rectangle([0, frame.shape[0] - h, frame.shape[1], frame.shape[0]],
                   fill=(0, 0, 0))
    for i, (text, color) in enumerate(lines):
        draw.text((8, frame.shape[0] - h + 3 + 18 * i), text, fill=color,
                  font=font)
    return np.asarray(img)


def run(method, label, colour, targets):
    world = build()
    rt = Runtime(world)
    rig = CameraRig(world.model, width=520, height=420)
    marker = world.model.body("target").id
    hand_site = world.model.site("right_ee").id
    frames, results = [], []

    pol = None
    if method == "rl":
        # the policy was trained on the nominal arm gains, so deploy it there
        rt._set_arm_gain(1.0)
        rt._standing = False
        rt.loco = None
        rt.stance = None
        pol = ReachPolicy(world.model, REPO / "third_party/policies/g1_reach.pt",
                          rt.body.cfg["locomotion"]["stand_angles"])

    state = {"next": 0.0, "target": None}

    def snap(t):
        if t >= state["next"]:
            # measure live, so the number on screen is the distance right now
            err = (float(np.linalg.norm(state["target"]
                                        - world.data.site_xpos[hand_site]))
                   if state["target"] is not None else None)
            frames.append(caption(
                rig.third_person(world.data, azimuth=118, distance=1.9,
                                 lookat=(0.22, -0.18, 0.85), elevation=-10),
                [(label, colour),
                 (f"hand-to-target {err:.3f} m" if err is not None
                  else "reaching...", (230, 230, 230))]))
            state["next"] += 1.0 / RECORD_FPS

    rt.on_frame = snap

    for L in targets:
        tgt = rt.to_world(np.array(L[:2]))
        tgt[2] = world.data.xpos[rt.root_id][2] + L[2]
        world.model.body_pos[marker] = tgt
        mujoco.mj_forward(world.model, world.data)
        state["target"] = tgt

        if method == "ik":
            rt.move_arm_ready("right")
            rt.reach_joint_space("right", tgt)
            rt.step(0.6)
        else:
            pol.aim(tgt)
            for i in range(int(4.0 / rt.dt)):
                pol.step(world.data)
                mujoco.mj_step(world.model, world.data)
                rt._t += rt.dt
                snap(rt._t)

        err = float(np.linalg.norm(tgt - world.data.site_xpos[hand_site]))
        upright = bool(world.data.xpos[rt.root_id][2] > 0.55)
        if method == "ik":
            rt.step(0.4)
        else:
            for _ in range(int(0.4 / rt.dt)):
                pol.step(world.data)
                mujoco.mj_step(world.model, world.data)
                rt._t += rt.dt
                snap(rt._t)
        results.append((L, err, upright))
        print(f"  {label:26s} {str(L):22} err {err:.3f} m "
              f"{'upright' if upright else 'FELL'}")
    rig.close()
    return frames, results


def main() -> None:
    print("reaching for the same three targets, two ways:")
    # A fresh robot for every target: run them back to back and whichever
    # method falls first is penalised on every target after it, which
    # measures the order of the list rather than the method.
    combined, ik_res, rl_res = [], [], []
    for L in TARGETS:
        ikf, ikr = run("ik", "DETERMINISTIC: measured IK, arm only",
                       (255, 190, 120), [L])
        rlf, rlr = run("rl", "LEARNED: whole body, 22 joints",
                       (130, 235, 165), [L])
        ik_res += ikr
        rl_res += rlr
        n = max(len(ikf), len(rlf))
        ikf += [ikf[-1]] * (n - len(ikf))
        rlf += [rlf[-1]] * (n - len(rlf))
        combined += [np.hstack([a, b]) for a, b in zip(ikf, rlf)]
    out = REPO / "demos" / "media" / "09_reach_ik_vs_rl.gif"
    imageio.mimsave(out, combined, fps=PLAY_FPS, loop=0)

    def summary(res):
        up = [e for _, e, u in res if u]
        return (sum(1 for _, e, u in res if e < 0.06 and u), len(res),
                float(np.mean(up)) if up else float("nan"))
    i_ok, i_n, i_mean = summary(ik_res)
    r_ok, r_n, r_mean = summary(rl_res)
    print(f"\n{'':28} {'within 6 cm':>12} {'mean error':>12}")
    print(f"{'deterministic IK':28} {i_ok}/{i_n:<11} {i_mean:11.3f} m")
    print(f"{'learned whole-body reach':28} {r_ok}/{r_n:<11} {r_mean:11.3f} m")
    print(f"\nwrote {out} ({out.stat().st_size / 1e6:.1f} MB, {n} frames)")


if __name__ == "__main__":
    main()
