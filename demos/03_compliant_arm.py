"""Milestone 2: compliant arm. A scripted external force drags the G1's right
hand; the ported thesis admittance controller turns the wrench into a
compliant EE twist, differential IK maps it to arm joint velocities, and the
stiffness term pulls the hand back to its reference on release.
"""

import pathlib
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from bodies.body import Body
from controllers.admittance import AdmittanceController, joint_limit_scale
from controllers.diff_ik import DiffIK
from sim.cameras import CameraRig
from sim.world import World

FPS = 25

# (t_start, t_end, force_world, caption)
PHASES = [
    (0.0, 0.8, np.zeros(3), "rest"),
    (0.8, 2.2, np.array([0.0, -18.0, 6.0]), "human pulls hand sideways"),
    (2.2, 3.8, np.zeros(3), "released: returns to reference"),
    (3.8, 5.2, np.array([12.0, 0.0, -16.0]), "human pushes hand down"),
    (5.2, 7.0, np.zeros(3), "released: returns to reference"),
]


def site_pose(model, data, site_id):
    pos = data.site_xpos[site_id].copy()
    quat = np.roll(Rotation.from_matrix(
        data.site_xmat[site_id].reshape(3, 3)).as_quat(), 1)
    return pos, quat


def caption(frame: np.ndarray, text: str) -> np.ndarray:
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
    world = World(body, objects=[])
    model, data = world.model, world.data
    dt = model.opt.timestep

    site_id = model.site("right_ee").id
    wrist_bid = model.body(body.cfg["arms"]["right"]["ee_parent"]).id
    joints = body.arm_joints["right"]
    act_ids = np.array([body.actuator_of[j] for j in joints])
    limits = np.array([body.joint_limits[j] for j in joints])
    qpos_ids = np.array([model.joint(j).qposadr[0] for j in joints])

    ctrl = AdmittanceController(dt=dt)
    ik = DiffIK(model, "right_ee", joints)

    pos0, quat0 = site_pose(model, data, site_id)
    ctrl.set_reference(pos0, quat0)

    rig = CameraRig(model, width=560, height=400)
    frames, next_frame_t, t = [], 0.0, 0.0
    max_disp = 0.0
    while t < PHASES[-1][1]:
        force = np.zeros(3)
        label = ""
        for t0, t1, f, cap in PHASES:
            if t0 <= t < t1:
                force, label = f, cap
        data.xfrc_applied[wrist_bid, :3] = force

        pos, quat = site_pose(model, data, site_id)
        max_disp = max(max_disp, np.linalg.norm(pos - pos0))
        twist = ctrl.update(np.r_[force, np.zeros(3)], pos, quat)
        qdot = ik.solve(data, twist)
        qdot *= joint_limit_scale(data.qpos[qpos_ids], limits)
        data.ctrl[act_ids] = np.clip(data.ctrl[act_ids] + qdot * dt,
                                     limits[:, 0], limits[:, 1])

        mujoco.mj_step(model, data)
        t += dt
        if t >= next_frame_t:
            frame = rig.third_person(data, azimuth=-140, distance=1.4,
                                     lookat=(0.2, -0.05, 0.85), elevation=-10)
            frames.append(caption(frame, label))
            next_frame_t += 1.0 / FPS

    pos_end, _ = site_pose(model, data, site_id)
    out = REPO / "demos" / "media" / "03_compliant_arm.gif"
    imageio.mimsave(out, frames, fps=FPS, loop=0)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"peak EE displacement {max_disp:.3f} m, "
          f"final error vs reference {np.linalg.norm(pos_end - pos0):.4f} m")


if __name__ == "__main__":
    main()
