"""Embodiment-agnostic body layer.

A Body loads a per-robot YAML config plus its menagerie MJCF, then edits the
spec so every humanoid presents the same interface to the layers above:
a fixed base (v1), a `head_cam` camera, `left_ee`/`right_ee` sites, named arm
joint groups, and a home pose. Joint limits and actuator maps are derived
from the compiled model, never hardcoded.
"""

from __future__ import annotations

import pathlib

import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

REPO = pathlib.Path(__file__).resolve().parents[1]

EE_SITE = {"left": "left_ee", "right": "right_ee"}
HEAD_CAM = "head_cam"


def _camera_quat(pitch_down_deg: float) -> np.ndarray:
    """wxyz quat for a camera on an upright, +X-facing body: looks along +X,
    image up = world +Z, pitched down by the given angle."""
    p = np.deg2rad(pitch_down_deg)
    forward = np.array([np.cos(p), 0.0, -np.sin(p)])
    z_cam = -forward                      # mujoco cameras look along -Z
    x_cam = np.array([0.0, -1.0, 0.0])    # image right
    y_cam = np.cross(z_cam, x_cam)
    q = Rotation.from_matrix(np.column_stack([x_cam, y_cam, z_cam])).as_quat()
    return np.roll(q, 1)  # xyzw -> wxyz


class Body:
    def __init__(self, config_path: str | pathlib.Path):
        config_path = pathlib.Path(config_path)
        cfg = yaml.safe_load(config_path.read_text())
        self.name: str = cfg["name"]
        self.cfg = cfg

        self.spec = mujoco.MjSpec.from_file(str(REPO / cfg["mjcf"]))

        # Home pose: from a model keyframe, or spelled out in the config for
        # models that keep their keyframes elsewhere (e.g. H1's scene.xml).
        if "keyframe" in cfg:
            key = next(k for k in self.spec.keys if k.name == cfg["keyframe"])
            key_qpos = np.array(key.qpos)
            key_ctrl = np.array(key.ctrl)
        else:
            home = cfg["home"]
            key_qpos = np.concatenate([home["root"], home["joints"]])
            key_ctrl = np.array(home["joints"], dtype=float)
        for k in list(self.spec.keys):
            self.spec.delete(k)

        # Actuator normalization: every body presents position actuators.
        # Torque-motor models (H1) get their motors replaced by position
        # servos so the layers above never see the difference.
        act_cfg = cfg.get("actuation")
        if act_cfg is not None:
            for a in list(self.spec.actuators):
                self.spec.delete(a)
            kp, kv = float(act_cfg["kp"]), float(act_cfg["kv"])
            for j in self.spec.joints:
                if j.type != mujoco.mjtJoint.mjJNT_HINGE:
                    continue
                act = self.spec.add_actuator(
                    name=j.name, target=j.name,
                    trntype=mujoco.mjtTrn.mjTRN_JOINT)
                act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
                act.gainprm[0] = kp
                act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
                act.biasprm[:3] = [0.0, -kp, -kv]
                act.ctrlrange = j.range
                act.ctrllimited = True

        # v1: fixed base. Drop the free joint and plant the root at its
        # keyframe pose; the remaining qpos layout is the joints only.
        root = self.spec.body(cfg["root_body"])
        free = [j for j in root.joints if j.type == mujoco.mjtJoint.mjJNT_FREE]
        for j in free:
            self.spec.delete(j)
        root.pos = key_qpos[:3]
        root.quat = key_qpos[3:7]
        self.home_qpos = key_qpos[7:]
        self.home_ctrl = key_ctrl

        # Body composition: bolt a dexterous hand from another menagerie
        # model onto the wrist (e.g. Sharpa Wave five-finger hands on the
        # H1, which ships with bare ball wrists).
        for side, arm in cfg["arms"].items():
            att = arm.get("hand_attach")
            if att is None:
                continue
            child = mujoco.MjSpec.from_file(str(REPO / att["mjcf"]))
            for k in list(child.keys):
                child.delete(k)
            mount = self.spec.body(arm["ee_parent"]).add_site(
                name=f"hand_mount_{side}", pos=att["pos"], quat=att["quat"])
            self.spec.attach(child, prefix=att["prefix"], site=mount)

        # Same interface on every humanoid: EE sites + head camera.
        for side, arm in cfg["arms"].items():
            self.spec.body(arm["ee_parent"]).add_site(
                name=EE_SITE[side], pos=arm["ee_offset"], size=[0.012, 0.012, 0.012]
            )
            # Finger closure is cosmetic (the hold is a weld), so curled
            # fingers get no collisions and never fight the held object.
            # The palm keeps its collisions: without them the hand sinks
            # through bin walls and tabletops like a ghost.
            hand = arm.get("hand")
            if hand and hand.get("link_prefix"):
                palm = hand.get("palm_link")
                for b in self.spec.bodies:
                    if b.name.startswith(hand["link_prefix"]) and b.name != palm:
                        for g in b.geoms:
                            g.contype = 0
                            g.conaffinity = 0
        hc = cfg["head_camera"]
        self.spec.body(hc["parent"]).add_camera(
            name=HEAD_CAM,
            pos=hc["pos"],
            quat=_camera_quat(hc["pitch_down_deg"]),
            fovy=hc["fovy"],
        )

    # -- post-compile introspection (call after the world compiles the spec) --

    def bind(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.joint_names = [model.joint(i).name for i in range(model.njnt)
                            if model.joint(i).type[0] != mujoco.mjtJoint.mjJNT_FREE]
        self.joint_limits = {n: tuple(model.joint(n).range) for n in self.joint_names}
        # joint name -> actuator id, via the actuator transmission target
        # (joint transmissions only; tendon actuators have no joint)
        self.actuator_of = {}
        for a in range(model.nu):
            if model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
                continue
            jid = model.actuator_trnid[a, 0]
            self.actuator_of[model.joint(jid).name] = a
        self.arm_joints = {s: list(cfg["joints"]) for s, cfg in self.cfg["arms"].items()}

    def reset(self, data: mujoco.MjData) -> None:
        """Home pose for the robot's own dofs; leaves any scene objects alone."""
        for i, n in enumerate(self.joint_names[: len(self.home_qpos)]):
            data.qpos[self.model.joint(n).qposadr[0]] = self.home_qpos[i]
        data.ctrl[: len(self.home_ctrl)] = self.home_ctrl
