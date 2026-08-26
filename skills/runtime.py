"""Shared execution context for skills: owns the world, per-arm differential
IK, the servo loop, weld-grasp toggling, and an optional frame-recording hook.
Skills are deterministic, blocking routines built on these primitives — the
mid-level API between the VLM brain and the physics.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

import mujoco

from bodies.body import EE_SITE
from controllers.diff_ik import DiffIK
from controllers.admittance import joint_limit_scale


class Runtime:
    def __init__(self, world):
        self.world = world
        self.model, self.data = world.model, world.data
        self.body = world.body
        self.dt = self.model.opt.timestep
        self.ik = {}
        self.arm = {}
        for side, joints in self.body.arm_joints.items():
            self.ik[side] = DiffIK(self.model, EE_SITE[side], joints)
            self.arm[side] = {
                "act": np.array([self.body.actuator_of[j] for j in joints]),
                "qpos": np.array([self.model.joint(j).qposadr[0] for j in joints]),
                "limits": np.array([self.body.joint_limits[j] for j in joints]),
                "home_ctrl": None,  # captured on first use
            }
        self.on_frame = None  # callable(t) for demos; None = headless
        self._t = 0.0
        self._held: dict[str, str | None] = {s: None for s in self.arm}

    # ---------------- core stepping ----------------

    def step(self, seconds: float) -> None:
        for _ in range(int(round(seconds / self.dt))):
            mujoco.mj_step(self.model, self.data)
            self._t += self.dt
            if self.on_frame is not None:
                self.on_frame(self._t)

    def ee_pos(self, side: str) -> np.ndarray:
        return self.data.site_xpos[self.ik[side].site_id].copy()

    def nearest_arm(self, pos) -> str:
        return "left" if pos[1] > 0 else "right"

    # ---------------- arm servoing ----------------

    def servo_ee(self, side: str, target, speed: float = 0.12,
                 tol: float = 0.02, timeout: float = 5.0) -> bool:
        """P-servo the arm's EE site to a world position via weighted DLS."""
        target = np.asarray(target, dtype=float)
        a = self.arm[side]
        if a["home_ctrl"] is None:
            a["home_ctrl"] = self.data.ctrl[a["act"]].copy()
        t_end = self._t + timeout
        while self._t < t_end:
            err = target - self.ee_pos(side)
            if np.linalg.norm(err) < tol:
                return True
            v = err * 4.0
            n = np.linalg.norm(v)
            if n > speed:
                v *= speed / n
            qdot = self.ik[side].solve(self.data, np.r_[v, np.zeros(3)])
            qdot *= joint_limit_scale(self.data.qpos[a["qpos"]], a["limits"])
            q = self.data.qpos[a["qpos"]]
            # anti-windup: the position target may never drift far from the
            # actual joints, or a blocked arm winds up and shoves through
            new_ctrl = self.data.ctrl[a["act"]] + qdot * self.dt
            new_ctrl = np.clip(new_ctrl, q - 0.15, q + 0.15)
            self.data.ctrl[a["act"]] = np.clip(
                new_ctrl, a["limits"][:, 0], a["limits"][:, 1])
            self.step(self.dt)
        return np.linalg.norm(target - self.ee_pos(side)) < tol * 2

    def _interp_ctrl(self, side: str, target, duration: float) -> None:
        a = self.arm[side]
        start = self.data.ctrl[a["act"]].copy()
        n = max(1, int(duration / self.dt))
        for i in range(n):
            s = (i + 1) / n
            s = s * s * (3 - 2 * s)  # smoothstep profile
            self.data.ctrl[a["act"]] = (1 - s) * start + s * np.asarray(target)
            self.step(self.dt)

    def move_arm_home(self, side: str, duration: float = 1.5) -> None:
        a = self.arm[side]
        if a["home_ctrl"] is not None:
            self._interp_ctrl(side, a["home_ctrl"], duration)

    def move_arm_ready(self, side: str, duration: float = 1.2) -> None:
        """Joint-space move to the config's elbow-up 'ready' pose. The home
        pose (arm hanging) is near-singular; Cartesian servoing starts here.
        An optional 'ready_via' waypoint routes the swing over the table edge
        instead of through it."""
        a = self.arm[side]
        if a["home_ctrl"] is None:
            a["home_ctrl"] = self.data.ctrl[a["act"]].copy()
        cfg = self.body.cfg["arms"][side]
        via, ready = cfg.get("ready_via"), cfg.get("ready")
        if via is not None:
            self._interp_ctrl(side, via, duration * 0.6)
        if ready is not None:
            self._interp_ctrl(side, ready, duration * 0.6)

    def square_up(self, duration: float = 0.6) -> None:
        """Return the gaze/waist joint to neutral: manipulation poses are
        tuned for an unrotated torso."""
        gaze = self.body.cfg.get("gaze_joint")
        if gaze is None:
            return
        act = self.body.actuator_of[gaze]
        start = float(self.data.ctrl[act])
        if abs(start) < 0.02:
            return
        n = max(1, int(duration / self.dt))
        for i in range(n):
            self.data.ctrl[act] = start * (1 - (i + 1) / n)
            self.step(self.dt)

    def recover(self, side: str) -> None:
        """After a failed reach: lift straight up out of clutter, then
        re-stage at the ready pose so the next attempt starts clean."""
        ee = self.ee_pos(side)
        self.servo_ee(side, [ee[0], ee[1], 0.95], speed=0.15, tol=0.04,
                      timeout=2.0)
        self.move_arm_ready(side)

    # ---------------- weld grasping ----------------

    def _weld_id(self, side: str, obj_body: str) -> int:
        return self.model.equality(f"grasp_{side}_{obj_body}").id

    def attach(self, side: str, obj_body: str) -> None:
        """Freeze the object's current pose relative to the hand (weld)."""
        m, d = self.model, self.data
        b1 = m.body(self.body.cfg["arms"][side]["ee_parent"]).id
        b2 = m.body(obj_body).id
        R1 = d.xmat[b1].reshape(3, 3)
        p_rel = R1.T @ (d.xpos[b2] - d.xpos[b1])
        q1 = Rotation.from_matrix(R1)
        q2 = Rotation.from_matrix(d.xmat[b2].reshape(3, 3))
        q_rel = np.roll((q1.inv() * q2).as_quat(), 1)  # wxyz
        eq = self._weld_id(side, obj_body)
        m.eq_data[eq, :3] = 0.0
        m.eq_data[eq, 3:6] = p_rel
        m.eq_data[eq, 6:10] = q_rel
        m.eq_data[eq, 10] = 1.0
        d.eq_active[eq] = 1
        self._held[side] = obj_body

    def release(self, side: str) -> None:
        if self._held[side] is not None:
            self.data.eq_active[self._weld_id(side, self._held[side])] = 0
            self._held[side] = None

    def held(self, side: str) -> str | None:
        return self._held[side]
