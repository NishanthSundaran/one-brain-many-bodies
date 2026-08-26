"""Shared execution context for skills: owns the world, per-arm differential
IK, the servo loop, weld-grasp toggling, and an optional frame-recording hook.
Skills are deterministic, blocking routines built on these primitives — the
mid-level API between the VLM brain and the physics.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

import mujoco

ACCEL_MAX = 0.8      # m/s^2 command-velocity ramp: no robotic jump-starts
ROT_GAIN = 2.5       # 1/s toward the canonical palm orientation
ROT_MAX = 1.2        # rad/s

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
            hand = self.body.cfg["arms"][side].get("hand")
            if hand is not None:
                self.arm[side]["hand_act"] = np.array(
                    [self.body.actuator_of[j] for j in hand["joints"]])
                self.arm[side]["hand_open"] = np.asarray(hand["open"], dtype=float)
                self.arm[side]["hand_closed"] = np.asarray(hand["closed"], dtype=float)
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

    def arm_can_reach_y(self, side: str, y: float) -> bool:
        """Cheap workspace gate from config: refuse cross-body targets
        before moving instead of flailing through the scene."""
        cfg = self.body.cfg["arms"][side]
        if side == "left" and "reach_min_y" in cfg:
            return y >= cfg["reach_min_y"]
        if side == "right" and "reach_max_y" in cfg:
            return y <= cfg["reach_max_y"]
        return True

    # ---------------- arm servoing ----------------

    def _orientation_twist(self, side: str) -> np.ndarray:
        """Angular velocity steering the palm back to the arm's canonical
        manipulation orientation (config `ee_quat`, world frame). Bodies
        without the config key (e.g. 4-DoF H1 arms) get no orientation task."""
        quat = self.body.cfg["arms"][side].get("ee_quat")
        if quat is None:
            return np.zeros(3)
        R_cur = self.data.site_xmat[self.ik[side].site_id].reshape(3, 3)
        R_des = Rotation.from_quat(np.roll(quat, -1)).as_matrix()
        w = ROT_GAIN * Rotation.from_matrix(R_des @ R_cur.T).as_rotvec()
        n = np.linalg.norm(w)
        return w * (ROT_MAX / n) if n > ROT_MAX else w

    def servo_ee(self, side: str, target, speed: float = 0.12,
                 tol: float = 0.02, timeout: float = 5.0) -> bool:
        """6D servo: P-control to a world position while holding the palm's
        canonical orientation, with an accel-limited velocity ramp."""
        target = np.asarray(target, dtype=float)
        a = self.arm[side]
        if a["home_ctrl"] is None:
            a["home_ctrl"] = self.data.ctrl[a["act"]].copy()
        v_cmd = np.zeros(3)
        t_end = self._t + timeout
        while self._t < t_end:
            err = target - self.ee_pos(side)
            if np.linalg.norm(err) < tol:
                return True
            v = err * 4.0
            n = np.linalg.norm(v)
            if n > speed:
                v *= speed / n
            dv = v - v_cmd
            dn = np.linalg.norm(dv)
            step_max = ACCEL_MAX * self.dt
            v_cmd = v_cmd + (dv if dn <= step_max else dv * step_max / dn)
            qdot = self.ik[side].solve(
                self.data, np.r_[v_cmd, self._orientation_twist(side)])
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
        """Reverse of the ready swing: retract over the table, out to the
        side, then down — a straight joint blend to hanging would drag the
        fist through whatever sits near the table edge."""
        a = self.arm[side]
        if a["home_ctrl"] is None:
            return
        cfg = self.body.cfg["arms"][side]
        via = cfg.get("ready_via")
        if via is not None:
            if not isinstance(via[0], (list, tuple)):
                via = [via]
            for pose in reversed(via):
                self._interp_ctrl(side, pose, duration * 0.5)
        self._interp_ctrl(side, a["home_ctrl"], duration * 0.5)

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
            if not isinstance(via[0], (list, tuple)):
                via = [via]
            for pose in via:  # e.g. swing back, lift high, come in over the table
                self._interp_ctrl(side, pose, duration * 0.6)
        if ready is not None:
            self._interp_ctrl(side, ready, duration * 0.6)

    def set_hand(self, side: str, pose: str, duration: float = 0.4) -> None:
        """Curl or open the fingers ('closed'/'open'). Cosmetic on bodies
        with hands (the weld provides the hold); no-op on bodies without."""
        a = self.arm[side]
        if "hand_act" not in a:
            return
        target = a["hand_closed"] if pose == "closed" else a["hand_open"]
        start = self.data.ctrl[a["hand_act"]].copy()
        n = max(1, int(duration / self.dt))
        for i in range(n):
            s = (i + 1) / n
            self.data.ctrl[a["hand_act"]] = (1 - s) * start + s * target
            self.step(self.dt)

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
