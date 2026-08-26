"""Physics-based locomotion: a pretrained RL policy closes the loop on the
floating base. No kinematic cheating — the robot balances and steps because
the policy reacts to real IMU and joint feedback, and it falls over if the
controller is wrong.

The policy (MuJoCo Playground, MJX/PPO, Apache-2.0) emits 29 joint targets at
50 Hz. This wrapper indexes joints *by name*, so it works on a model whose
joint order differs from the training model — e.g. this project's G1 with
dexterous hands attached, where finger joints interleave with the arms.

`controlled` narrows what the policy writes: during manipulation the legs and
waist stay under the policy (it keeps balancing) while the arm controller owns
the arms.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort

LEGS = [f"{s}_{j}_joint" for s in ("left", "right")
        for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee",
                  "ankle_pitch", "ankle_roll")]
WAIST = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
ARMS = [f"{s}_{j}_joint" for s in ("left", "right")
        for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                  "wrist_roll", "wrist_pitch", "wrist_yaw")]
POLICY_JOINTS = LEGS + WAIST + ARMS          # canonical training order


class LocomotionPolicy:
    def __init__(self, model, policy_path: str, default_angles,
                 ctrl_dt: float = 0.02, action_scale: float = 0.5,
                 gait_freq: float = 1.5):
        self.model = model
        self.session = ort.InferenceSession(
            str(policy_path), providers=["CPUExecutionProvider"])
        self.default = np.asarray(default_angles, dtype=np.float32)
        self.action_scale = action_scale
        self.ctrl_dt = ctrl_dt
        self.n_substeps = max(1, int(round(ctrl_dt / model.opt.timestep)))

        self.qpos_adr = np.array([model.joint(n).qposadr[0] for n in POLICY_JOINTS])
        self.qvel_adr = np.array([model.joint(n).dofadr[0] for n in POLICY_JOINTS])
        self.act_ids = np.array([model.actuator(n).id for n in POLICY_JOINTS])
        self.leg_waist = np.array(
            [i for i, n in enumerate(POLICY_JOINTS) if n in set(LEGS + WAIST)])

        self.imu_site = model.site("imu_in_pelvis").id
        self.linvel_sensor = model.sensor("local_linvel_pelvis").id
        self.gyro_sensor = model.sensor("gyro_pelvis").id
        self._sensor_adr = {
            s: (model.sensor_adr[s], model.sensor_dim[s])
            for s in (self.linvel_sensor, self.gyro_sensor)}

        self.command = np.zeros(3, dtype=np.float32)   # vx, vy, yaw rate
        self.last_action = np.zeros(29, dtype=np.float32)
        self.phase = np.array([0.0, np.pi])
        self.phase_dt = 2 * np.pi * gait_freq * ctrl_dt
        self.controlled = None      # None = all 29 joints
        self._counter = 0

    def _sensor(self, data, sid):
        adr, dim = self._sensor_adr[sid]
        return data.sensordata[adr:adr + dim]

    def observe(self, data) -> np.ndarray:
        gravity = data.site_xmat[self.imu_site].reshape(3, 3).T @ np.array(
            [0.0, 0.0, -1.0])
        return np.hstack([
            self._sensor(data, self.linvel_sensor),
            self._sensor(data, self.gyro_sensor),
            gravity,
            self.command,
            data.qpos[self.qpos_adr] - self.default,
            data.qvel[self.qvel_adr],
            self.last_action,
            np.concatenate([np.cos(self.phase), np.sin(self.phase)]),
        ]).astype(np.float32)

    def step(self, data) -> None:
        """Call once per physics step; runs inference every `ctrl_dt`."""
        self._counter += 1
        if self._counter % self.n_substeps:
            return
        action = self.session.run(
            ["continuous_actions"],
            {"obs": self.observe(data).reshape(1, -1)})[0][0]
        self.last_action = action.copy()
        targets = action * self.action_scale + self.default
        idx = self.leg_waist if self.controlled == "leg_waist" else slice(None)
        data.ctrl[self.act_ids[idx]] = targets[idx]
        self.phase = np.fmod(self.phase + self.phase_dt + np.pi,
                             2 * np.pi) - np.pi

    def policy_joint_names(self) -> list:
        """Joints this policy owns (legs and waist when narrowed)."""
        if self.controlled == "leg_waist":
            return LEGS + WAIST
        return list(POLICY_JOINTS)

    def set_command(self, vx: float = 0.0, vy: float = 0.0,
                    wz: float = 0.0) -> None:
        self.command[:] = (vx, vy, wz)
