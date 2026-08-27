"""Stance controller: the policy trained by `training/train_stance.py`.

The borrowed walking policy is a travelling policy — it creeps, holds a spot
only to ±0.2 m, and treats a sustained arm reach as something to step away
from. This one is trained for the opposite job: keep the pelvis on its mark
while the arms swing through a full reach, which is what manipulation needs.

Same shape as `LocomotionPolicy` (`step(data)` each physics tick, owns legs
and waist) so the runtime can switch between walking and standing without
anything above the controller layer noticing.
"""

from __future__ import annotations

import numpy as np
import torch

from controllers.locomotion import LEGS, WAIST, ARMS

CONTROLLED = LEGS + WAIST
CTRL_DT = 0.02
ACTION_SCALE = 0.4


class StancePolicy:
    def __init__(self, model, checkpoint, stand_angles):
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        from training.ppo import ActorCritic

        self.net = ActorCritic(ck["obs_dim"], ck["act_dim"])
        self.net.load_state_dict(ck["state"])
        self.net.eval()
        self.mean = np.asarray(ck["mean"])
        self.std = np.sqrt(np.asarray(ck["var"]) + 1e-8)

        self.model = model
        stance = dict(zip(LEGS + WAIST + ARMS, stand_angles))
        self.default = np.array([stance[n] for n in CONTROLLED])
        self.arm_default = np.array([stance[n] for n in ARMS])

        self.act = np.array([model.actuator(n).id for n in CONTROLLED])
        self.qpos = np.array([model.joint(n).qposadr[0] for n in CONTROLLED])
        self.qvel = np.array([model.joint(n).dofadr[0] for n in CONTROLLED])
        self.arm_qpos = np.array([model.joint(n).qposadr[0] for n in ARMS])
        self.arm_qvel = np.array([model.joint(n).dofadr[0] for n in ARMS])
        self.pelvis = model.body("pelvis").id
        self.imu = model.site("imu_in_pelvis").id
        self.gyro = model.sensor("gyro_pelvis").id
        self.linvel = model.sensor("local_linvel_pelvis").id

        self.n_substeps = max(1, int(round(CTRL_DT / model.opt.timestep)))
        self.last_action = np.zeros(len(CONTROLLED), dtype=np.float32)
        self.hold_xy = None      # the mark it stands on
        self.hold_yaw = 0.0      # and the direction it faces there
        self._counter = 0

    def _sensor(self, data, sid):
        adr = self.model.sensor_adr[sid]
        return data.sensordata[adr:adr + self.model.sensor_dim[sid]]

    def observe(self, data) -> np.ndarray:
        gravity = data.site_xmat[self.imu].reshape(3, 3).T @ np.array(
            [0.0, 0.0, -1.0])
        err = self.hold_xy - data.xpos[self.pelvis][:2]
        R = data.xmat[self.pelvis].reshape(3, 3)
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        yaw_err = (self.hold_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
        return np.concatenate([
            gravity,
            self._sensor(data, self.gyro),
            self._sensor(data, self.linvel),
            data.qpos[self.qpos] - self.default,
            data.qvel[self.qvel] * 0.1,
            data.qpos[self.arm_qpos] - self.arm_default,
            data.qvel[self.arm_qvel] * 0.1,
            self.last_action,
            err * 5.0,
            [np.cos(yaw_err), np.sin(yaw_err)],
        ]).astype(np.float32)

    def hold(self, xy, yaw: float = 0.0) -> None:
        """Stand here, facing this way. Position alone is not enough: a robot
        that turns on the spot puts the table where its arms are not."""
        self.hold_xy = np.asarray(xy, dtype=float)[:2].copy()
        self.hold_yaw = float(yaw)

    def step(self, data) -> None:
        if self.hold_xy is None:
            self.hold(data.xpos[self.pelvis][:2])
        self._counter += 1
        if self._counter % self.n_substeps:
            return
        obs = np.clip((self.observe(data) - self.mean) / self.std, -10, 10)
        with torch.no_grad():
            action = self.net.mu(self.net.pi(
                torch.as_tensor(obs, dtype=torch.float32))).numpy()
        action = np.clip(action, -1.0, 1.0)
        self.last_action = action.astype(np.float32)
        data.ctrl[self.act] = self.default + ACTION_SCALE * action
