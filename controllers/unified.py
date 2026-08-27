"""One controller for walking and for manipulating.

Trained by `training/train_locomanip.py`. It takes a base velocity command,
exactly like the pretrained walking policy — but zero means *stand exactly
here*, which the walking policy cannot do (it creeps forward at ~0.25 m/s and
wanders in a ±0.2 m band), and it was trained with the arms moving through
reaching poses throughout, so a reach is not a disturbance it has never seen.

That removes the handover: walking, standing and manipulating are the same
controller with different commands, so nothing has to decide when to switch
and nothing can fall over in the switch.
"""

from __future__ import annotations

import numpy as np
import torch

from controllers.locomotion import ARMS, LEGS, WAIST

CONTROLLED = LEGS + WAIST
CTRL_DT = 0.02
ACTION_SCALE = 0.4
GAIT_HZ = 1.4


class UnifiedPolicy:
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
        self.command = np.zeros(3, dtype=np.float32)
        self.last_action = np.zeros(len(CONTROLLED), dtype=np.float32)
        self.phase = np.array([0.0, np.pi])
        self.phase_dt = 2 * np.pi * GAIT_HZ * CTRL_DT
        self._counter = 0

    def _sensor(self, data, sid):
        adr = self.model.sensor_adr[sid]
        return data.sensordata[adr:adr + self.model.sensor_dim[sid]]

    def observe(self, data) -> np.ndarray:
        gravity = data.site_xmat[self.imu].reshape(3, 3).T @ np.array(
            [0.0, 0.0, -1.0])
        return np.concatenate([
            gravity,
            self._sensor(data, self.gyro),
            self._sensor(data, self.linvel),
            self.command,
            data.qpos[self.qpos] - self.default,
            data.qvel[self.qvel] * 0.1,
            data.qpos[self.arm_qpos] - self.arm_default,
            data.qvel[self.arm_qvel] * 0.1,
            self.last_action,
            np.concatenate([np.cos(self.phase), np.sin(self.phase)]),
        ]).astype(np.float32)

    def set_command(self, vx: float = 0.0, vy: float = 0.0,
                    wz: float = 0.0) -> None:
        self.command[:] = (vx, vy, wz)

    def step(self, data) -> None:
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
        self.phase = np.fmod(self.phase + self.phase_dt + np.pi,
                             2 * np.pi) - np.pi

    def policy_joint_names(self) -> list:
        return CONTROLLED
