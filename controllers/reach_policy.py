"""Whole-body reaching, learned. Trained by `training/train_reach.py`.

Given a hand target the policy drives the arm *and* the legs and waist (22
joints), so it can lean and shift its stance to reach places an arm on its
own cannot — which is exactly what the deterministic IK path cannot express,
since only the arm moves there.
"""

from __future__ import annotations

import numpy as np
import torch

from training.reach_env import CONTROLLED, LEFT_ARM, LEGS, RIGHT_ARM, WAIST

ACTION_SCALE = 0.4
CTRL_DT = 0.02


class ReachPolicy:
    def __init__(self, model, checkpoint, stand_angles):
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        from training.ppo import ActorCritic

        self.net = ActorCritic(ck["obs_dim"], ck["act_dim"])
        self.net.load_state_dict(ck["state"])
        self.net.eval()
        self.mean = np.asarray(ck["mean"])
        self.std = np.sqrt(np.asarray(ck["var"]) + 1e-8)

        self.model = model
        order = LEGS + WAIST + LEFT_ARM + RIGHT_ARM
        stance = dict(zip(order, stand_angles))
        self.default = np.array([stance[n] for n in CONTROLLED])
        self.left_default = np.array([stance[n] for n in LEFT_ARM])

        self.act = np.array([model.actuator(n).id for n in CONTROLLED])
        self.qpos = np.array([model.joint(n).qposadr[0] for n in CONTROLLED])
        self.qvel = np.array([model.joint(n).dofadr[0] for n in CONTROLLED])
        self.left_act = np.array([model.actuator(n).id for n in LEFT_ARM])
        self.limits = np.array([model.joint(n).range for n in CONTROLLED])
        self.pelvis = model.body("pelvis").id
        self.imu = model.site("imu_in_pelvis").id
        self.gyro = model.sensor("gyro_pelvis").id
        self.linvel = model.sensor("local_linvel_pelvis").id
        self.hand = model.site("right_ee").id

        self.n_substeps = max(1, int(round(CTRL_DT / model.opt.timestep)))
        self.last_action = np.zeros(len(CONTROLLED), dtype=np.float32)
        self.target = None
        self.hand0 = None
        self.reach_time = 1.0
        self.elapsed = 0.0
        self._counter = 0

    def _sensor(self, data, sid):
        adr = self.model.sensor_adr[sid]
        return data.sensordata[adr:adr + self.model.sensor_dim[sid]]

    def _to_base(self, data, world_pos):
        R = data.xmat[self.pelvis].reshape(3, 3)
        return R.T @ (np.asarray(world_pos) - data.xpos[self.pelvis])

    def aim(self, world_target, hand_pos=None) -> None:
        """Begin a reach, exactly as the training episode does.

        The policy was trained to track a minimum-jerk reference sweeping
        from where the hand started to the goal, not to charge at a static
        point, so deployment has to hand it the same thing. Told instead to
        go straight to a target, a policy rewarded on final distance lunges:
        it hits peak speed in the first 5% of the motion and then dithers,
        tracing five times the direct distance. Correct, and nothing like a
        person.
        """
        self.target = np.asarray(world_target, dtype=float)
        self.hand0 = (self.target.copy() if hand_pos is None
                      else np.asarray(hand_pos, dtype=float))
        self.reach_time = float(np.clip(
            0.5 + 1.7 * np.sqrt(np.linalg.norm(self.target - self.hand0)),
            0.6, 2.6))
        self.elapsed = 0.0

    def reference(self) -> np.ndarray:
        """Where the hand should be now: the minimum-jerk point on the line."""
        s = min(1.0, self.elapsed / self.reach_time)
        u = s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)
        return self.hand0 + u * (self.target - self.hand0)

    def observe(self, data) -> np.ndarray:
        gravity = data.site_xmat[self.imu].reshape(3, 3).T @ np.array(
            [0.0, 0.0, -1.0])
        t_local = self._to_base(data, self.target)
        h_local = self._to_base(data, data.site_xpos[self.hand])
        r_local = self._to_base(data, self.reference())
        return np.concatenate([
            gravity,
            self._sensor(data, self.gyro),
            self._sensor(data, self.linvel),
            data.qpos[self.qpos] - self.default,
            data.qvel[self.qvel] * 0.1,
            t_local,
            r_local - h_local,
            self.last_action,
        ]).astype(np.float32)

    def step(self, data) -> None:
        if self.target is None:
            return
        self._counter += 1
        if self._counter % self.n_substeps:
            return
        self.elapsed += CTRL_DT
        obs = np.clip((self.observe(data) - self.mean) / self.std, -10, 10)
        with torch.no_grad():
            action = self.net.mu(self.net.pi(
                torch.as_tensor(obs, dtype=torch.float32))).numpy()
        action = np.clip(action, -1.0, 1.0)
        self.last_action = action.astype(np.float32)
        data.ctrl[self.act] = np.clip(self.default + ACTION_SCALE * action,
                                      self.limits[:, 0], self.limits[:, 1])
        data.ctrl[self.left_act] = self.left_default
