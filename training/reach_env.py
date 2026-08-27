"""Whole-body reaching: put the hand on a target, using the whole body.

The deterministic stack solves the same problem with IK, and does it
accurately — but only inside an envelope I had to measure by hand, with the
scene tuned to a few centimetres of slack around it. Objects inboard of the
shoulder make the IK roll the upper arm into the torso; objects past ~0.40 m
are simply unreachable, because the arm alone is all that moves.

Here the policy owns the arm *and* the legs and waist (22 joints), so it can
lean, bend and shift its stance to get the hand where it needs to be, and it
has to keep its balance while doing it. That is the thing an arm-only IK
cannot do, and the reason to spend RL on reaching rather than on grasping.

Reward: hand-to-target distance, staying upright, staying near the spot it
started from. Targets are sampled across the workspace in front of the robot,
including places a fixed arm cannot get to.
"""

from __future__ import annotations

import pathlib

import mujoco
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]

LEGS = [f"{s}_{j}_joint" for s in ("left", "right")
        for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee",
                  "ankle_pitch", "ankle_roll")]
WAIST = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
RIGHT_ARM = [f"right_{j}_joint" for j in
             ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
              "wrist_roll", "wrist_pitch", "wrist_yaw")]
LEFT_ARM = [f"left_{j}_joint" for j in
            ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
             "wrist_roll", "wrist_pitch", "wrist_yaw")]
CONTROLLED = LEGS + WAIST + RIGHT_ARM        # 22 joints: legs, waist, one arm

CTRL_DT = 0.02
EPISODE_S = 5.0
ACTION_SCALE = 0.4
REACH_TOL = 0.05


class ReachEnv:
    def __init__(self, seed: int = 0, body: str = "g1_walk"):
        from bodies.body import Body

        self.rng = np.random.default_rng(seed)
        b = Body(REPO / "bodies" / body / "config.yaml")
        w = b.spec.worldbody
        w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                   size=[8, 8, 0.1])
        self.model = b.spec.compile()
        self.data = mujoco.MjData(self.model)
        b.bind(self.model)
        self.body = b

        cfg = b.cfg["locomotion"]
        stance = dict(zip(LEGS + WAIST + LEFT_ARM + RIGHT_ARM,
                          cfg["stand_angles"]))
        # stand_angles is in the canonical order legs, waist, left, right
        order = LEGS + WAIST + LEFT_ARM + RIGHT_ARM
        stance = dict(zip(order, cfg["stand_angles"]))
        self.default = np.array([stance[n] for n in CONTROLLED])
        self.left_default = np.array([stance[n] for n in LEFT_ARM])

        m = self.model
        self.act = np.array([m.actuator(n).id for n in CONTROLLED])
        self.qpos_adr = np.array([m.joint(n).qposadr[0] for n in CONTROLLED])
        self.qvel_adr = np.array([m.joint(n).dofadr[0] for n in CONTROLLED])
        self.left_act = np.array([m.actuator(n).id for n in LEFT_ARM])
        self.limits = np.array([m.joint(n).range for n in CONTROLLED])
        self.pelvis = m.body("pelvis").id
        self.imu = m.site("imu_in_pelvis").id
        self.gyro = m.sensor("gyro_pelvis").id
        self.linvel = m.sensor("local_linvel_pelvis").id
        self.hand = m.site("right_ee").id

        self.n_substeps = int(round(CTRL_DT / m.opt.timestep))
        self.max_steps = int(EPISODE_S / CTRL_DT)
        self.obs_dim = 3 + 3 + 3 + 22 + 22 + 3 + 3 + 22
        self.act_dim = len(CONTROLLED)

    def _sensor(self, sid):
        adr = self.model.sensor_adr[sid]
        return self.data.sensordata[adr:adr + self.model.sensor_dim[sid]]

    def _hand_pos(self):
        return self.data.site_xpos[self.hand].copy()

    def _to_base(self, world_pos):
        R = self.data.xmat[self.pelvis].reshape(3, 3)
        return R.T @ (np.asarray(world_pos) - self.data.xpos[self.pelvis])

    def _obs(self):
        d = self.data
        gravity = d.site_xmat[self.imu].reshape(3, 3).T @ np.array([0, 0, -1.0])
        target_local = self._to_base(self.target)
        hand_local = self._to_base(self._hand_pos())
        return np.concatenate([
            gravity,
            self._sensor(self.gyro),
            self._sensor(self.linvel),
            d.qpos[self.qpos_adr] - self.default,
            d.qvel[self.qvel_adr] * 0.1,
            target_local,
            target_local - hand_local,        # what is left to close
            self.last_action,
        ]).astype(np.float32)

    def _new_target(self):
        """Anywhere useful in front of the robot, including places an arm on
        its own cannot get to — that is the point of using the whole body."""
        base = self.data.xpos[self.pelvis]
        R = self.data.xmat[self.pelvis].reshape(3, 3)
        local = np.array([self.rng.uniform(0.20, 0.52),
                          self.rng.uniform(-0.34, 0.02),
                          self.rng.uniform(0.02, 0.32)])
        return base + R @ local

    def reset(self):
        d, m = self.data, self.model
        mujoco.mj_resetData(m, d)
        self.body.reset(d)
        d.qpos[self.qpos_adr] += self.rng.uniform(-0.05, 0.05, self.act_dim)
        mujoco.mj_forward(m, d)
        self.home_xy = d.xpos[self.pelvis][:2].copy()
        self.last_action = np.zeros(self.act_dim, dtype=np.float32)
        self.target = self._new_target()
        self.t = 0
        self.best = np.linalg.norm(self.target - self._hand_pos())
        return self._obs()

    def step(self, action):
        d, m = self.data, self.model
        action = np.clip(action, -1.0, 1.0)
        d.ctrl[self.act] = np.clip(self.default + ACTION_SCALE * action,
                                   self.limits[:, 0], self.limits[:, 1])
        d.ctrl[self.left_act] = self.left_default     # other arm stays put
        for _ in range(self.n_substeps):
            mujoco.mj_step(m, d)

        pos = d.xpos[self.pelvis]
        gravity = d.site_xmat[self.imu].reshape(3, 3).T @ np.array([0, 0, -1.0])
        upright = float(-gravity[2])
        dist = float(np.linalg.norm(self.target - self._hand_pos()))
        drift = float(np.linalg.norm(pos[:2] - self.home_xy))

        reward = (
            1.0                                          # alive
            + 3.0 * np.exp(-(dist / 0.10) ** 2)          # get the hand there
            - 1.5 * dist                                 # ... at every scale
            + 1.5 * float(dist < REACH_TOL)              # and hold it there
            + 1.0 * max(0.0, upright)
            + 0.8 * np.exp(-((pos[2] - 0.70) / 0.08) ** 2)
            + 1.0 * np.exp(-(drift / 0.15) ** 2)         # keep your feet
            - 1.2 * drift
            - 0.10 * float(np.linalg.norm(action - self.last_action))
            - 0.03 * float(np.linalg.norm(action))
        )
        self.last_action = action.astype(np.float32)
        self.t += 1
        self.best = min(self.best, dist)
        fell = bool(pos[2] < 0.5 or upright < 0.6 or drift > 0.7)
        if fell:
            reward -= 8.0
        done = fell or self.t >= self.max_steps
        return self._obs(), float(reward), done, {
            "fell": fell, "drift": dist, "reached": bool(dist < REACH_TOL),
            "best": float(self.best)}
