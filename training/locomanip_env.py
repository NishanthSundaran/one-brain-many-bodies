"""One policy for walking and for manipulating.

Until now the robot needed two: a pretrained joystick policy that travels but
cannot hold a spot or tolerate a reach, and a stance policy trained here that
holds a spot beautifully but only shuffles. Splitting them works, and it is
still two controllers with a handover between them.

This trains a single one. The policy owns the legs and waist; the arms are the
*environment*, driven through the poses a real reach visits. Every episode
mixes both jobs:

  * a velocity command it must track — including zero, which means "stand
    exactly here", the case the borrowed walking policy fails
  * arm motion throughout, so balancing against a reach is never a surprise

A gait phase is supplied as an observation and rewarded through foot air time,
which is what gets a stepping gait rather than a shuffle.
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
ARMS = [f"{s}_{j}_joint" for s in ("left", "right")
        for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                  "wrist_roll", "wrist_pitch", "wrist_yaw")]
CONTROLLED = LEGS + WAIST

CTRL_DT = 0.02
EPISODE_S = 8.0
ACTION_SCALE = 0.4
GAIT_HZ = 1.4
STAND_FRACTION = 0.45      # how often the command is "stand exactly here"


class LocoManipEnv:
    def __init__(self, seed: int = 0, body: str = "g1_walk"):
        from bodies.body import Body

        self.rng = np.random.default_rng(seed)
        b = Body(REPO / "bodies" / body / "config.yaml")
        w = b.spec.worldbody
        w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                   size=[12, 12, 0.1])
        self.model = b.spec.compile()
        self.data = mujoco.MjData(self.model)
        b.bind(self.model)
        self.body = b

        cfg = b.cfg["locomotion"]
        stance = dict(zip(LEGS + WAIST + ARMS, cfg["stand_angles"]))
        self.default = np.array([stance[n] for n in CONTROLLED])
        self.arm_default = np.array([stance[n] for n in ARMS])

        m = self.model
        self.act = np.array([m.actuator(n).id for n in CONTROLLED])
        self.qpos_adr = np.array([m.joint(n).qposadr[0] for n in CONTROLLED])
        self.qvel_adr = np.array([m.joint(n).dofadr[0] for n in CONTROLLED])
        self.arm_act = np.array([m.actuator(n).id for n in ARMS])
        self.arm_qpos = np.array([m.joint(n).qposadr[0] for n in ARMS])
        self.arm_qvel = np.array([m.joint(n).dofadr[0] for n in ARMS])
        self.arm_limits = np.array([m.joint(n).range for n in ARMS])
        self.pelvis = m.body("pelvis").id
        self.imu = m.site("imu_in_pelvis").id
        self.gyro = m.sensor("gyro_pelvis").id
        self.linvel = m.sensor("local_linvel_pelvis").id
        self.feet = [m.body(f"{s}_ankle_roll_link").id
                     for s in ("left", "right")]

        self.n_substeps = int(round(CTRL_DT / m.opt.timestep))
        self.max_steps = int(EPISODE_S / CTRL_DT)
        self.obs_dim = 3 + 3 + 3 + 3 + 15 + 15 + 14 + 14 + 15 + 4
        self.act_dim = len(CONTROLLED)
        self.phase_dt = 2 * np.pi * GAIT_HZ * CTRL_DT

    # ---------------- helpers ----------------

    def _sensor(self, sid):
        adr = self.model.sensor_adr[sid]
        return self.data.sensordata[adr:adr + self.model.sensor_dim[sid]]

    def _yaw(self) -> float:
        R = self.data.xmat[self.pelvis].reshape(3, 3)
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def _obs(self):
        d = self.data
        gravity = d.site_xmat[self.imu].reshape(3, 3).T @ np.array([0, 0, -1.0])
        return np.concatenate([
            gravity,
            self._sensor(self.gyro),
            self._sensor(self.linvel),
            self.command,
            d.qpos[self.qpos_adr] - self.default,
            d.qvel[self.qvel_adr] * 0.1,
            d.qpos[self.arm_qpos] - self.arm_default,
            d.qvel[self.arm_qvel] * 0.1,
            self.last_action,
            np.concatenate([np.cos(self.phase), np.sin(self.phase)]),
        ]).astype(np.float32)

    def _new_arm_goal(self):
        """The poses a real reach visits, not a wiggle around the rest pose."""
        low = np.array([-1.6, -0.5, -0.8, -0.3, -0.6, -0.8, -0.6])
        high = np.array([0.8, 1.0, 0.8, 1.8, 0.6, 0.8, 0.6])
        mirror = np.array([1, -1, -1, 1, 1, 1, 1])
        r_low = np.minimum(low * mirror, high * mirror)
        r_high = np.maximum(low * mirror, high * mirror)
        goal = np.concatenate([self.rng.uniform(low, high),
                               self.rng.uniform(r_low, r_high)])
        return np.clip(goal, self.arm_limits[:, 0], self.arm_limits[:, 1])

    def _new_command(self):
        """Half the time: stand exactly here. Otherwise: go somewhere."""
        if self.rng.random() < STAND_FRACTION:
            self.command = np.zeros(3, dtype=np.float32)
        else:
            self.command = np.array([
                self.rng.uniform(-0.4, 0.9),
                self.rng.uniform(-0.35, 0.35),
                self.rng.uniform(-0.8, 0.8)], dtype=np.float32)
        # Whether this counts as standing is decided once, here — not
        # re-derived from the command each step. A "moving" command can be
        # sampled small enough to read as standing, and then there is no
        # mark to hold and the reward crashes.
        self.standing = bool(np.linalg.norm(self.command[:2]) < 0.05
                             and abs(self.command[2]) < 0.05)
        self.hold_xy = self.data.xpos[self.pelvis][:2].copy()
        self.hold_yaw = self._yaw()
        self.cmd_timer = int(self.rng.uniform(2.5, 4.5) / CTRL_DT)

    # ---------------- gym-ish API ----------------

    def reset(self):
        d, m = self.data, self.model
        mujoco.mj_resetData(m, d)
        self.body.reset(d)
        d.qpos[self.qpos_adr] += self.rng.uniform(-0.06, 0.06, self.act_dim)
        adr = m.jnt_qposadr[m.body_jntadr[self.pelvis]]
        d.qpos[adr + 2] += self.rng.uniform(-0.01, 0.02)
        d.qvel[:6] = self.rng.uniform(-0.05, 0.05, 6)
        mujoco.mj_forward(m, d)

        self.last_action = np.zeros(self.act_dim, dtype=np.float32)
        self.phase = np.array([0.0, np.pi])
        self.arm_goal = self.arm_default.copy()
        self.arm_cmd = self.arm_default.copy()
        self.arm_timer = 0
        self.air_time = np.zeros(2)
        self.command = np.zeros(3, dtype=np.float32)
        self.standing = True
        self.hold_xy = d.xpos[self.pelvis][:2].copy()
        self.hold_yaw = self._yaw()
        self._new_command()
        self.t = 0
        return self._obs()

    def step(self, action):
        d, m = self.data, self.model
        action = np.clip(action, -1.0, 1.0)
        d.ctrl[self.act] = self.default + ACTION_SCALE * action

        if self.cmd_timer <= 0:
            self._new_command()
        self.cmd_timer -= 1
        if self.arm_timer <= 0:
            self.arm_goal = self._new_arm_goal()
            self.arm_timer = int(self.rng.uniform(0.8, 2.0) / CTRL_DT)
        self.arm_timer -= 1
        self.arm_cmd += np.clip(self.arm_goal - self.arm_cmd, -0.02, 0.02)
        d.ctrl[self.arm_act] = self.arm_cmd

        for _ in range(self.n_substeps):
            mujoco.mj_step(m, d)
        self.phase = np.fmod(self.phase + self.phase_dt + np.pi,
                             2 * np.pi) - np.pi

        pos = d.xpos[self.pelvis]
        gravity = d.site_xmat[self.imu].reshape(3, 3).T @ np.array([0, 0, -1.0])
        upright = float(-gravity[2])
        local_v = self._sensor(self.linvel)
        gyro = self._sensor(self.gyro)

        # feet contact and air time — this is what buys a gait rather than a
        # shuffle when the robot is asked to travel
        down = [False, False]
        for c in range(d.ncon):
            for g in (d.contact[c].geom1, d.contact[c].geom2):
                b = m.geom_bodyid[g]
                if b in self.feet:
                    down[self.feet.index(b)] = True
        air_reward = 0.0
        for i, on_ground in enumerate(down):
            if on_ground:
                if 0.15 < self.air_time[i] < 0.6:
                    air_reward += 0.6
                self.air_time[i] = 0.0
            else:
                self.air_time[i] += CTRL_DT

        moving = not self.standing
        v_err = float(np.linalg.norm(local_v[:2] - self.command[:2]))
        w_err = abs(float(gyro[2] - self.command[2]))
        reward = (
            1.0                                          # alive
            + 1.8 * np.exp(-(v_err / 0.35) ** 2)         # go at the asked speed
            + 0.8 * np.exp(-(w_err / 0.4) ** 2)          # turn at the asked rate
            + 1.0 * max(0.0, upright)
            + 1.2 * np.exp(-((pos[2] - 0.73) / 0.07) ** 2)   # stand tall
            - 0.12 * float(np.linalg.norm(action - self.last_action))
            - 0.04 * float(np.linalg.norm(action))
        )
        drift = 0.0
        if moving:
            reward += air_reward
        else:
            # "stand exactly here": the job the walking policy cannot do
            drift = float(np.linalg.norm(pos[:2] - self.hold_xy))
            yaw_err = abs((self.hold_yaw - self._yaw() + np.pi)
                          % (2 * np.pi) - np.pi)
            reward += (2.0 * np.exp(-(drift / 0.12) ** 2) - 2.5 * drift
                       + 0.8 * np.exp(-(yaw_err / 0.15) ** 2) - 1.0 * yaw_err)

        self.last_action = action.astype(np.float32)
        self.t += 1
        fell = bool(pos[2] < 0.5 or upright < 0.6 or (not moving and drift > 0.7))
        if fell:
            reward -= 8.0
        done = fell or self.t >= self.max_steps
        return self._obs(), float(reward), done, {
            "fell": fell, "drift": drift, "v_err": v_err,
            "standing": not moving}
