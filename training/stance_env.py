"""Training environment for a stance policy: stand still, on a spot, while
somebody else moves your arms.

This is the capability the borrowed walking policy does not have. That policy
was trained to travel, so it creeps forward, holds a spot only to about
±0.2 m, and treats a sustained arm reach as a disturbance it must step away
from. Manipulation needs the opposite: hold the base within a couple of
centimetres while the arms swing through a full reach.

So the arms here are the *environment*, not the agent: on every episode the
arms are driven through random reaching motions, and the policy (legs and
waist only) is rewarded for not moving the pelvis.
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
CONTROLLED = LEGS + WAIST                      # 15 actuators the policy owns

CTRL_DT = 0.02                                 # 50 Hz control
EPISODE_S = 6.0
ACTION_SCALE = 0.4


class StanceEnv:
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
        self.feet = [m.body(f"{s}_ankle_roll_link").id for s in ("left", "right")]

        self.n_substeps = int(round(CTRL_DT / m.opt.timestep))
        self.max_steps = int(EPISODE_S / CTRL_DT)
        self.obs_dim = 3 + 3 + 3 + 15 + 15 + 14 + 14 + 15 + 2 + 2
        self.act_dim = len(CONTROLLED)

    # ---------------- helpers ----------------

    def _sensor(self, sid):
        adr = self.model.sensor_adr[sid]
        return self.data.sensordata[adr:adr + self.model.sensor_dim[sid]]

    def _obs(self):
        d = self.data
        gravity = d.site_xmat[self.imu].reshape(3, 3).T @ np.array([0, 0, -1.0])
        base_err = self.home_xy - d.xpos[self.pelvis][:2]
        yaw_err = self._yaw_err()
        return np.concatenate([
            gravity,
            self._sensor(self.gyro),
            self._sensor(self.linvel),
            d.qpos[self.qpos_adr] - self.default,
            d.qvel[self.qvel_adr] * 0.1,
            d.qpos[self.arm_qpos] - self.arm_default,
            d.qvel[self.arm_qvel] * 0.1,
            self.last_action,
            base_err * 5.0,
            [np.cos(yaw_err), np.sin(yaw_err)],
        ]).astype(np.float32)

    def _yaw_err(self) -> float:
        """Heading error. Holding position is not enough: a robot that slowly
        turns on the spot puts the table somewhere its arms are not, which is
        just as fatal to a reach as drifting away from it."""
        R = self.data.xmat[self.pelvis].reshape(3, 3)
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        return (self.home_yaw - yaw + np.pi) % (2 * np.pi) - np.pi

    def _new_arm_goal(self):
        """Sample the poses the robot actually adopts when it reaches.

        The first version sampled symmetrically around the arms' resting
        pose, which never produced a real reach: picking something off a
        bench puts the shoulder near -1.2 rad, well outside that range. The
        policy balanced beautifully against poses it had seen, and fell over
        the moment the arm went somewhere new — because it had never seen it.
        """
        # left arm bounds, per joint: shoulder pitch/roll/yaw, elbow,
        # wrist roll/pitch/yaw. Roll and yaw mirror for the right arm.
        low = np.array([-1.6, -0.5, -0.8, -0.3, -0.6, -0.8, -0.6])
        high = np.array([0.8, 1.0, 0.8, 1.8, 0.6, 0.8, 0.6])
        mirror = np.array([1, -1, -1, 1, 1, 1, 1])
        r_low = np.minimum(low * mirror, high * mirror)
        r_high = np.maximum(low * mirror, high * mirror)
        goal = np.concatenate([self.rng.uniform(low, high),
                               self.rng.uniform(r_low, r_high)])
        return np.clip(goal, self.arm_limits[:, 0], self.arm_limits[:, 1])

    # ---------------- gym-ish API ----------------

    def reset(self):
        d, m = self.data, self.model
        mujoco.mj_resetData(m, d)
        self.body.reset(d)
        # start from a randomly perturbed stance so the policy learns to
        # recover, not just to sit still in one exact configuration
        d.qpos[self.qpos_adr] += self.rng.uniform(-0.06, 0.06, self.act_dim)
        adr = m.jnt_qposadr[m.body_jntadr[self.pelvis]]
        d.qpos[adr + 2] += self.rng.uniform(-0.01, 0.02)
        d.qvel[:6] = self.rng.uniform(-0.05, 0.05, 6)
        mujoco.mj_forward(m, d)

        self.home_xy = d.xpos[self.pelvis][:2].copy()
        R = d.xmat[self.pelvis].reshape(3, 3)
        self.home_yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        self.last_action = np.zeros(self.act_dim, dtype=np.float32)
        self.arm_goal = self.arm_default.copy()
        self.arm_cmd = self.arm_default.copy()
        self.arm_timer = 0
        self.t = 0
        return self._obs()

    def step(self, action):
        d, m = self.data, self.model
        action = np.clip(action, -1.0, 1.0)
        target = self.default + ACTION_SCALE * action

        # the arms move on their own schedule — this is the disturbance
        if self.arm_timer <= 0:
            self.arm_goal = self._new_arm_goal()
            self.arm_timer = int(self.rng.uniform(0.8, 2.0) / CTRL_DT)
        self.arm_timer -= 1
        self.arm_cmd += np.clip(self.arm_goal - self.arm_cmd, -0.02, 0.02)

        d.ctrl[self.act] = target
        d.ctrl[self.arm_act] = self.arm_cmd
        for _ in range(self.n_substeps):
            mujoco.mj_step(m, d)

        pos = d.xpos[self.pelvis]
        gravity = d.site_xmat[self.imu].reshape(3, 3).T @ np.array([0, 0, -1.0])
        drift = float(np.linalg.norm(pos[:2] - self.home_xy))
        yaw_err = abs(self._yaw_err())
        upright = float(-gravity[2])                   # 1.0 when vertical
        base_vel = float(np.linalg.norm(d.qvel[:3]))
        rate = float(np.linalg.norm(action - self.last_action))
        feet_down = sum(
            1 for c in range(d.ncon)
            for g in (d.contact[c].geom1, d.contact[c].geom2)
            if m.geom_bodyid[g] in self.feet)

        # Drift shaping matters more than it looks. A narrow Gaussian alone
        # (exp(-(drift/0.05)^2)) is numerically zero past ~0.1 m, so a policy
        # that has drifted 0.3 m gets no gradient saying 0.15 m would be
        # better, and it plateaus at whatever lean it first found. The linear
        # term keeps a slope at every scale; the Gaussian sharpens the last
        # few centimetres.
        reward = (
            1.0                                        # alive
            + 2.0 * np.exp(-(drift / 0.12) ** 2)       # hold the spot
            - 2.5 * drift                              # ... at every scale
            + 1.0 * np.exp(-(yaw_err / 0.15) ** 2)     # and keep facing front
            - 1.5 * yaw_err
            + 1.0 * max(0.0, upright)                  # stay vertical
            + 0.5 * np.exp(-(base_vel / 0.15) ** 2)    # stand quietly
            - 0.15 * rate                              # smooth commands
            - 0.05 * float(np.linalg.norm(action))
            # Stand tall. Left alone the policy discovers that crouching is
            # stable, settles at a 0.64 m pelvis, and quietly costs the arm
            # ~8 cm of forward reach — which is the difference between the
            # robot being able to pick something up and not.
            + 1.5 * np.exp(-((pos[2] - 0.73) / 0.06) ** 2)
            - 1.0 * float(pos[2] < 0.62)               # do not sag
            + 0.2 * float(feet_down > 0)
        )

        self.last_action = action.astype(np.float32)
        self.t += 1
        fell = bool(pos[2] < 0.5 or upright < 0.6 or drift > 0.6
                    or yaw_err > 0.9)
        if fell:
            reward -= 8.0
        done = fell or self.t >= self.max_steps
        info = {"drift": drift, "fell": fell, "upright": upright,
                "yaw": yaw_err}
        return self._obs(), float(reward), done, info
