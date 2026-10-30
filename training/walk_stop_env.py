"""Training environment: teach the walking policy to stop when told to.

The borrowed Playground policy, commanded to zero, keeps stepping and creeps
at ~0.3 m/s for as long as it is left (up to 3.9 m in 8 s, measured in open
floor). Handing the legs to the stance policy while it creeps drops the
robot about half the time. So the fix belongs in the walking policy: on a
zero command it must come to rest itself, after which the stance policy is
handed a robot that is already standing.

Each episode walks under a random command for 1.5 to 3.5 s, eases the
command to zero over 0 to 0.6 s (the runtime ramps over 0.6 s), then holds
zero for the rest. One episode in five never walks. Rewards: velocity
tracking while walking, being at rest once stopped, smoothness throughout.
Same plant, same observation and the same fall check as the runtime.

`priv` (time since the stop, stop flag) goes to the critic only; the actor
sees exactly what LocomotionPolicy.observe builds.
"""

from __future__ import annotations

import os
import pathlib

import mujoco
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
EPISODE_S = float(os.environ.get("WALKSTOP_EPISODE_S", 8.0))
STAND_ONLY_PROB = 0.2
# Stage 2: once stopped, the arms reach the way manipulation moves them and
# the policy keeps only the legs and waist (the runtime's "leg_waist" mode,
# arms stiffened by ARM_GAIN as Runtime._set_arm_gain does). The policy then
# holds the robot for manipulation itself and the stance policy, which
# cannot take over from a walking robot, is no longer needed.
ARMS = bool(int(os.environ.get("WALKSTOP_ARMS", 0)))
ARM_GAIN = float(os.environ.get("WALKSTOP_ARM_GAIN", 3.0))
ARMS_AFTER_S = 1.0
# Stage 3: goals from training/arm_goal_bank.npz (poses the runtime's IK
# actually produces for bench targets, none jammed into the torso), moved
# at the runtime's pace, with the arm gravity torque fed forward as
# Runtime._gravity_comp does.
GOALS = os.environ.get("WALKSTOP_GOALS")
ARM_RATE = float(os.environ.get("WALKSTOP_ARM_RATE", 0.02))   # rad per tick
GRAV_COMP = bool(int(os.environ.get("WALKSTOP_GRAV", 0)))
# Human-like walking (measured against LAFAN1 in eval/score_walk_quality.py:
# 2.6 steps/s against 1.9, knees to 64 degrees, 39% of leg motion above
# 8 Hz). GAIT_FREQ "lo,hi" samples the gait clock per episode (cadence is
# twice it; the borrowed policy falls below 1.2 untrained). HUMAN_W weights
# style terms while walking: pelvis height, torso upright, swing foot no
# higher than needed. ACT_FILTER low-passes the joint targets as the stance
# policy does (1.0 = off).
GAIT_FREQ = os.environ.get("WALKSTOP_GAIT_FREQ")
HUMAN_W = float(os.environ.get("WALKSTOP_HUMAN_W", 0.0))
WALK_HEIGHT = float(os.environ.get("WALKSTOP_HEIGHT", 0.76))
ACT_FILTER = float(os.environ.get("WALKSTOP_ACT_FILTER", 1.0))
TRACK_W = float(os.environ.get("WALKSTOP_TRACK_W", 0.0))
# How long each episode walks before the stop. Demo 13 walks ~5 s and the
# human-gait policy stumbled at 4 to 5 s, past the 1.5 to 3.5 s it trained on.
WALK_S = [float(x) for x in os.environ.get("WALKSTOP_WALK_S", "1.5,3.5").split(",")]
# Random shoves on the pelvis while walking (peak N, mean interval s): a gait
# that only ever saw a perfect floor stumbles on its own small errors.
PUSH = [float(x) for x in os.environ.get("WALKSTOP_PUSH", "0,0").split(",")]


class WalkStopEnv:
    def __init__(self, seed: int = 0, body: str = "g1_walk"):
        from bodies.body import Body
        from controllers.locomotion import POLICY_JOINTS

        self.rng = np.random.default_rng(seed)
        b = Body(REPO / "bodies" / body / "config.yaml")
        b.spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                                  size=[20, 20, 0.1])
        self.model = m = b.spec.compile()
        self.data = mujoco.MjData(m)
        b.bind(m)
        self.body = b
        cfg = b.cfg["locomotion"]
        self.default = np.asarray(cfg["stand_angles"], np.float32)
        self.action_scale = cfg["action_scale"]
        self.ctrl_dt = cfg["ctrl_dt"]
        self.n_substeps = int(round(self.ctrl_dt / m.opt.timestep))
        self.phase_dt = 2 * np.pi * cfg["gait_freq"] * self.ctrl_dt
        self.phase_dt0 = self.phase_dt
        self.torso = m.body("torso_link").id
        self.qpos_adr = np.array([m.joint(n).qposadr[0] for n in POLICY_JOINTS])
        self.qvel_adr = np.array([m.joint(n).dofadr[0] for n in POLICY_JOINTS])
        self.act_ids = np.array([m.actuator(n).id for n in POLICY_JOINTS])
        self.root = m.body(cfg.get("root_body", "pelvis")).id if False else m.body("pelvis").id
        self.imu = m.site("imu_in_pelvis").id
        self.gyro = m.sensor("gyro_pelvis").id
        self.linvel = m.sensor("local_linvel_pelvis").id
        self.feet = [m.body(f).id for f in cfg["feet"]]
        self.max_steps = int(EPISODE_S / self.ctrl_dt)
        arm_names = [n for n in POLICY_JOINTS if "shoulder" in n
                     or "elbow" in n or "wrist" in n]
        self.arm_idx = np.array([POLICY_JOINTS.index(n) for n in arm_names])
        self.arm_qpos = self.qpos_adr[self.arm_idx]
        self.arm_act = self.act_ids[self.arm_idx]
        self.arm_limits = np.array([m.joint(n).range for n in arm_names])
        self.kp0 = m.actuator_gainprm[self.arm_act, 0].copy()
        self.arm_dofs = self.qvel_adr[self.arm_idx]
        self.bank = None
        if GOALS:
            bank = np.load(REPO / GOALS)
            right = bank["right"]
            mirror = np.array([1, -1, -1, 1, -1, 1, -1])
            self.bank = {"right": right, "left": right * mirror}
            self.ready = {"right": bank["ready"], "left": bank["ready"] * mirror}
            self.side_of = {"left": slice(0, 7), "right": slice(7, 14)}
            assert arm_names[0].startswith("left") and arm_names[7].startswith("right")
        # Reach curriculum, per worker: full-range reaches from the start
        # fell in every episode for 3000 iterations, which teaches nothing.
        # The reach amplitude grows after an episode that stays up and
        # shrinks after a fall.
        self.reach = float(os.environ.get("WALKSTOP_REACH0", 0.0))
        self.obs_dim, self.act_dim, self.priv_dim = 103, 29, 2

    def _sensor(self, sid):
        adr = self.model.sensor_adr[sid]
        return self.data.sensordata[adr:adr + self.model.sensor_dim[sid]]

    def _obs(self):
        d = self.data
        gravity = d.site_xmat[self.imu].reshape(3, 3).T @ np.array([0, 0, -1.0])
        return np.hstack([
            self._sensor(self.linvel), self._sensor(self.gyro), gravity,
            self.command, d.qpos[self.qpos_adr] - self.default,
            d.qvel[self.qvel_adr], self.last_action,
            np.cos(self.phase), np.sin(self.phase)]).astype(np.float32)

    def _priv(self):
        since = max(0.0, (self.t - self.stop_at) * self.ctrl_dt)
        return np.array([min(since, 4.0) / 4.0,
                         float(self.t >= self.stop_at)], np.float32)

    def reset(self):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        self.body.reset(d)
        d.qpos[self.qpos_adr] += self.rng.uniform(-0.02, 0.02, 29)
        mujoco.mj_forward(m, d)
        p = self.rng.uniform(-np.pi, np.pi)
        self.phase = np.array([p, np.fmod(p + 2 * np.pi, 2 * np.pi) - np.pi])
        self.last_action = np.zeros(29, np.float32)
        self.filt = d.qpos[self.qpos_adr].copy()
        if GAIT_FREQ:
            lo, hi = (float(x) for x in GAIT_FREQ.split(","))
            self.phase_dt = 2 * np.pi * self.rng.uniform(lo, hi) * self.ctrl_dt
        if self.rng.random() < STAND_ONLY_PROB:
            self.walk_cmd = np.zeros(3)
            self.stop_at, self.ramp = 0, 1
        else:
            self.walk_cmd = np.array([self.rng.uniform(0.2, 0.75), 0.0,
                                      self.rng.uniform(-0.5, 0.5)])
            self.stop_at = int(self.rng.uniform(*WALK_S) / self.ctrl_dt)
            self.ramp = max(1, int(self.rng.uniform(0.0, 0.6) / self.ctrl_dt))
        self.t = 0
        self.arms_on = False
        self.rest_xy = None
        self._set_arm_gain(1.0)
        self.command = self._cmd()
        return self._obs(), self._priv()

    def _set_arm_gain(self, k):
        m = self.model
        m.actuator_gainprm[self.arm_act, 0] = self.kp0 * k
        m.actuator_biasprm[self.arm_act, 1] = -self.kp0 * k

    def _new_arm_goal(self):
        """Reach poses, as training/stance_env.py samples them."""
        if self.bank is not None:
            # one arm works, the other holds where it was (a grasp, a place)
            goal = self.arm_home.copy()
            side = "right" if self.rng.random() < 0.7 else "left"
            sl = self.side_of[side]
            pick = (self.ready[side] if self.rng.random() < 0.2 else
                    self.bank[side][self.rng.integers(len(self.bank[side]))])
            goal[sl] = self.arm_home[sl] + self.ep_reach * (pick - self.arm_home[sl])
            return goal
        low = np.array([-1.6, -0.5, -0.8, -0.3, -0.6, -0.8, -0.6])
        high = np.array([0.8, 1.0, 0.8, 1.8, 0.6, 0.8, 0.6])
        mirror = np.array([1, -1, -1, 1, 1, 1, 1])
        r_low = np.minimum(low * mirror, high * mirror)
        r_high = np.maximum(low * mirror, high * mirror)
        goal = np.concatenate([self.rng.uniform(low, high),
                               self.rng.uniform(r_low, r_high)])
        if self.rng.random() < 0.5:       # one arm reaches, like a grasp
            k = self.rng.integers(2)
            goal[7 * k:7 * k + 7] = self.arm_home[7 * k:7 * k + 7]
        goal = self.arm_home + self.ep_reach * (goal - self.arm_home)
        return np.clip(goal, self.arm_limits[:, 0], self.arm_limits[:, 1])

    def _cmd(self):
        k = (self.t - self.stop_at) / self.ramp
        return (self.walk_cmd * float(np.clip(1.0 - k, 0.0, 1.0))
                ).astype(np.float32)

    def step(self, u):
        """`u` is the pre-tanh action (the PPO sample space)."""
        m, d = self.model, self.data
        a = np.tanh(u).astype(np.float32)
        since = (self.t - self.stop_at - self.ramp) * self.ctrl_dt
        if ARMS and not self.arms_on and since >= ARMS_AFTER_S:
            self.arms_on = True
            self._set_arm_gain(ARM_GAIN)
            self.arm_cmd = d.qpos[self.arm_qpos].copy()
            self.arm_home = self.arm_cmd.copy()
            self.ep_reach = self.rng.uniform(0.0, self.reach)
            self.arm_goal, self.arm_timer = self.arm_cmd.copy(), 0
            self.rest_xy = d.xpos[self.root][:2].copy()
        target = a * self.action_scale + self.default
        prev = self.filt.copy()
        self.filt = ACT_FILTER * target + (1.0 - ACT_FILTER) * self.filt
        d.ctrl[self.act_ids] = target if ACT_FILTER >= 1.0 else prev
        if self.arms_on:
            if self.arm_timer <= 0:
                self.arm_goal = (self._new_arm_goal()
                                 if self.rng.random() < 0.8 else self.arm_cmd)
                self.arm_timer = int(self.rng.uniform(0.8, 2.0) / self.ctrl_dt)
            self.arm_timer -= 1
            self.arm_cmd += np.clip(self.arm_goal - self.arm_cmd,
                                    -ARM_RATE, ARM_RATE)
            d.ctrl[self.arm_act] = self.arm_cmd
        legs = (np.setdiff1d(np.arange(29), self.arm_idx) if self.arms_on
                else np.arange(29))
        d.xfrc_applied[self.root] = 0.0
        if PUSH[0] > 0 and self.t < self.stop_at and self.rng.random() < (
                self.ctrl_dt / max(PUSH[1], 1e-3)):
            ang = self.rng.uniform(-np.pi, np.pi)
            self.push_left = int(0.1 / self.ctrl_dt)          # 0.1 s shove
            self.push_f = self.rng.uniform(0.3, 1.0) * PUSH[0] * np.array(
                [np.cos(ang), np.sin(ang), 0.0])
        if getattr(self, "push_left", 0) > 0:
            d.xfrc_applied[self.root, :3] = self.push_f
            self.push_left -= 1
        for k in range(self.n_substeps):
            if ACT_FILTER < 1.0:
                d.ctrl[self.act_ids[legs]] = (prev + (self.filt - prev)
                                              * (k + 1) / self.n_substeps)[legs]
            if GRAV_COMP:
                d.qfrc_applied[self.arm_dofs] = (
                    d.qfrc_bias[self.arm_dofs] if self.arms_on else 0.0)
            mujoco.mj_step(m, d)
        self.phase = np.fmod(self.phase + self.phase_dt + np.pi,
                             2 * np.pi) - np.pi
        self.t += 1

        R = d.xmat[self.root].reshape(3, 3)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        v = d.qvel[:2]
        v_local = np.array([np.cos(yaw) * v[0] + np.sin(yaw) * v[1],
                            -np.sin(yaw) * v[0] + np.cos(yaw) * v[1]])
        wz = d.qvel[5]
        rate = float(np.sum((a - self.last_action) ** 2))
        stopped = self.t >= self.stop_at + self.ramp
        if not stopped:
            r = (2.0 * np.exp(-np.sum((v_local - self.command[:2]) ** 2) / 0.05)
                 + 1.0 * np.exp(-(wz - self.command[2]) ** 2 / 0.1))
            if HUMAN_W > 0:
                z = d.xpos[self.root][2]
                tilt = float(np.arccos(np.clip(d.xmat[self.torso][8], -1, 1)))
                swing = sum(max(0.0, d.xpos[f][2] - 0.12) for f in self.feet)
                r += HUMAN_W * (np.exp(-((z - WALK_HEIGHT) / 0.03) ** 2)
                                - 3.0 * max(0.0, WALK_HEIGHT - 0.02 - z)
                                + 0.5 * np.exp(-(tilt / 0.08) ** 2)
                                - 2.0 * tilt - 10.0 * swing
                                # steer as told: the first human-gait round
                                # drifted 15 to 40 cm sideways over 2.8 m
                                - TRACK_W * abs(v_local[1] - self.command[1])
                                - 0.5 * TRACK_W * abs(wz - self.command[2]))
        else:
            # come to rest: body still, feet still and on the floor
            since = (self.t - self.stop_at - self.ramp) * self.ctrl_dt
            speed = float(np.linalg.norm(v))
            foot_v = sum(float(np.linalg.norm(d.cvel[f][3:5])) for f in self.feet)
            foot_z = [d.xpos[f][2] for f in self.feet]
            down = sum(z < 0.09 for z in foot_z)
            w = min(1.0, 0.3 + since / 1.5)     # allow ~1 s to brake
            r = (w * (2.0 * np.exp(-(speed / 0.08) ** 2) - 1.0 * speed
                      - 0.3 * foot_v + 0.25 * down)
                 + 0.5 * np.exp(-wz ** 2 / 0.05))
            if self.rest_xy is not None:     # hold the spot through the reach
                drift = float(np.linalg.norm(d.xpos[self.root][:2] - self.rest_xy))
                r += 1.5 * np.exp(-(drift / 0.05) ** 2) - 3.0 * drift
        r += 1.0 + 0.5 * R[2, 2] - 0.02 * rate - 0.2 * float(np.sum(d.qvel[3:5] ** 2) > 1.0)

        self.last_action = a
        self.command = self._cmd()
        fell = bool(d.xpos[self.root][2] < 0.55 or R[2, 2] < 0.7)
        if fell:
            r -= 10.0
        done = fell or self.t >= self.max_steps
        if done and self.arms_on:
            self.reach = (max(0.0, self.reach - 0.1) if fell
                          else min(1.0, self.reach + 0.05))
        info = {"fell": fell, "stopped": stopped, "arms": self.arms_on,
                "fell_walking": bool(fell and not stopped),
                "reach": self.reach,
                "drift": (float(np.linalg.norm(d.xpos[self.root][:2] - self.rest_xy))
                          if self.rest_xy is not None else 0.0),
                "speed": float(np.linalg.norm(v))}
        return self._obs(), self._priv(), float(r), done, info
