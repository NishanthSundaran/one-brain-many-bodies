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
import os

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
EPISODE_S = float(os.environ.get("STANCE_EPISODE_S", 6.0))
# Fraction of episodes given a goal to walk to rather than a mark to hold.
# 0.0 reproduces the shipped stance policy exactly; set it to train a
# travel-capable policy (see the README section on the handover).
TRAVEL_PROB = float(os.environ.get("STANCE_TRAVEL_PROB", 0.0))
GOAL_MAX = float(os.environ.get("STANCE_GOAL_MAX", 1.0))
GOAL_MIN = float(os.environ.get("STANCE_GOAL_MIN", 0.25))
ACTION_SCALE = 0.4
# Fraction of episodes that start from a recorded walking state instead of a
# perturbed stand: the moment the walking policy hands the legs over. The
# shipped policy only ever saw quiet standing starts, and a robot still
# moving at 0.1-0.3 m/s is outside everything it learned; handed over at
# 0.1 s intervals through a slow-down, 10-60% of handovers fell
# (eval/score_handover.py). States come from training/record_walk_states.py.
WALK_START_PROB = float(os.environ.get("STANCE_WALK_START_PROB", 0.0))
# A catch policy (training/train_stance.py with these set) bridges only the
# first seconds after the handover, so it trains under the runtime's exact
# conditions then: arms held still where they are at three times the
# walking gain (Runtime.stand_still), and the runtime's fall check.
ARMS_STILL = bool(int(os.environ.get("STANCE_ARMS_STILL", 0)))
ARM_GAIN = float(os.environ.get("STANCE_ARM_GAIN", 1.0))
UPRIGHT_MIN = float(os.environ.get("STANCE_UPRIGHT_MIN", 0.6))
Z_MIN = float(os.environ.get("STANCE_Z_MIN", 0.5))
# Fine-tuning for demo 13 (walk across, stop, fetch): one arm at a time
# moves through the poses the runtime actually commands there, from
# training/arm_goal_bank.npz plus the ready route over the bench, at the
# runtime's arm stiffness and with its arm gravity feedforward
# (Runtime._gravity_comp). The box sampled above never reaches these:
# the ready route puts the wrist pitch at 1.6 rad.
GOALS = os.environ.get("STANCE_GOALS")
GRAV_COMP = bool(int(os.environ.get("STANCE_GRAV", 0)))
ARM_RATE = float(os.environ.get("STANCE_ARM_RATE", 0.02))     # rad per tick
READY_VIA = np.array([0.62, -0.97, -1.29, -0.95, 0.58, 1.61, 0.8])
READY = np.array([-0.39, -1.04, -0.27, -0.61, 0.55, 1.61, 0.31])
# demo 13's route over the bench edge: elbow folded, then back and out
READY_VIA2 = [np.array([0.5, -0.5, -0.02, 1.9, 0.03, 0.07, 0.1]),
              np.array([0.5, -1.2, -0.6, 0.6, 0.4, 1.2, 0.5])]
MIRROR = np.array([1, -1, -1, 1, -1, 1, -1])      # right arm pose -> left
# Keep both feet planted. Without it the policy balances by stepping in
# place (17 to 21 foot lifts a second) and creeps 1 to 9 cm in 4 s, which
# looks like it never stopped walking. Penalises lift-offs, a foot in the
# air, and feet moved from where they stood.
FEET_W = float(os.environ.get("STANCE_FEET_W", 0.0))
# Stand up straight: penalise the torso leaning off vertical and the waist
# bending. Without it the policy counterbalances an arm load by slumping
# 20 to 30 degrees over the table at the waist.
TORSO_W = float(os.environ.get("STANCE_TORSO_W", 0.0))
# Replay arm commands recorded from demo 13 (training/record_arm_traj.py)
# in this share of episodes: the real disturbance, not random poses.
ARM_TRAJ = os.environ.get("STANCE_ARM_TRAJ")
# Stand like a person: legs near a straighter pose than the walking policy's
# default (knees 0.67 rad, a deep crouch; the demo sagged further, to a 0.71
# m pelvis), and legs that keep still. POSE_W pulls toward STAND_POSE (flat
# feet: hip + knee + ankle pitch = 0), LEGVEL_W penalises leg joint speed
# (the 5 rad/s "dancing"), RATE_W scales the action-rate penalty.
POSE_W = float(os.environ.get("STANCE_POSE_W", 0.0))
LEGVEL_W = float(os.environ.get("STANCE_LEGVEL_W", 0.0))
RATE_W = float(os.environ.get("STANCE_RATE_W", 0.15))
HEIGHT = float(os.environ.get("STANCE_HEIGHT", 0.73))
# Low-pass the joint targets (first order, this weight on the new target
# per 50 Hz tick) and ramp between ticks instead of stepping. Measured:
# 78% of leg motion was above 8 Hz and the targets jumped at 30 to 125 Hz,
# which is the "dancing". 1.0 = off (the policy's raw staircase).
ACT_FILTER = float(os.environ.get("STANCE_ACT_FILTER", 1.0))
_LEG = [-0.14, 0.0, 0.0, 0.30, -0.16, 0.0]
STAND_POSE = np.array(_LEG + _LEG)
ARM_TRAJ_PROB = float(os.environ.get("STANCE_ARM_TRAJ_PROB", 0.6))
WALK_STATES = os.environ.get("STANCE_WALK_STATES",
                             str(REPO / "training" / "walk_states.npz"))


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
        self.torso = m.body("torso_link").id
        self.waist_i = np.array([CONTROLLED.index(n) for n in WAIST])
        self.imu = m.site("imu_in_pelvis").id
        self.gyro = m.sensor("gyro_pelvis").id
        self.linvel = m.sensor("local_linvel_pelvis").id
        self.feet = [m.body(f"{s}_ankle_roll_link").id for s in ("left", "right")]

        for a in self.arm_act:
            kp = m.actuator_gainprm[a, 0] * ARM_GAIN
            m.actuator_gainprm[a, 0] = kp
            m.actuator_biasprm[a, 1] = -kp

        self.walk_states = None
        if WALK_START_PROB > 0:
            bank = np.load(WALK_STATES)
            names = [str(n) for n in bank["joints"]]
            mine = [m.joint(j).name for j in range(m.njnt)]
            assert names == mine, "walk-state bank recorded on another body"
            self.walk_states = (bank["qpos"], bank["qvel"])

        self.bank = None
        if GOALS:
            right = np.load(REPO / GOALS)["right"]
            self.bank = {"right": right, "left": right * MIRROR}
            route = [READY_VIA, READY] + READY_VIA2
            self.route = {"right": route,
                          "left": [r * MIRROR for r in route]}
            assert ARMS[0].startswith("left") and ARMS[7].startswith("right")
        self.arm_dofs = self.arm_qvel
        self.traj_bank, self.traj = None, None
        if ARM_TRAJ:
            z = np.load(REPO / ARM_TRAJ)
            self.traj_bank = [z["cmd"][s:s + n]
                              for s, n in zip(z["start"], z["length"])]

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
            np.clip(base_err * 5.0, -5.0, 5.0),
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
        if self.bank is not None and self.rng.random() < 0.8:
            # one arm works, the other stays where it is (a grasp, a place)
            goal = self.arm_cmd.copy()
            side = "right" if self.rng.random() < 0.75 else "left"
            sl = slice(7, 14) if side == "right" else slice(0, 7)
            r = self.rng.random()
            if r < 0.25:
                pick = self.route[side][self.rng.integers(len(self.route[side]))]
            elif r < 0.35:
                pick = self.arm_default[sl]          # arm back down
            else:
                pick = self.bank[side][self.rng.integers(len(self.bank[side]))]
            goal[sl] = pick
            return np.clip(goal, self.arm_limits[:, 0], self.arm_limits[:, 1])
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
        walk_start = (self.walk_states is not None
                      and self.rng.random() < WALK_START_PROB)
        if walk_start:
            # hold wherever the handover happens, facing wherever it faces
            k = self.rng.integers(len(self.walk_states[0]))
            d.qpos[:] = self.walk_states[0][k]
            d.qvel[:] = self.walk_states[1][k]
        self.traj = None
        if self.traj_bank is not None and self.rng.random() < ARM_TRAJ_PROB:
            run = self.traj_bank[self.rng.integers(len(self.traj_bank))]
            k = self.rng.integers(max(1, len(run) - self.max_steps // 2))
            self.traj = run[k:]
            d.qpos[self.arm_qpos] = self.traj[0]   # arm already where it is
        mujoco.mj_forward(m, d)

        # The goal is COMMANDED, not simply wherever the robot woke up.
        # This is the whole change: the policy that holds a mark and the
        # policy that travels to one become the same policy, which removes
        # the handover that no amount of tuning could survive. Measured: the
        # walking policy falls whenever it is engaged from a stance-held
        # state, because the stance policy idles at 9-16 rad/s of joint
        # chatter and never presents the quiet gait pose it needs, and
        # freezing the legs to fix the pose topples the robot outright.
        here = d.xpos[self.pelvis][:2].copy()
        # Half the episodes hold station. At 35% the policy learned to
        # travel and forgot how to stand still through an arm reach: it
        # travelled 0.9 m to within 0.13 m but then fell during a grasp,
        # which is the capability the project already had and must not lose.
        if walk_start or self.rng.random() >= TRAVEL_PROB:
            self.home_xy = here            # hold station, as before
        else:
            ang = self.rng.uniform(-np.pi, np.pi)
            dist = self.rng.uniform(GOAL_MIN, GOAL_MAX)
            self.home_xy = here + dist * np.array([np.cos(ang), np.sin(ang)])
        self.goal_dist0 = float(np.linalg.norm(self.home_xy - here))
        R = d.xmat[self.pelvis].reshape(3, 3)
        self.home_yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        self.last_action = np.zeros(self.act_dim, dtype=np.float32)
        self.filt = d.qpos[self.qpos_adr].copy()
        self.arm_goal = self.arm_default.copy()
        self.arm_cmd = (d.qpos[self.arm_qpos].copy() if walk_start
                        else self.arm_default.copy())
        if ARMS_STILL:
            self.arm_goal = self.arm_cmd.copy()
        if self.traj is not None:
            self.arm_cmd = self.traj[0].astype(float).copy()
        self.arm_timer = 0
        self.t = 0
        self.foot_prev = np.array([d.xpos[f][:2] for f in self.feet])
        self.foot_z0 = min(d.xpos[f][2] for f in self.feet)
        self.foot_travel = 0.0
        self.foot_home = self.foot_prev.copy()
        self.feet_were_down = [True, True]
        self.lift_offs = 0
        self.settle_t = 0 if self.goal_dist0 == 0 else None
        return self._obs()

    def step(self, action):
        d, m = self.data, self.model
        action = np.clip(action, -1.0, 1.0)
        target = self.default + ACTION_SCALE * action

        # the arms move on their own schedule — this is the disturbance
        if self.arm_timer <= 0 and not ARMS_STILL:
            self.arm_goal = self._new_arm_goal()
            self.arm_timer = int(self.rng.uniform(0.8, 2.0) / CTRL_DT)
        self.arm_timer -= 1
        if self.traj is not None:
            self.arm_cmd = self.traj[min(self.t, len(self.traj) - 1)].astype(float)
        else:
            self.arm_cmd += np.clip(self.arm_goal - self.arm_cmd,
                                    -ARM_RATE, ARM_RATE)

        prev = self.filt.copy()
        self.filt = ACT_FILTER * target + (1.0 - ACT_FILTER) * self.filt
        d.ctrl[self.arm_act] = self.arm_cmd
        for k in range(self.n_substeps):
            d.ctrl[self.act] = (prev + (self.filt - prev) * (k + 1) / self.n_substeps
                                if ACT_FILTER < 1.0 else target)
            if GRAV_COMP:
                d.qfrc_applied[self.arm_dofs] = d.qfrc_bias[self.arm_dofs]
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
            - RATE_W * rate                            # smooth commands
            - 0.05 * float(np.linalg.norm(action))
            # Stand tall. Left alone the policy discovers that crouching is
            # stable, settles at a 0.64 m pelvis, and quietly costs the arm
            # ~8 cm of forward reach — which is the difference between the
            # robot being able to pick something up and not.
            + 1.5 * np.exp(-((pos[2] - HEIGHT) / 0.06) ** 2)
            - 1.0 * float(pos[2] < 0.62)               # do not sag
            + 0.2 * float(feet_down > 0)
        )

        if POSE_W > 0:
            err = float(np.sum((d.qpos[self.qpos_adr[:12]] - STAND_POSE) ** 2))
            reward += POSE_W * (np.exp(-err / 0.1) - 1.0 * err)
        if LEGVEL_W > 0:
            reward -= LEGVEL_W * float(np.mean(d.qvel[self.qvel_adr[:12]] ** 2))
        if TORSO_W > 0:
            # both bodies: with the torso alone penalised the policy kept
            # the torso vertical and rolled the pelvis 20 degrees instead
            tilt = float(np.arccos(np.clip(d.xmat[self.torso][8], -1, 1))
                         + np.arccos(np.clip(d.xmat[self.pelvis][8], -1, 1)))
            bend = float(np.linalg.norm(
                d.qpos[self.qpos_adr[self.waist_i]] - self.default[self.waist_i]))
            reward += TORSO_W * (np.exp(-(tilt / 0.08) ** 2) - 2.0 * tilt
                                 - 1.0 * bend)

        foot_xy = np.array([d.xpos[f][:2] for f in self.feet])
        moved = float(np.sum(np.linalg.norm(foot_xy - self.foot_prev, axis=1)))
        self.foot_prev = foot_xy
        self.foot_travel += moved
        # What a step is: a foot off the floor, and a foot that ends up
        # somewhere else. Penalising foot speed instead (the first version)
        # mostly measured ankle jitter and changed nothing. The first 0.5 s
        # are free: a handover can arrive mid-step.
        down = [any(m.geom_bodyid[g] == f
                    for c in d.contact[:d.ncon] for g in (c.geom1, c.geom2))
                for f in self.feet]
        # Planted only once it is where it was sent: an episode with a goal
        # to step to may step until it arrives (step_to needs exactly that),
        # then must stand. Holding episodes are "arrived" from the start.
        if self.settle_t is None and drift < 0.04:
            self.settle_t = self.t
        grace = int(0.5 / CTRL_DT)
        if self.settle_t is not None and self.t == self.settle_t + grace:
            self.foot_home = foot_xy.copy()
        if self.settle_t is not None and self.t >= self.settle_t + grace:
            lift_offs = sum(p and not x
                            for p, x in zip(self.feet_were_down, down))
            self.lift_offs += lift_offs
            if FEET_W > 0:
                away = float(np.sum(np.linalg.norm(
                    foot_xy - self.foot_home, axis=1)))
                reward -= FEET_W * (0.5 * sum(not x for x in down)
                                    + 2.0 * lift_offs + 10.0 * away)
        self.feet_were_down = down

        self.last_action = action.astype(np.float32)
        self.t += 1
        # Drift is distance from the GOAL, so a 1 m command starts at 1 m of
        # drift. Terminate on going *backwards* from where it was sent, not
        # on the goal being far away.
        fell = bool(pos[2] < Z_MIN or upright < UPRIGHT_MIN
                    or drift > self.goal_dist0 + 0.6
                    or yaw_err > 0.9)
        if fell:
            reward -= 8.0
        done = fell or self.t >= self.max_steps
        info = {"drift": drift, "fell": fell, "upright": upright,
                "yaw": yaw_err, "foot_travel": self.foot_travel,
                "lift_offs": self.lift_offs}
        return self._obs(), float(reward), done, info
