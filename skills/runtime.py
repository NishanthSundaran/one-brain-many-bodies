"""Shared execution context for skills: owns the world, per-arm differential
IK, the servo loop, weld-grasp toggling, and an optional frame-recording hook.
Skills are deterministic, blocking routines built on these primitives — the
mid-level API between the VLM brain and the physics.
"""

from __future__ import annotations

import pathlib

import numpy as np
from scipy.spatial.transform import Rotation

import mujoco

REPO = pathlib.Path(__file__).resolve().parents[1]

ACCEL_MAX = 0.8      # m/s^2 command-velocity ramp: no robotic jump-starts
ROT_GAIN = 2.5       # 1/s toward the canonical palm orientation
ROT_MAX = 1.2        # rad/s
STATION_SPEED = 0.3  # m/s cap while holding a station
LATERAL_SPEED = 0.3  # m/s cap on sideways commands (topples above this)

from bodies.body import EE_SITE
from controllers.locomotion import POLICY_JOINTS
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
                    [self.model.actuator(n).id for n in hand["actuators"]])
                self.arm[side]["hand_open"] = np.asarray(hand["open"], dtype=float)
                self.arm[side]["hand_closed"] = np.asarray(hand["closed"], dtype=float)
        self.on_frame = None  # callable(t) for demos; None = headless
        self._t = 0.0
        self._held: dict[str, str | None] = {s: None for s in self.arm}

        # Locomotion. A mobile body walks under a learned policy running in
        # closed loop on the real physics state; there is no scripted base
        # motion anywhere. While the arms work, the policy keeps the legs
        # and waist (it must go on balancing) and the arm controller owns
        # the arms.
        self.root_id = self.model.body(self.body.cfg["root_body"]).id
        self.loco = None
        self.walk_speed = 0.5
        self.station = None   # (xy, yaw) the standing controller holds
        self._station_i = np.zeros(2)   # integral state of the station PI
        self._station_speed = STATION_SPEED
        self._planted = False           # stance mode (see plant/unplant)
        self._nominal_gain = self.model.actuator_gainprm[:, 0].copy()
        if self.body.mobile:
            from controllers.locomotion import LocomotionPolicy
            cfg = self.body.cfg["locomotion"]
            self.loco = LocomotionPolicy(
                self.model, REPO / cfg["policy"], cfg["stand_angles"],
                ctrl_dt=cfg.get("ctrl_dt", 0.02),
                action_scale=cfg.get("action_scale", 0.5),
                gait_freq=cfg.get("gait_freq", 1.5))
            self.loco.controlled = (
                None if cfg.get("policy_owns") == "all" else "leg_waist")
            self.walk_speed = float(cfg.get("walk_speed", 0.5))
            # hold the start pose: facing +x at the primary table's station
            self.station = (np.array(self.model.body_pos[self.root_id][:2]), 0.0)
            self.step(3.0)   # PI station keeping converges in ~3 s
        self._fixed_base_xy = np.array(self.model.body_pos[self.root_id][:2])

    # ---------------- core stepping ----------------

    def step(self, seconds: float) -> None:
        for _ in range(int(round(seconds / self.dt))):
            if self.loco is not None and not self._planted:
                self._hold_station()
                self.loco.step(self.data)
            mujoco.mj_step(self.model, self.data)
            self._t += self.dt
            if self.on_frame is not None:
                self.on_frame(self._t)

    def ee_pos(self, side: str) -> np.ndarray:
        return self.data.site_xpos[self.ik[side].site_id].copy()

    # ---------------- base frame ----------------

    @property
    def base_xy(self) -> np.ndarray:
        """Live pelvis position: for a walking robot this is physics state,
        not a number we set."""
        if self.body.mobile:
            return self.data.xpos[self.root_id][:2].copy()
        return self._fixed_base_xy

    @property
    def base_yaw(self) -> float:
        if not self.body.mobile:
            return 0.0
        R = self.data.xmat[self.root_id].reshape(3, 3)
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def to_base(self, world_pos) -> np.ndarray:
        """World point -> robot base frame (+x forward, +y left). Arm choice
        and reach gates are body-relative, so they must survive walking."""
        p = np.asarray(world_pos, dtype=float)
        d = p[:2] - self.base_xy
        c, s = np.cos(-self.base_yaw), np.sin(-self.base_yaw)
        return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1],
                         p[2] if len(p) > 2 else 0.0])

    def to_world(self, base_pos) -> np.ndarray:
        p = np.asarray(base_pos, dtype=float)
        c, s = np.cos(self.base_yaw), np.sin(self.base_yaw)
        return np.array([self.base_xy[0] + c * p[0] - s * p[1],
                         self.base_xy[1] + s * p[0] + c * p[1],
                         p[2] if len(p) > 2 else 0.0])

    def nearest_arm(self, pos) -> str:
        return "left" if self.to_base(pos)[1] > 0 else "right"

    def arm_can_reach(self, side: str, world_pos) -> bool:
        """Cheap workspace gate from config: refuse cross-body targets
        before moving instead of flailing through the scene. Takes a world
        point and judges it in the base frame, so it stays correct after
        the robot has walked and turned."""
        cfg = self.body.cfg["arms"][side]
        y = self.to_base(world_pos)[1]
        if side == "left" and "reach_min_y" in cfg:
            return y >= cfg["reach_min_y"]
        if side == "right" and "reach_max_y" in cfg:
            return y <= cfg["reach_max_y"]
        return True

    # ---------------- locomotion ----------------

    @staticmethod
    def _wrap(a: float) -> float:
        return (a + np.pi) % (2 * np.pi) - np.pi

    def _hold_station(self) -> None:
        """Station keeping. The policy is a *locomotion* policy: commanded to
        zero velocity it still creeps (~0.2 m/s here), and a robot that creeps
        walks into the table it is working at. Standing still is therefore its
        own closed-loop problem. Proportional feedback alone leaves a standing
        offset against that constant bias, so this is a PI controller — the
        integral term learns the creep and cancels it."""
        if self.station is None:
            return
        xy, yaw = self.station
        err = np.asarray(xy) - self.base_xy
        c, s = np.cos(-self.base_yaw), np.sin(-self.base_yaw)
        e = np.array([c * err[0] - s * err[1], s * err[0] + c * err[1]])
        yaw_err = self._wrap(yaw - self.base_yaw)

        # The integral exists to cancel the policy's standing creep, so it
        # only accumulates near the target. Left running during a metre-long
        # approach it saturates and sails the robot straight past the goal.
        near = np.abs(e) < 0.25
        self._station_i = np.where(
            near, np.clip(self._station_i + e * self.dt * 0.8, -0.35, 0.35),
            self._station_i * 0.995)
        # Forward and sideways get different caps: this robot walks forward
        # well but a fast lateral command topples it, exactly as a person
        # cannot side-step at walking speed.
        cap = np.array([self._station_speed,
                        min(self._station_speed, LATERAL_SPEED)])
        v = np.clip(1.2 * e + self._station_i, -cap, cap)
        # Slow down while turning hard — but only when actually travelling.
        # Applied while holding station it throttles the very correction that
        # cancels the policy's creep, and the robot wanders metres away as it
        # turns.
        if np.linalg.norm(e) > 0.35:
            v *= float(np.clip(1.0 - 1.4 * abs(yaw_err), 0.25, 1.0))
        v[np.abs(e) < 0.02] *= 0.5                      # calm near the target
        self.loco.set_command(
            float(v[0]), float(v[1]),
            0.0 if abs(yaw_err) < 0.05
            else float(np.clip(2.0 * yaw_err, -0.7, 0.7)))

    def _both_feet_down(self) -> bool:
        feet = {self.model.body(f"{s}_ankle_roll_link").id
                for s in ("left", "right")}
        touching = set()
        for c in range(self.data.ncon):
            for g in (self.data.contact[c].geom1, self.data.contact[c].geom2):
                b = self.model.geom_bodyid[g]
                if b in feet:
                    touching.add(b)
        return len(touching) == 2

    def reposition_for(self, world_pos, forward: float = 0.30,
                       band=(0.22, 0.38)) -> bool:
        """Walking robots do not stop on a mark: the policy leaves the base a
        few centimetres from its station, which is the difference between a
        target being reachable and not. If the target sits outside the arm's
        comfortable band, step the base so it lands inside. Returns True if
        the robot repositioned."""
        if self.loco is None:
            return False
        local = self.to_base(world_pos)
        if band[0] <= local[0] <= band[1] and abs(local[1]) <= 0.28:
            return False
        yaw = self.base_yaw
        side = float(np.clip(local[1], -0.16, 0.16))
        offset = np.array([np.cos(yaw) * forward - np.sin(yaw) * side,
                           np.sin(yaw) * forward + np.cos(yaw) * side])
        self.unplant()
        self.walk_to(*(np.asarray(world_pos)[:2] - offset), face=world_pos[:2])
        self.plant()
        return True

    def plant(self, stiffness: float = 6.0,
              damping_ratio: float = 0.08) -> None:
        """Enter stance: stop stepping and stiffen the legs around the pose
        the policy left them in. The base stays free and fully physical —
        the robot balances statically on its feet, and would topple if the
        arms pulled the centre of mass outside the support polygon. This is
        how the walking controller hands over to the manipulation
        controller: the locomotion policy cannot rebalance while the arm
        controller owns the arms, so it is not left fighting for them."""
        if self.loco is None or self._planted:
            return
        names = self.loco.policy_joint_names()
        acts = [self.model.actuator(n).id for n in names]
        loco_cfg = self.body.cfg["locomotion"]
        stance = {n: float(v) for n, v in
                  zip(POLICY_JOINTS, loco_cfg.get("stance_angles",
                                                  loco_cfg["stand_angles"]))}
        target = np.array([stance[n] for n in names])

        # Wait for double support: freezing mid-step, with one foot in the
        # air, drops the robot on the spot.
        deadline = self._t + 3.0
        while self._t < deadline and not self._both_feet_down():
            self.step(self.dt)
        # read the pose *after* the wait — it moved while we waited
        measured = np.array([self.data.qpos[self.model.joint(n).qposadr[0]]
                             for n in names])

        # Hand over in one shot. Ramping first would fight the policy, which
        # keeps overwriting the same actuators at 50 Hz.
        self._planted = True
        self.station = None
        # Stance needs stiff joints to hold the pose against gravity, and
        # stiff position gains are only stable under implicit integration —
        # the explicit Euler step the policy runs under blows up. The policy
        # is stopped now, so the integrator is free to change.
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        for act in acts:
            kp = self._nominal_gain[act] * stiffness
            self.model.actuator_gainprm[act, 0] = kp
            self.model.actuator_biasprm[act, 1] = -kp
            self.model.actuator_biasprm[act, 2] = -damping_ratio * kp
        # Take over at the measured pose, then ease to the balanced stance.
        # Snapping straight to the stance shoves the base ~0.15 m forward as
        # the legs straighten; holding the measured mid-gait pose is not
        # statically stable. Ramping does both jobs.
        self.data.ctrl[acts] = measured
        n_ramp = max(1, int(1.0 / self.dt))
        for i in range(n_ramp):
            s = (i + 1) / n_ramp
            s = s * s * (3 - 2 * s)
            self.data.ctrl[acts] = (1 - s) * measured + s * target
            self.step(self.dt)
        self.step(0.4)

    def unplant(self) -> None:
        """Leave stance and give the legs back to the walking policy."""
        if self.loco is None or not self._planted:
            return
        for name in self.loco.policy_joint_names():
            act = self.model.actuator(name).id
            kp = self._nominal_gain[act]
            self.model.actuator_gainprm[act, 0] = kp
            self.model.actuator_biasprm[act, 1] = -kp
            self.model.actuator_biasprm[act, 2] = 0.0
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
        self._planted = False
        self._station_i[:] = 0.0
        self.station = (self.base_xy.copy(), self.base_yaw)
        self.step(0.5)

    def _goto(self, xy, yaw: float, tol: float, yaw_tol: float,
              timeout: float, speed: float) -> bool:
        """Drive the base to a pose with the station PI. The same controller
        walks and stands: its integral term is what cancels the policy's
        forward creep, which is exactly the error that ruins both."""
        self.station = (np.asarray(xy, dtype=float), yaw)
        self._station_i[:] = 0.0
        self._station_speed = speed
        t_end = self._t + timeout
        while self._t < t_end:
            if (np.linalg.norm(np.asarray(xy) - self.base_xy) < tol
                    and abs(self._wrap(yaw - self.base_yaw)) < yaw_tol):
                break
            self.step(self.dt)
        self._station_speed = STATION_SPEED
        return np.linalg.norm(np.asarray(xy) - self.base_xy) < tol * 2.0

    def walk_to(self, x: float, y: float, face=None, speed: float = None,
                tol: float = 0.12, timeout: float = 45.0) -> bool:
        """Walk to (x, y) under the locomotion policy.

        A unicycle pursuit controller: always walk forwards, steering with
        yaw rate, so the robot follows a curved path the way a person does.
        Turning on the spot was tried first and is a trap here — this policy
        creeps forward through every turn, so 'turn, then go' wanders metres
        off course, and fighting the creep with backwards commands topples it.
        """
        if self.loco is None:
            raise RuntimeError("this body has no locomotion policy")
        target = np.array([x, y], dtype=float)
        speed = self.walk_speed if speed is None else speed
        self.home()
        self.unplant()
        self.station = None          # pursuit owns the commands while walking

        t_end = self._t + timeout
        while self._t < t_end:
            err = target - self.base_xy
            dist = float(np.linalg.norm(err))
            if dist < tol:
                break
            bearing = self._wrap(float(np.arctan2(err[1], err[0]))
                                 - self.base_yaw)
            # Steer towards the target; walk slower the worse the heading is,
            # and ease off over the last stride so it does not sail past.
            # Yaw authority sets the turn radius (r = v / w). Too little and
            # the robot orbits a target it cannot turn tightly enough to
            # reach — which is exactly what a 0.6 rad/s cap did here.
            wz = float(np.clip(1.4 * bearing, -1.1, 1.1))
            # Never let forward speed reach zero: at a right-angle bearing
            # that becomes a pivot on the spot, which this policy cannot do
            # without creeping away. Keeping it walking turns the correction
            # into an arc — which is how people walk round corners anyway.
            vx = speed * float(np.clip(np.cos(bearing), 0.12, 1.0))
            vx *= float(np.clip(dist / 0.5, 0.35, 1.0))   # ease off at the end
            self.loco.set_command(vx, 0.0, wz)
            self.step(self.dt)

        # Close the last stride with the station controller, holding the
        # *target* rather than wherever the pursuit ran out: a walking robot
        # carries momentum and creep through the final step.
        final_yaw = self.base_yaw
        if face is not None:
            fd = np.asarray(face, dtype=float)[:2] - target
            final_yaw = float(np.arctan2(fd[1], fd[0]))
        # Hold where it stopped, not the target: pulling a walking robot
        # backwards over the last half metre topples it. The cost is honest
        # stopping accuracy of a few decimetres, which is what a velocity-
        # commanded policy without a footstep planner gives you.
        self._goto(self.base_xy.copy(), final_yaw, 0.30, 0.12, 8.0, 0.3)
        self.station = (self.base_xy.copy(), final_yaw)
        self._station_i[:] = 0.0
        self.step(1.0)
        return bool(np.linalg.norm(target - self.base_xy) < tol * 3.0)

    # ---------------- arm servoing ----------------

    def _orientation_twist(self, side: str) -> np.ndarray:
        """Angular velocity steering the palm back to the arm's canonical
        manipulation orientation. `ee_quat` is given in the BASE frame and
        rotated by the current heading, so it stays correct after walking.
        Bodies without the key (e.g. 4-DoF H1 arms) get no orientation task."""
        quat = self.body.cfg["arms"][side].get("ee_quat")
        if quat is None:
            return np.zeros(3)
        R_cur = self.data.site_xmat[self.ik[side].site_id].reshape(3, 3)
        R_des = (Rotation.from_euler("z", self.base_yaw).as_matrix()
                 @ Rotation.from_quat(np.roll(quat, -1)).as_matrix())
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

    def set_down_spot(self, world) -> list:
        """A neutral spot on the table in front of the robot, reachable by
        either arm — where a stuck hand puts its object down."""
        front = self.to_world([0.42, 0.0])
        table = world.table_of([front[0], front[1], world.table_top]) \
            or world.tables["table"]
        return [front[0], front[1], table["top"] + 0.07]

    def home(self) -> None:
        """Empty arms to the resting pose; an arm holding something keeps it
        raised in the ready pose instead of swinging it down to the floor."""
        for side in self.arm:
            if self.held(side):
                self.move_arm_ready(side)
            else:
                self.move_arm_home(side)

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
        tuned for an unrotated torso. On a balancing robot the waist belongs
        to the locomotion policy — writing it here fights the balance loop
        at 50 Hz and walks the robot away from its station."""
        if self.loco is not None:
            return
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
