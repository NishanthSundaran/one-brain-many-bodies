"""Shared execution context for skills: owns the world, per-arm differential
IK, the servo loop, weld-grasp toggling, and an optional frame-recording hook.
Skills are deterministic, blocking routines built on these primitives — the
mid-level API between the VLM brain and the physics.
"""

from __future__ import annotations

import pathlib

from contextlib import contextmanager

import numpy as np
from scipy.spatial.transform import Rotation

import mujoco

REPO = pathlib.Path(__file__).resolve().parents[1]

ACCEL_MAX = 0.8      # m/s^2 command-velocity ramp: no robotic jump-starts
ROT_GAIN = 2.5       # 1/s toward the canonical palm orientation
ROT_MAX = 1.2        # rad/s
STATION_SPEED = 0.3  # m/s cap while holding a station
LATERAL_SPEED = 0.3  # m/s cap on sideways commands (topples above this)
GAZE_MAX = 0.20      # rad: how far the torso turns to look at a target

# Trajectory tracking, tuned against the human-motion metrics in
# eval/human_motion.py (see the sweep in the README).
LEAD = 0.0          # seconds of anticipation ahead of the plan
TRACK_GAIN = 6.0    # feedback on the aim point
SPEED_CAP = 1.0     # m/s; a human hand peaks near 1 m/s on a long reach
CORRECT_TOL = 0.010  # m: closer than this and the reach needs no correction
CORRECT_TIME = 0.35  # s: longest corrective submovement
CORRECT_MIN = 0.15   # s: shortest; the correction is scaled to the error so
                     # it stays a coda to the reach rather than a third of it
WINDUP = 0.15        # rad: anti-windup bound on how far ctrl may lead the joint
# Below this distance to the target the palm-orientation task is faded out,
# handing the arm's whole velocity budget to position. See reach_line.
ORIENT_FADE = 0.06


from bodies.body import EE_SITE
from controllers.locomotion import POLICY_JOINTS
from controllers.diff_ik import DiffIK
from controllers.navigator import GaitNavigator, arrived
from controllers.admittance import joint_limit_scale


class Runtime:
    def __init__(self, world, start: str = "stance"):
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
            # The joint limits say what a joint can do on its own, not what
            # the body lets it do: the G1's upper arm hits the torso a few
            # degrees past straight down, long before its roll limit. An IK
            # solve that only knows the joint limits picks those jammed poses
            # and the hand lands 10 to 30 cm off. `ik_limits` narrows a joint
            # for the solve.
            ik_lim = self.arm[side]["limits"].copy()
            for j, lim in (self.body.cfg["arms"][side].get("ik_limits")
                           or {}).items():
                ik_lim[joints.index(j)] = lim
            self.arm[side]["ik_limits"] = ik_lim
            hand = self.body.cfg["arms"][side].get("hand")
            if hand is not None:
                self.arm[side]["hand_act"] = np.array(
                    [self.model.actuator(n).id for n in hand["actuators"]])
                self.arm[side]["hand_open"] = np.asarray(hand["open"], dtype=float)
                self.arm[side]["hand_closed"] = np.asarray(hand["closed"], dtype=float)
        self._arm_dofs = np.array(
            [self.model.joint(j).dofadr[0]
             for joints in self.body.arm_joints.values() for j in joints],
            dtype=int)
        # Orientation regulation is gentle on a balancing robot: at full
        # strength the task saturates its cap every step, eats the whole IK
        # solution, and the hand rotates instead of reaching.
        self.rot_gain = ROT_GAIN
        self.rot_cap = ROT_MAX
        self.qdot_cap = None
        self.on_frame = None  # callable(t) for demos; None = headless
        self._t = 0.0
        self._held: dict[str, str | None] = {s: None for s in self.arm}
        # Set before anything steps: a mobile body starts stepping inside
        # this constructor, and step() consults these.
        self._trace = None
        self._trace_side = "right"

        # Locomotion. A mobile body walks under a learned policy running in
        # closed loop on the real physics state; there is no scripted base
        # motion anywhere. While the arms work, the policy keeps the legs
        # and waist (it must go on balancing) and the arm controller owns
        # the arms.
        self.root_id = self.model.body(self.body.cfg["root_body"]).id
        self.loco = None
        self.walk_speed = 0.5
        self.use_plant = False
        self.arm_stance_gain = 1.0
        self.stance = None
        self.stepper = None
        # how far to the side a target may sit before approach_object steps
        # toward it; a caller that has already lined the body up for a
        # one-handed reach widens it so the skill does not undo that
        self.approach_lateral = 0.30
        # Arm motion on a balancing body. Defaults keep the tuned behaviour
        # (smoothstep, via poses as separate moves); a caller whose stance
        # policy trained against smoother arm motion can opt in to the
        # human profile and to passing through via poses without stopping.
        self.arm_profile = None        # None | "min_jerk"
        self.blend_vias = False
        self.catch = None       # bridges the walk-to-stance handover
        self.unified = None
        self._standing = False
        self.station = None   # (xy, yaw) the standing controller holds
        self._station_i = np.zeros(2)   # integral state of the station PI
        self._station_bias = np.zeros(2)  # measured hold offset (base frame)
        self._station_speed = STATION_SPEED
        self._planted = False           # stance mode (see plant/unplant)
        self.hold_with_walk = False
        self.hold_station = False
        self.arm_gravity_comp = False
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
            self.use_plant = bool(cfg.get("plant_to_manipulate", False))
            self.arm_stance_gain = float(cfg.get("arm_stance_gain", 3.0))
            self.rot_gain = float(cfg.get("rot_gain", 0.8))
            self.rot_cap = float(cfg.get("rot_cap", 0.25))
            self.qdot_cap = float(cfg.get("qdot_cap", 0.35))
            self.hold_with_walk = cfg.get("hold_with") == "walk_policy"
            self.hold_station = bool(cfg.get("hold_station", False))
            self.arm_gravity_comp = bool(cfg.get("arm_gravity_comp", False))
            # One policy for both jobs when it is available: walking and
            # standing become the same controller with different commands,
            # so there is no handover to fall over in.
            uni = REPO / cfg.get("unified_policy", "")
            if cfg.get("unified_policy") and uni.exists():
                from controllers.unified import UnifiedPolicy
                self.unified = UnifiedPolicy(self.model, uni,
                                             cfg["stand_angles"])
                self.loco = self.unified
                self.stance = self.unified
            else:
                ck = REPO / cfg.get("stance_policy", "")
                if cfg.get("stance_policy") and ck.exists():
                    from controllers.stance import StancePolicy
                    self.stance = StancePolicy(self.model, ck,
                                               cfg["stand_angles"])
                    # A stance policy trained to keep its feet planted
                    # cannot take the short steps step_to needs; this one
                    # (one that balances by stepping) does those, and hands
                    # back once the robot has arrived.
                    stepper = REPO / cfg.get("step_policy", "")
                    if cfg.get("step_policy") and stepper.exists():
                        self.stepper = StancePolicy(self.model, stepper,
                                                    cfg["stand_angles"])
                    catch = REPO / cfg.get("catch_policy", "")
                    if cfg.get("catch_policy") and catch.exists():
                        self.catch = StancePolicy(self.model, catch,
                                                  cfg["stand_angles"])
            # hold the start pose: facing +x at the primary table's station
            self.home_station = np.array(self.model.body_pos[self.root_id][:2])
            self.station = (self.home_station.copy(), 0.0)
            if start == "walk":
                # Begin under the walking policy, holding the spot it woke
                # on. The stance-to-walk handover has never worked (the
                # stance policy idles with too much joint chatter for the
                # walking policy to take over), so a fetch from across the
                # room must start walking and hand over to stance only once,
                # at the bench.
                self.station = (self.base_xy.copy(), self.base_yaw)
                self.step(1.5)
                self._fixed_base_xy = self.base_xy.copy()
                return
            if self.stance is not None:
                # Stand on the stance policy from the first tick: every second
                # spent settling under the travelling policy is drift.
                self.stand_still()
                self.step(1.5)
                self._station_bias = np.zeros(2)
                self._fixed_base_xy = self.base_xy.copy()
                return
            self.step(3.0)
            # The policy cannot walk backwards against its own creep, so the
            # hold settles a repeatable distance *ahead* of its setpoint.
            # Measure that bias once and aim behind the target by it.
            d_world = self.base_xy - self.home_station
            cy, sy = np.cos(-self.base_yaw), np.sin(-self.base_yaw)
            self._station_bias = np.array([cy * d_world[0] - sy * d_world[1],
                                           sy * d_world[0] + cy * d_world[1]])
            self.station = (self.home_station.copy(), 0.0)
            self.step(2.0)
            # Standing is the stance policy's job, not the travelling
            # policy's: it holds the mark to millimetres, where the walking
            # policy settles a variable 0.2-0.45 m ahead of its setpoint.
            if self.stance is not None:
                self.stand_still()
        self._fixed_base_xy = np.array(self.model.body_pos[self.root_id][:2])

    # ---------------- core stepping ----------------

    def _gravity_comp(self) -> None:
        """Feed the arms' gravity torque forward while the runtime moves them.

        The walking body's arms run on the policy's gains, low enough at the
        wrist (6 Nm/rad) that the hand's own weight bends it 0.15 to 0.3 rad
        off its command, and the hand lands a hand's length from where the IK
        put it. A real arm controller cancels gravity with a feedforward
        torque instead of stiffening the servo; this does the same. Only the
        arm joints, and only while standing: when walking, the walking policy
        owns the arms and was trained without it.
        """
        dofs = self._arm_dofs
        on = self._standing and self.arm_gravity_comp
        self.data.qfrc_applied[dofs] = self.data.qfrc_bias[dofs] if on else 0.0

    def step(self, seconds: float) -> None:
        for _ in range(int(round(seconds / self.dt))):
            if self.arm_gravity_comp:
                self._gravity_comp()
            if self.unified is not None:
                # Same controller either way: hold a station or track the
                # pursuit command that walk_to is issuing.
                if self._standing:
                    self._hold_station()
                self.unified.step(self.data)
            elif (self._standing and self.stance is not None
                  and not self.hold_with_walk):
                self.stance.step(self.data)
            elif self.loco is not None and not self._planted:
                self._hold_station()
                self.loco.step(self.data)
            mujoco.mj_step(self.model, self.data)
            self._t += self.dt
            if self._trace is not None:
                self._trace.append(self.ee_pos(self._trace_side))
            if self.on_frame is not None:
                self.on_frame(self._t)

    def ee_pos(self, side: str) -> np.ndarray:
        return self.data.site_xpos[self.ik[side].site_id].copy()

    @contextmanager
    def trace(self, side: str):
        """Record the hand path, one sample per control step, for scoring.

        Sampling here rather than at render rate matters: the human-motion
        metrics are speed-profile shapes, and a 30 Hz GIF-frame sample of a
        500 Hz loop smears the peak that is being measured.
        """
        prev, prev_side = self._trace, self._trace_side
        self._trace, self._trace_side = [], side
        try:
            yield self._trace
        finally:
            self._trace, self._trace_side = prev, prev_side

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
        # aim behind the setpoint by the measured hold bias
        c, s_ = np.cos(yaw), np.sin(yaw)
        b = self._station_bias
        xy = np.asarray(xy) - np.array([c * b[0] - s_ * b[1],
                                        s_ * b[0] + c * b[1]])
        err = xy - self.base_xy
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

    def step_to(self, xy, yaw: float | None = None, tol: float = 0.05,
                stride: float = 0.18, max_steps: int = 8) -> bool:
        """Fine positioning: shuffle to a spot using the stance policy.

        The walking policy travels but cannot stop accurately (it lands
        within a few tenths of a metre). The stance policy was trained to
        return to its hold point when pushed off it, which makes it a
        precise short-range positioner: move its hold point a stride at a
        time and it walks the base there and holds, to within a couple of
        centimetres. Coarse travel and fine placement, each by the controller
        that is good at it.
        """
        if self.stance is None:
            return False
        self.stand_still()
        if self.stepper is not None and self.unified is None:
            quiet, self.stance = self.stance, self.stepper
            self.stance.last_action[:] = 0.0
            try:
                return self._step_to(xy, yaw, tol, stride, max_steps)
            finally:
                self.stance = quiet
                quiet.hold(self.base_xy, self.base_yaw)
                quiet.last_action[:] = 0.0
                self.step(0.5)
        return self._step_to(xy, yaw, tol, stride, max_steps)

    def _step_to(self, xy, yaw, tol, stride, max_steps) -> bool:
        target = np.asarray(xy, dtype=float)[:2]
        yaw = self.base_yaw if yaw is None else yaw
        for _ in range(max_steps):
            err = target - self.base_xy
            dist = float(np.linalg.norm(err))
            # Keep the heading fixed while translating. Turning is what
            # costs this policy its balance; it can side-step and back up
            # perfectly well without it.
            yaw_err = self._wrap(yaw - self.base_yaw)
            if dist < tol and abs(yaw_err) < 0.08:
                return True
            step = err if dist <= stride else err / dist * stride
            hold_xy = self.base_xy + step
            hold_yaw = self.base_yaw + float(np.clip(yaw_err, -0.3, 0.3))
            if self.unified is not None:
                # One controller: aim the station at the next stride and let
                # the same policy walk the base there.
                self.station = (hold_xy, hold_yaw)
                self._station_i[:] = 0.0
            else:
                self.stance.hold(hold_xy, hold_yaw)
            self.step(2.5)
        return bool(np.linalg.norm(target - self.base_xy) < tol * 2.5)

    def approach_object(self, world_pos, forward: float = 0.32,
                        band=(0.18, 0.46), tries: int = 2) -> bool:
        """Get the target into the arm's envelope before reaching for it.

        Walking stops within a few decimetres of where it was aimed, which is
        the difference between reachable and not. So after arriving: stand,
        look at where the target actually is, and if it is outside the band,
        walk a short correction and stand again.
        """
        if self.loco is None:
            return True
        for _ in range(tries):
            self.stand_still()
            local = self.to_base(world_pos)
            if (band[0] <= local[0] <= band[1]
                    and abs(local[1]) <= self.approach_lateral):
                return True
            yaw = self.base_yaw
            side = float(np.clip(local[1], -0.20, 0.20))
            offset = np.array([np.cos(yaw) * forward - np.sin(yaw) * side,
                               np.sin(yaw) * forward + np.cos(yaw) * side])
            self.step_to(np.asarray(world_pos)[:2] - offset, yaw)
        self.stand_still()
        local = self.to_base(world_pos)
        return bool(band[0] <= local[0] <= band[1])

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

    def _set_arm_gain(self, scale: float) -> None:
        """The locomotion policy is trained on soft actuators (kp 75) and its
        gains are applied to every joint it owns, arms included. That is far
        too soft for an arm to hold a hand where the IK puts it — it simply
        droops backwards under gravity — so the arms stiffen while the robot
        manipulates and relax again before it walks."""
        for side in self.arm:
            for act in self.arm[side]["act"]:
                kp = self._nominal_gain[act] * scale
                self.model.actuator_gainprm[act, 0] = kp
                self.model.actuator_biasprm[act, 1] = -kp

    def stand_still(self) -> None:
        """Hold position while the arms work. With the unified policy this is
        just a zero velocity command plus the station controller; with the
        two-policy setup it hands the legs to the stance policy."""
        if self._standing:
            return
        if self.hold_with_walk:
            # The walking policy, fine-tuned to stop and to hold through an
            # arm reach (training/train_walkstop.py), keeps the legs; the
            # arms go to the arm controller where they are. No handover.
            self.loco.set_command(0.0, 0.0, 0.0)
            for a in self.arm.values():
                self.data.ctrl[a["act"]] = [
                    self.data.qpos[self.model.jnt_qposadr[
                        self.model.actuator_trnid[i, 0]]] for i in a["act"]]
            self.loco.controlled = "leg_waist"
            self._set_arm_gain(self.arm_stance_gain)
            self._standing = True
            # station keeping on top: the policy holds to a few decimetres
            # through a reach and turns, so the PI nudges it back on its mark
            self.station = ((self.base_xy.copy(), self.base_yaw)
                            if self.hold_station else None)
            self._station_i[:] = 0.0
            self.step(0.5)
            return
        if self.stance is None:
            return
        self._set_arm_gain(self.arm_stance_gain)
        if self.unified is not None:
            self._standing = True
            self.station = (self.base_xy.copy(), self.base_yaw)
            self._station_i[:] = 0.0
            self.step(0.5)
            return
        self.stance.hold(self.base_xy, self.base_yaw)
        self.stance.last_action[:] = 0.0
        self._standing = True
        self.station = None
        self.step(0.5)

    def walk_again(self) -> None:
        if not self._standing:
            return
        if self.hold_with_walk:
            self.loco.controlled = None
        self._set_arm_gain(1.0)
        self._standing = False
        self._station_i[:] = 0.0
        self.station = (self.base_xy.copy(), self.base_yaw)
        self.step(0.3)

    def plant(self, stiffness: float = 6.0,
              damping_ratio: float = 0.08) -> None:
        """Enter stance: stop stepping and stiffen the legs around the pose
        the policy left them in. The base stays free and fully physical —
        the robot balances statically on its feet, and would topple if the
        arms pulled the centre of mass outside the support polygon. This is
        how the walking controller hands over to the manipulation
        controller: the locomotion policy cannot rebalance while the arm
        controller owns the arms, so it is not left fighting for them."""
        # Planting (freezing the legs to work) was the intuitive design and
        # is the wrong one: a frozen robot has no way to absorb the momentum
        # of its own arm and topples. Left walking, the policy simply takes a
        # step and recovers. So this is opt-in and off by default.
        if self.loco is None or self._planted or not self.use_plant:
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
                tol: float = 0.30, timeout: float = 45.0,
                using: str = "stance") -> bool:
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
        if using == "gait":
            face_yaw = None
            if face is not None:
                fd = np.asarray(face, dtype=float)[:2] - target
                face_yaw = float(np.arctan2(fd[1], fd[0]))
            if not self.gait_travel(target, arrive=tol, timeout=timeout):
                return False
            self.gait_stop()
            if self.fallen():
                return False
            if face_yaw is not None or np.linalg.norm(
                    target - self.base_xy) > 0.05:
                self.step_to(target, face_yaw)
            return bool(np.linalg.norm(target - self.base_xy) < tol)
        if self.stance is not None:
            # Navigate with the stance policy, a stride at a time. The
            # travelling policy has the better gait (demos/07) but cannot be
            # steered to a spot: commanded to a target it orbits it and
            # eventually falls, landing a metre or two out. Stepping is
            # slower and lands within a few centimetres, and precision is
            # what deciding to reach for something needs.
            face_yaw = None
            if face is not None:
                fd = np.asarray(face, dtype=float)[:2] - target
                face_yaw = float(np.arctan2(fd[1], fd[0]))
            return self.step_to(target, face_yaw, tol=0.08,
                                max_steps=int(6 + 6 * np.linalg.norm(
                                    target - self.base_xy)))
        self.home()
        self.unplant()
        self.walk_again()   # the stance policy holds position: hand the legs
                            # back to the walking policy before asking it to go
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
            # Ease all the way to zero. A floor on the terminal speed means
            # the robot can never stop, so it orbits the target instead.
            vx *= float(np.clip(dist / 0.6, 0.0, 1.0))
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
        # Coarse travel is done; the stance policy places the base exactly.
        self.station = (self.base_xy.copy(), final_yaw)
        self.step_to(target, final_yaw)
        self._station_i[:] = 0.0
        self.step(1.0)
        return bool(np.linalg.norm(target - self.base_xy) < tol * 3.0)

    def fallen(self) -> bool:
        """Low, or tipped past ~45 degrees: the check every demo uses."""
        R = self.data.xmat[self.root_id].reshape(3, 3)
        return bool(self.data.xpos[self.root_id][2] < 0.55 or R[2, 2] < 0.7)

    def gait_travel(self, target, arrive: float = 0.30,
                    timeout: float = 30.0, until=None) -> bool:
        """Walk to within `arrive` of `target` under the walking policy,
        steered by controllers/navigator.py, which keeps every command
        inside the envelope measured by eval/score_gait_commands.py.

        Leaves the robot walking: the policy cannot stop itself, so the
        caller hands over with gait_stop(). Refuses to start from a standing
        robot, because the stance-to-walk handover is the one that falls.
        `until(runtime)` ends the walk early when it returns True: aiming at
        a point beyond where the walk ends keeps the steering active the
        whole way (it freezes within FREEZE_R of the target).
        """
        if self._standing:
            raise RuntimeError(
                "gait_travel starts from a walking robot; stance-to-walk "
                "handover is unsupported (see README). Use Runtime(..., "
                "start='walk').")
        # `target` may be a callable(runtime) -> xy, re-evaluated every
        # tick: a lookahead point on a line gives line following.
        goal = target if callable(target) else (
            lambda _r, t=np.asarray(target, dtype=float)[:2]: t)
        self.station = None          # the navigator owns the command
        nav = GaitNavigator()
        nav.reset(float(self.loco.command[0]), float(self.loco.command[2]))
        tick = self.loco.n_substeps * self.dt
        t_end = self._t + timeout
        while self._t < t_end:
            target = np.asarray(goal(self), dtype=float)[:2]
            if arrived(self.base_xy, self.base_yaw, target, arrive):
                return True
            if until is not None and until(self):
                return True
            vx, wz = nav.command(self.base_xy, self.base_yaw, target, tick)
            self.loco.set_command(vx, 0.0, wz)
            self.step(tick)
            if self.fallen():
                return False
        return False

    def gait_stop(self, ramp: float = 0.6, timed: bool = True,
                  max_speed: float = 0.2, foot_tol: float = 0.012,
                  max_wait: float = 4.0, lead: float = 0.0,
                  brake: float = 0.0, brake_max: float = 0.3,
                  still_arms: bool = False, catch_s: float = 2.0) -> dict:
        """Ease the walking command to zero, then give the legs to the
        stance policy.

        Zero alone does not stop the walking policy (it creeps ~0.3 m/s),
        and the stance policy was trained to hold a standing robot, not to
        catch a walking one: handed over at a random moment it drops the
        robot 9 times in 15 (eval/score_handover.py). Trials that survived
        were moving ~0.20 m/s at handover, falls ~0.37 m/s, and handovers
        with a foot in the air fell. So when `timed`, keep the walking
        policy running at zero command and hand over only once the body is
        slower than `max_speed` with both feet down, or after `max_wait`.
        `lead` places the stance hold point that many seconds ahead along
        the current velocity, where the robot would come to rest anyway.
        `brake` > 0 replaces the zero command during the wait with one
        opposing the measured velocity (gain `brake`, capped at
        `brake_max` m/s), so the creep dies sooner instead of on its own.
        `still_arms` sets the arm targets to where the arms are at handover.
        Otherwise they keep the walking policy's last mid-swing target, one
        arm forward and one back, and stand_still() stiffens them onto it.
        With a catch policy configured (locomotion.catch_policy), it takes
        the legs for `catch_s` first: trained only on this moment, from
        recorded walking states with the arms held still. The stance policy
        then takes over a robot that is standing quietly, which it handles.
        """
        v0 = self.loco.command.copy()
        tick = self.loco.n_substeps * self.dt
        n = max(1, int(round(ramp / tick)))
        for k in range(1, n + 1):
            self.loco.set_command(*(v0 * (1.0 - k / n)))
            self.step(tick)
        root_dof = self.model.jnt_dofadr[self.model.body_jntadr[self.root_id]]
        feet = [self.model.body(f).id
                for f in self.body.cfg.get("locomotion", {}).get("feet", [])]
        waited, forced = 0.0, False
        if timed:
            floor_z = None
            while True:
                z = [self.data.xpos[f][2] for f in feet]
                floor_z = min(z) if floor_z is None else min(floor_z, *z)
                speed = float(np.linalg.norm(
                    self.data.qvel[root_dof:root_dof + 2]))
                planted = all(h < floor_z + foot_tol for h in z)
                if speed < max_speed and planted:
                    break
                if waited >= max_wait:
                    forced = True
                    break
                if brake > 0:
                    v = self.data.qvel[root_dof:root_dof + 2]
                    c, s_ = np.cos(self.base_yaw), np.sin(self.base_yaw)
                    local = np.array([c * v[0] + s_ * v[1],
                                      -s_ * v[0] + c * v[1]])
                    cmd = np.clip(-brake * local, -brake_max, brake_max)
                    self.loco.set_command(cmd[0], cmd[1], 0.0)
                self.step(self.dt)
                waited += self.dt
        vel = self.data.qvel[root_dof:root_dof + 2].copy()
        if still_arms or self.catch is not None:
            for a in self.arm.values():
                self.data.ctrl[a["act"]] = [
                    self.data.qpos[self.model.jnt_qposadr[
                        self.model.actuator_trnid[i, 0]]] for i in a["act"]]
        if self.catch is not None and self.unified is None:
            stance, self.stance = self.stance, self.catch
            try:
                self.stand_still()
                self.step(max(0.0, catch_s - 0.5))
            finally:
                self.stance = stance
            stance.hold(self.catch.hold_xy, self.catch.hold_yaw)
            stance.last_action[:] = 0.0
        elif self.stance is not None:
            self.stand_still()
            if lead and self.unified is None:
                self.stance.hold(self.base_xy + lead * vel, self.base_yaw)
        return {"waited": waited, "forced": forced,
                "speed": float(np.linalg.norm(vel))}

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
        w = self.rot_gain * Rotation.from_matrix(R_des @ R_cur.T).as_rotvec()
        cap = self.rot_cap
        n = np.linalg.norm(w)
        return w * (cap / n) if n > cap else w

    def reach_joint_space(self, side: str, target, duration: float = 2.5,
                          tol: float = 0.04, seed: str = "ready",
                          passes: int = 3) -> bool:
        """Reach by solving the IK offline, then moving the arm smoothly in
        joint space.

        A balancing robot tolerates a smooth arm motion (3 cm of base drift
        here) but not a closed-loop Cartesian servo: the servo re-plans every
        tick, so the legs feel a jerky, direction-changing disturbance that
        never settles, and the robot goes down. Solving first and then moving
        once gives the same hand pose with a disturbance the stance policy
        was trained to absorb.
        """
        a, ik = self.arm[side], self.ik[side]
        target = np.asarray(target, dtype=float)
        # Bias the solve towards the arm's natural working pose. Without it
        # the IK is free to wander into contorted answers — elbow
        # hyperextended, wrist pinned against its limit, wrist links in
        # collision — which are kinematically valid and physically jammed:
        # the joint sits half a radian from its command and the hand never
        # arrives.
        posture = self.body.cfg["arms"][side].get("ready")
        posture = np.asarray(posture, dtype=float) if posture else None
        snapshot = self.data.qpos.copy()
        # A descent should start from where the arm already is — reseeding
        # from the ready pose finds a different solution branch and the arm
        # swings across the bench to get to it.
        if seed == "current":
            posture = self.data.qpos[a["qpos"]].copy()
        if posture is not None:
            # Seed every solve from the same natural pose. Seeding from
            # wherever the arm happens to be lets one contorted answer feed
            # the next, and the solutions drift further from anything the
            # arm can actually hold.
            self.data.qpos[a["qpos"]] = posture
            mujoco.mj_forward(self.model, self.data)
        for _ in range(600):          # kinematic solve, physics untouched
            err = target - self.data.site_xpos[ik.site_id]
            if np.linalg.norm(err) < 0.008:
                break
            qd = ik.solve(self.data, np.r_[3.0 * err, np.zeros(3)],
                          posture=posture, posture_gain=1.2)
            self.data.qpos[a["qpos"]] = np.clip(
                self.data.qpos[a["qpos"]] + qd * 0.01,
                a["ik_limits"][:, 0], a["ik_limits"][:, 1])
            mujoco.mj_forward(self.model, self.data)
        solution = self.data.qpos[a["qpos"]].copy()
        reached = float(np.linalg.norm(
            target - self.data.site_xpos[ik.site_id]))
        self.data.qpos[:] = snapshot
        mujoco.mj_forward(self.model, self.data)
        if reached > tol:
            return False
        # Pace the move by how far the arm has to travel: a fixed duration
        # makes a large joint change a fast one, and speed is what the legs
        # feel.
        travel = float(np.max(np.abs(solution - self.data.qpos[a["qpos"]])))
        self._interp_ctrl(side, solution, max(duration, 2.0 * travel))
        self.step(0.3)
        err = float(np.linalg.norm(target - self.ee_pos(side)))
        if err < tol * 2.5 or passes <= 1:
            return bool(err < tol * 2.5)
        # The solve happened before the move: the base is balancing, so it
        # shifts and turns while the arm travels and the joint solution no
        # longer points where it did. Re-solve from where the robot is now
        # and close the gap — each correction is small, so it costs the legs
        # almost nothing.
        return self.reach_joint_space(side, target, duration=1.0, tol=tol,
                                      seed="current", passes=passes - 1)

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
            if self.qdot_cap is not None:   # limit the disturbance the legs feel
                n = np.linalg.norm(qdot)
                if n > self.qdot_cap:
                    qdot *= self.qdot_cap / n
            q = self.data.qpos[a["qpos"]]
            # anti-windup: the position target may never drift far from the
            # actual joints, or a blocked arm winds up and shoves through
            new_ctrl = self.data.ctrl[a["act"]] + qdot * self.dt
            new_ctrl = np.clip(new_ctrl, q - WINDUP, q + WINDUP)
            self.data.ctrl[a["act"]] = np.clip(
                new_ctrl, a["limits"][:, 0], a["limits"][:, 1])
            self.step(self.dt)
        return np.linalg.norm(target - self.ee_pos(side)) < tol * 2

    @staticmethod
    def _min_jerk(s: float) -> float:
        """Minimum-jerk time scaling: 10s^3 - 15s^4 + 6s^5.

        People do not move between two points on a smoothstep. Reaching
        follows a minimum-jerk profile (Flash & Hogan): a single smooth
        bell-shaped speed curve, zero velocity *and* zero acceleration at
        both ends. It is the difference between a motion that starts and
        stops crisply and one that eases the way an arm actually does.
        """
        return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)

    @staticmethod
    def _smoothstep(s: float) -> float:
        """The gentler profile a balancing body's arms move on."""
        return s * s * (3.0 - 2.0 * s)

    def _interp_ctrl(self, side: str, target, duration: float | None = None,
                     counterbalance: bool = True) -> None:
        """Move one arm along a minimum-jerk path, with the rest of the body
        joining in the way a person's does."""
        a = self.arm[side]
        start = self.data.ctrl[a["act"]].copy()
        target = np.asarray(target, dtype=float)
        travel = float(np.max(np.abs(target - start)))
        if duration is None:
            # Bigger motions take longer, but sub-linearly: a reach twice as
            # far is not twice as slow. People scale duration with distance
            # roughly like this, and a fixed duration makes long motions
            # frantic and short ones sluggish.
            duration = float(np.clip(0.55 + 1.5 * np.sqrt(travel), 0.7, 3.2))

        # Minimum jerk on a fixed base, smoothstep on a balancing one. The
        # human profile is not free here: the stance policy learned to reject
        # arm disturbances shaped like a smoothstep, and swapping in minimum
        # jerk makes this robot reach correctly and then fall over, on every
        # target. Stretching the move to 1.75x reaches better in isolation
        # (3/3 upright, 63 mm) but costs the walk-and-fetch, which needs the
        # arm quick enough to not hold a load out through a long move. Moving
        # a balancing body's arms like a person's is a retraining job, not a
        # tuning one, the same conclusion the gait reached.
        profile = (self._min_jerk if self.loco is None
                   or self.arm_profile == "min_jerk" else self._smoothstep)
        if self.loco is not None:
            # The free arm stays out of it too: a counterbalancing swing is a
            # second disturbance bought for a cosmetic gain.
            counterbalance = False

        other = "left" if side == "right" else "right"
        free_arm = (counterbalance and other in self.arm
                    and self.held(other) is None
                    and self.arm[other]["home_ctrl"] is not None)
        if free_arm:
            other_start = self.data.ctrl[self.arm[other]["act"]].copy()
            swing = np.zeros_like(other_start)
            swing[0] = 0.10 * np.sign(travel)   # shoulder pitch, gently

        n = max(1, int(duration / self.dt))
        for i in range(n):
            s = profile((i + 1) / n)
            self.data.ctrl[a["act"]] = (1 - s) * start + s * target
            if free_arm:
                # The free arm does not hang dead while the other works: it
                # drifts out and back, which is what makes a one-armed
                # motion read as a person rather than a manipulator.
                self.data.ctrl[self.arm[other]["act"]] = (
                    other_start + swing * np.sin(np.pi * s))
            self.step(self.dt)

    def _interp_ctrl_path(self, side: str, poses, duration: float) -> None:
        """Pass through joint-space via poses as ONE motion: a Catmull-Rom
        curve through them, on a single minimum-jerk clock, so the arm does
        not come to rest at every via pose (three stop-start bell curves
        read as a machine; one rounded swing reads as a person)."""
        a = self.arm[side]
        pts = np.array([self.data.ctrl[a["act"]].copy()]
                       + [np.asarray(p, dtype=float) for p in poses])
        curve = self._catmull_rom(pts)
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0),
                                                  axis=1))]
        n = max(1, int(duration / self.dt))
        for i in range(n):
            d = self._min_jerk((i + 1) / n) * arc[-1]
            k = int(np.clip(np.searchsorted(arc, d) - 1, 0, len(curve) - 2))
            seg = max(arc[k + 1] - arc[k], 1e-9)
            u = (d - arc[k]) / seg
            self.data.ctrl[a["act"]] = (1 - u) * curve[k] + u * curve[k + 1]
            self.step(self.dt)

    def reach_via(self, side: str, waypoints, duration: float | None = None,
                  settle: float = 0.25, **kw) -> bool:
        """Move the hand through waypoints as ONE motion.

        Reaching over an obstacle as "up, stop, across, stop, down" is the
        loudest machine tell left in the stack: three separate bell curves
        with the hand at rest between them. People pass *through* a via point
        without stopping: the corner rounds off and the whole reach keeps a
        single bell-shaped speed profile. So the waypoints become a smooth
        curve (Catmull-Rom, sampled by arc length) traversed on one
        minimum-jerk clock.
        """
        pts = [self.ee_pos(side)] + [np.asarray(w, dtype=float)
                                     for w in waypoints]
        pts = [p for i, p in enumerate(pts)
               if i == 0 or np.linalg.norm(p - pts[i - 1]) > 1e-4]
        if len(pts) < 2:
            return True
        if len(pts) == 2:
            return self.reach_line(side, pts[1], duration, settle, **kw)
        curve = self._catmull_rom(np.array(pts))
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0),
                                                  axis=1))]
        total = float(arc[-1])
        if duration is None:
            duration = float(np.clip(0.5 + 1.7 * np.sqrt(total), 0.6, 3.0))

        def path(u):
            return np.array([np.interp(u * total, arc, curve[:, k])
                             for k in range(3)])

        return self._track(side, path, pts[-1], duration, settle,
                           via=pts[1:-1], **kw)

    @staticmethod
    def _catmull_rom(pts: np.ndarray, per_span: int = 40) -> np.ndarray:
        """Smooth interpolating spline through the waypoints. Interpolating,
        not approximating: a via point placed to clear an obstacle has to be
        passed through, not merely approached."""
        ext = np.vstack([pts[0] + (pts[0] - pts[1]), pts,
                         pts[-1] + (pts[-1] - pts[-2])])
        out = []
        for i in range(len(pts) - 1):
            p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
            t = np.linspace(0, 1, per_span, endpoint=(i == len(pts) - 2))[:, None]
            out.append(0.5 * ((2 * p1)
                              + (-p0 + p2) * t
                              + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
        return np.vstack(out)

    def reach_line(self, side: str, target, duration: float | None = None,
                   settle: float = 0.25, correct: bool = True,
                   counterbalance: bool = True,
                   lead: float | None = None, cap: float | None = None,
                   gain: float | None = None) -> bool:
        """Move the hand along a straight line with a bell-shaped speed.

        Those two things are what make a reach look human. A position servo
        gets the straight line for free but its speed profile is wrong: it
        creeps in and peaks at ~83% of the motion, accelerating into the
        target, where a person decelerates into it and peaks near the middle.
        So the trajectory is planned in time (minimum jerk along the line)
        and tracked, rather than left to a controller chasing an error.
        """
        a = self.arm[side]
        start = self.ee_pos(side)
        target = np.asarray(target, dtype=float)
        dist = float(np.linalg.norm(target - start))
        if dist < 1e-4:
            return True
        if duration is None:
            duration = float(np.clip(0.5 + 1.7 * np.sqrt(dist), 0.6, 2.6))
        return self._track(side, lambda u: start + u * (target - start),
                           target, duration, settle, correct=correct,
                           counterbalance=counterbalance, lead=lead, cap=cap,
                           gain=gain)

    def _track(self, side: str, path, target, duration: float,
               settle: float = 0.25, correct: bool = True,
               counterbalance: bool = True,
               lead: float | None = None, cap: float | None = None,
               gain: float | None = None, via=None) -> bool:
        """Track `path(u)`, u in [0, 1], on a minimum-jerk clock.

        `path` is the geometry and the clock is the timing; separating them
        is what lets a straight reach and a blended via-point reach share one
        controller and one speed profile. `via` is carried only so an
        evaluator can see which waypoints this motion was meant to round.
        """
        a = self.arm[side]
        target = np.asarray(target, dtype=float)
        lead = LEAD if lead is None else lead
        cap = SPEED_CAP if cap is None else cap
        gain = TRACK_GAIN if gain is None else gain
        n = max(2, int(duration / self.dt))
        prev = path(0.0)
        for i in range(n):
            ref = path(self._min_jerk((i + 1) / n))
            # The arm is position-actuated, so the hand lags whatever the
            # plan asks for. Tracking the reference at the present instant
            # therefore peaks late (measured: 66% of the motion) and lands
            # tens of millimetres short. Aiming at where the plan will be
            # LEAD seconds from now cancels that lag, which is also what a
            # person does: reaching is feed-forward, not error-chasing.
            aim = path(self._min_jerk(min(1.0, (i + 1) / n + lead / duration)))
            v = (ref - prev) / self.dt + gain * (aim - self.ee_pos(side))
            n_v = np.linalg.norm(v)
            if n_v > cap:
                v *= cap / n_v
            # Position and orientation are one 6-D task to a damped
            # least-squares solve, which means they compete for the same
            # joint velocity. The palm's canonical orientation is not always
            # reachable, so its term never converges and sits at the rate cap
            # and once the min-jerk profile decelerates, that constant
            # angular demand is the largest thing left in the command and
            # drags the hand off target. Measured: the last stretch of every
            # reach walked *away*, 30 mm out to 60 mm. So orientation is
            # demoted near the target: get the palm level during transport,
            # then hold the wrist still and land the hand.
            fade = min(1.0, np.linalg.norm(target - self.ee_pos(side))
                       / ORIENT_FADE)
            qdot = self.ik[side].solve(
                self.data, np.r_[v, fade * self._orientation_twist(side)])
            qdot *= joint_limit_scale(self.data.qpos[a["qpos"]], a["limits"])
            q = self.data.qpos[a["qpos"]]
            self.data.ctrl[a["act"]] = np.clip(
                np.clip(self.data.ctrl[a["act"]] + qdot * self.dt,
                        q - WINDUP, q + WINDUP),
                a["limits"][:, 0], a["limits"][:, 1])
            prev = ref
            self.step(self.dt)
        # Settle only if nothing follows. A person's corrective submovement
        # runs straight out of the primary one; pausing between them is the
        # stop-and-go that makes a motion read as a machine.
        err = float(np.linalg.norm(target - self.ee_pos(side)))
        self.step(0.05 if (correct and err > CORRECT_TOL) else settle)
        # Corrective submovement. A ballistic reach undershoots, as it does for
        # people too, which is why human reaching is a fast primary movement
        # followed by a short, slow correction (Woodworth 1899). Doing the
        # same here is both the honest model and what takes the landing error
        # from ~24 mm to under 10 mm, which is the difference between a grasp
        # and a nudge.
        err = float(np.linalg.norm(target - self.ee_pos(side)))
        if correct and err > CORRECT_TOL:
            # No lead and a low speed cap: anticipation is what makes the
            # primary movement fast, and exactly what makes a short
            # correction overshoot and ring.
            self.reach_line(side, target,
                            duration=float(np.clip(0.15 + 4.0 * err,
                                                   CORRECT_MIN, CORRECT_TIME)),
                            settle=0.15, correct=False, counterbalance=False,
                            lead=0.0, cap=0.15)
            err = float(np.linalg.norm(target - self.ee_pos(side)))
        return bool(err < 0.05)

    def gaze_at(self, world_pos, duration: float = 0.5) -> None:
        """Turn to look at what you are about to touch.

        People orient head and torso to a target before the hand arrives:
        gaze leads the reach by a few hundred milliseconds. Skipping it is a
        large part of why a robot doing an otherwise-correct motion still
        reads as a machine. On a balancing body the waist belongs to the
        locomotion policy, so this stays out of its way.
        """
        gaze = self.body.cfg.get("gaze_joint")
        if self.loco is not None or gaze is None:
            return
        act = self.body.actuator_of.get(gaze)
        if act is None:
            return
        local = self.to_base(world_pos)
        # Capped deliberately: the manipulation poses are tuned for a
        # square torso, and past ~0.2 rad the twist moves the arm's frame
        # enough that the reach misses. A glance, not a turn.
        yaw = float(np.clip(np.arctan2(local[1], max(local[0], 0.05)),
                            -GAZE_MAX, GAZE_MAX))
        lo, hi = self.model.actuator_ctrlrange[act]
        target = float(np.clip(yaw, lo, hi))
        start = float(self.data.ctrl[act])
        n = max(1, int(duration / self.dt))
        for i in range(n):
            s = self._min_jerk((i + 1) / n)
            self.data.ctrl[act] = (1 - s) * start + s * target
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
        if via is not None and not isinstance(via[0], (list, tuple)):
            via = [via]
        if self.blend_vias:
            path = list(reversed(via or [])) + [a["home_ctrl"]]
            self._interp_ctrl_path(side, path, duration * 0.5 * len(path))
            return
        if via is not None:
            for pose in reversed(via):
                self._interp_ctrl(side, pose, duration * 0.5)
        self._interp_ctrl(side, a["home_ctrl"], duration * 0.5)

    def move_arm_ready(self, side: str, duration: float = None) -> None:
        """Joint-space move to the config's elbow-up 'ready' pose. The home
        pose (arm hanging) is near-singular; Cartesian servoing starts here.
        An optional 'ready_via' waypoint routes the swing over the table edge
        instead of through it."""
        a = self.arm[side]
        if a["home_ctrl"] is None:
            a["home_ctrl"] = self.data.ctrl[a["act"]].copy()
        # A balancing robot needs the arm moved gently: fast swings are a
        # momentum disturbance it has to catch with a step.
        if duration is None:
            duration = 3.0 if self.loco is not None else 1.2
        cfg = self.body.cfg["arms"][side]
        via, ready = cfg.get("ready_via"), cfg.get("ready")
        if via is not None and not isinstance(via[0], (list, tuple)):
            via = [via]
        if self.blend_vias and ready is not None:
            path = list(via or []) + [ready]
            self._interp_ctrl_path(side, path, duration * 0.6 * len(path))
            return
        if via is not None:
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
