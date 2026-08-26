"""6-DOF admittance control, ported line-by-line from the thesis controller
(`excellent_admittance_node.py`, UR3e/ROS 2) into framework-free NumPy.

Pipeline per tick:
  wrench:  hard clip -> per-axis spike rate limit -> soft cubic deadband -> EMA
  dynamics: a = M^-1 (F_shaped - D_eff(|F|) v - K dx),  clamp a, integrate v,
            no-contact decay (hysteretic), anti-windup speed saturation
where D_eff = D (1 + k tanh(|F|/F_ref)) and F_shaped amplifies small forces
and compresses large ones (intent shaping). All quantities world-frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class AdmittanceParams:
    mass_linear: float = 3.0
    mass_angular: float = 0.5
    damping_linear: float = 40.0
    damping_angular: float = 8.0
    guiding_damping_linear: float = 25.0
    guiding_damping_angular: float = 4.0
    stiffness_linear: float = 50.0          # N/m: restores toward reference
    stiffness_angular: float = 5.0          # N·m/rad
    stiffness_guiding_scale: float = 0.15

    force_deadband: float = 3.0             # N per axis
    torque_deadband: float = 0.2            # N·m per axis
    alpha_wrench: float = 0.15              # EMA coefficient
    force_hard_max: float = 80.0
    torque_hard_max: float = 10.0
    force_spike_rate: float = 200.0         # N/s
    torque_spike_rate: float = 20.0

    adaptive_damping_k_lin: float = 1.5
    adaptive_damping_k_ang: float = 0.5
    adaptive_damping_ref_force: float = 10.0
    adaptive_damping_ref_torque: float = 1.0

    intent_saturation_force: float = 15.0
    intent_ref_force: float = 5.0
    intent_saturation_torque: float = 2.0
    intent_ref_torque: float = 0.5

    contact_enter: float = 3.5              # N (hysteretic)
    contact_exit: float = 1.5
    no_contact_decay_tau: float = 0.15      # s

    max_linear_speed: float = 0.20          # m/s
    max_angular_speed: float = 1.0
    max_linear_accel: float = 0.7
    max_angular_accel: float = 3.0

    selection: tuple = (1, 1, 1, 1, 1, 1)   # compliance axis mask


def soft_deadband(x: float, deadband: float) -> float:
    """Per-axis soft deadband with a cubic transition in [0.7db, 1.3db]."""
    ax = abs(x)
    if ax <= deadband * 0.7:
        return 0.0
    if ax >= deadband * 1.3:
        return np.sign(x) * (ax - deadband)
    t = (ax - 0.7 * deadband) / (0.6 * deadband)
    return np.sign(x) * (0.3 * deadband) * (t * t)


def joint_limit_scale(q: np.ndarray, limits: np.ndarray, buffer: float = 0.15) -> float:
    """[0, 1] smoothstep rolloff as any joint approaches its range limit.
    `limits` is (n, 2); embodiment-agnostic (derive limits from the model)."""
    buffer = max(buffer, 1e-4)
    dist = np.minimum(q - limits[:, 0], limits[:, 1] - q)
    s = np.clip(dist / buffer, 0.0, 1.0)
    s = s * s * (3.0 - 2.0 * s)
    return float(s.min(initial=1.0))


class WrenchFilter:
    def __init__(self, p: AdmittanceParams, dt: float):
        self.p, self.dt = p, dt
        self._last_raw = np.zeros(6)
        self._lp = np.zeros(6)

    def __call__(self, wrench: np.ndarray) -> np.ndarray:
        p = self.p
        raw = np.asarray(wrench, dtype=float).copy()
        raw[:3] = np.clip(raw[:3], -p.force_hard_max, p.force_hard_max)
        raw[3:] = np.clip(raw[3:], -p.torque_hard_max, p.torque_hard_max)

        lim = np.array([p.force_spike_rate * self.dt] * 3
                       + [p.torque_spike_rate * self.dt] * 3)
        raw = self._last_raw + np.clip(raw - self._last_raw, -lim, lim)
        self._last_raw = raw.copy()

        db = [p.force_deadband] * 3 + [p.torque_deadband] * 3
        filtered = np.array([soft_deadband(raw[i], db[i]) for i in range(6)])

        self._lp = p.alpha_wrench * filtered + (1.0 - p.alpha_wrench) * self._lp
        return self._lp.copy()


class AdmittanceController:
    """Maps a filtered external wrench (world frame) to a compliant 6D
    end-effector twist [v; w] (world frame)."""

    def __init__(self, params: AdmittanceParams | None = None, dt: float = 0.002):
        self.p = params or AdmittanceParams()
        self.dt = dt
        p = self.p
        self.M_inv = np.linalg.inv(np.diag([p.mass_linear] * 3 + [p.mass_angular] * 3))
        self.D_normal = np.diag([p.damping_linear] * 3 + [p.damping_angular] * 3)
        self.D_guiding = np.diag([p.guiding_damping_linear] * 3
                                 + [p.guiding_damping_angular] * 3)
        self.K_full = np.diag([p.stiffness_linear] * 3 + [p.stiffness_angular] * 3)
        self.filter = WrenchFilter(p, dt)
        self.v = np.zeros(6)
        self.in_contact = False
        self._ref_pos: np.ndarray | None = None
        self._ref_rot: Rotation | None = None

    def set_reference(self, pos: np.ndarray, quat_wxyz: np.ndarray) -> None:
        self._ref_pos = np.asarray(pos, dtype=float).copy()
        self._ref_rot = Rotation.from_quat(np.roll(quat_wxyz, -1))  # wxyz -> xyzw

    def _drift(self, pos, quat_wxyz) -> np.ndarray:
        if self._ref_pos is None:
            return np.zeros(6)
        dx = np.zeros(6)
        dx[:3] = np.asarray(pos) - self._ref_pos
        r_curr = Rotation.from_quat(np.roll(quat_wxyz, -1))
        dx[3:] = (r_curr * self._ref_rot.inv()).as_rotvec()
        return dx

    def _intent(self, w: np.ndarray) -> np.ndarray:
        p, out = self.p, w.copy()
        f = np.linalg.norm(w[:3])
        if f > 1e-9:
            out[:3] = p.intent_saturation_force * np.tanh(f / p.intent_ref_force) * w[:3] / f
        t = np.linalg.norm(w[3:])
        if t > 1e-9:
            out[3:] = p.intent_saturation_torque * np.tanh(t / p.intent_ref_torque) * w[3:] / t
        return out

    def update(self, wrench: np.ndarray, pos: np.ndarray, quat_wxyz: np.ndarray,
               guiding: bool = False) -> np.ndarray:
        p = self.p
        f_raw = self.filter(wrench)
        f_mag, t_mag = np.linalg.norm(f_raw[:3]), np.linalg.norm(f_raw[3:])

        if self.in_contact:
            if f_mag < p.contact_exit and t_mag < p.contact_exit * 0.2:
                self.in_contact = False
        elif f_mag > p.contact_enter or t_mag > p.contact_enter * 0.2:
            self.in_contact = True

        if guiding:
            K = self.K_full * p.stiffness_guiding_scale
            D_base = self.D_guiding
            self.set_reference(pos, quat_wxyz)  # snap: user moves robot freely
        else:
            K, D_base = self.K_full, self.D_normal

        mult = np.array(
            [1.0 + p.adaptive_damping_k_lin * np.tanh(f_mag / p.adaptive_damping_ref_force)] * 3
            + [1.0 + p.adaptive_damping_k_ang * np.tanh(t_mag / p.adaptive_damping_ref_torque)] * 3)
        D_eff = D_base * mult  # diagonal scaling

        net = self._intent(f_raw) - D_eff @ self.v - K @ self._drift(pos, quat_wxyz)
        a = self.M_inv @ net
        a_max = np.array([p.max_linear_accel] * 3 + [p.max_angular_accel] * 3)
        v_new = self.v + self.dt * np.clip(a, -a_max, a_max)

        if not self.in_contact and not guiding:
            v_new *= np.exp(-self.dt / max(p.no_contact_decay_tau, 1e-6))
            if np.linalg.norm(v_new) < 1e-3:
                v_new[:] = 0.0

        # Anti-windup: saturate AND write back to the integrator state.
        lin = np.linalg.norm(v_new[:3])
        if lin > p.max_linear_speed:
            v_new[:3] *= p.max_linear_speed / lin
        ang = np.linalg.norm(v_new[3:])
        if ang > p.max_angular_speed:
            v_new[3:] *= p.max_angular_speed / ang

        self.v = v_new
        return v_new * np.asarray(p.selection, dtype=float)
