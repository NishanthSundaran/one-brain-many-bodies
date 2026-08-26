import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from controllers.admittance import (AdmittanceController, AdmittanceParams,
                                    WrenchFilter, joint_limit_scale,
                                    soft_deadband)

POS = np.zeros(3)
QUAT = np.array([1.0, 0, 0, 0])


def make(dt=0.002, **overrides):
    c = AdmittanceController(AdmittanceParams(**overrides), dt=dt)
    c.set_reference(POS, QUAT)
    return c


# ---- soft deadband (ported cubic transition) ----

def test_deadband_zero_below_and_shifted_above():
    assert soft_deadband(2.0, 3.0) == 0.0
    assert soft_deadband(-2.0, 3.0) == 0.0
    assert soft_deadband(10.0, 3.0) == pytest.approx(7.0)
    assert soft_deadband(-10.0, 3.0) == pytest.approx(-7.0)


def test_deadband_transition_continuous_and_monotonic():
    xs = np.linspace(0, 8, 400)
    ys = [soft_deadband(x, 3.0) for x in xs]
    assert np.all(np.diff(ys) >= -1e-12)
    assert max(abs(np.diff(ys))) < 0.1  # no jumps


# ---- joint limit rolloff ----

def test_joint_limit_scale_endpoints_and_midpoint():
    limits = np.array([[-1.0, 1.0]])
    assert joint_limit_scale(np.array([0.0]), limits, 0.2) == 1.0
    assert joint_limit_scale(np.array([1.0]), limits, 0.2) == 0.0
    # smoothstep at half buffer = 0.5
    assert joint_limit_scale(np.array([0.9]), limits, 0.2) == pytest.approx(0.5)


# ---- wrench filter ----

def test_filter_hard_clip_and_spike_rate():
    p = AdmittanceParams()
    f = WrenchFilter(p, dt=0.002)
    f(np.zeros(6))
    out_state = f._last_raw.copy()
    f(np.array([1000.0, 0, 0, 0, 0, 0]))  # absurd spike
    step = f._last_raw[0] - out_state[0]
    assert step <= p.force_spike_rate * 0.002 + 1e-12


def test_filter_passes_sustained_force():
    f = WrenchFilter(AdmittanceParams(), dt=0.002)
    for _ in range(2000):
        out = f(np.array([10.0, 0, 0, 0, 0, 0]))
    assert out[0] == pytest.approx(7.0, abs=0.05)  # 10 N minus 3 N deadband


# ---- controller dynamics ----

def test_velocity_saturation_anti_windup():
    c = make()
    for _ in range(5000):
        v = c.update(np.array([80.0, 0, 0, 0, 0, 0]), POS, QUAT)
        assert np.linalg.norm(v[:3]) <= c.p.max_linear_speed + 1e-9
        assert np.linalg.norm(c.v[:3]) <= c.p.max_linear_speed + 1e-9  # state too


def test_adaptive_damping_bounded_and_monotonic():
    p = AdmittanceParams()
    mults = [1 + p.adaptive_damping_k_lin * np.tanh(f / p.adaptive_damping_ref_force)
             for f in (0, 5, 20, 1000)]
    assert mults == sorted(mults)
    assert mults[-1] <= 1 + p.adaptive_damping_k_lin


def test_no_contact_decay_to_rest():
    c = make()
    for _ in range(1000):
        c.update(np.array([20.0, 0, 0, 0, 0, 0]), POS, QUAT)
    assert np.linalg.norm(c.v) > 0.01
    for _ in range(2000):
        v = c.update(np.zeros(6), POS, QUAT)
    assert np.linalg.norm(v) == 0.0


def test_stiffness_restores_toward_reference():
    c = make()
    # displaced 10 cm from reference, no force: acceleration must oppose drift
    c.in_contact = True  # suppress decay so we see the pure K term
    v = c.update(np.zeros(6), np.array([0.1, 0, 0]), QUAT)
    assert v[0] < 0.0


def test_guiding_snaps_reference():
    c = make()
    far = np.array([0.3, 0, 0])
    c.update(np.zeros(6), far, QUAT, guiding=True)
    assert np.allclose(c._ref_pos, far)


def test_regression_snapshot():
    """Pinned output of one deterministic run (formulas ported line-by-line
    from excellent_admittance_node.py; values verified against hand calc)."""
    c = make()
    for _ in range(500):
        v = c.update(np.array([15.0, -5.0, 0, 0, 0, 0.5]), POS, QUAT)
    # steady state ~ F_shaped / D_eff: x: 15 N -> 12 N deadbanded, intent-shaped
    # to 14.55 N, D_eff = 40*(1+1.5 tanh(1.217)) = 90.3 -> 0.161 m/s
    np.testing.assert_allclose(
        v, [0.16131569, -0.02688595, 0.0, 0.0, 0.0, 0.11719256], atol=1e-6)
