"""Unit tests for models.py: the Wilson-Cowan ODE, its fixed-point solver,
and its analytic Jacobian."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models import WilsonCowanParams, WilsonCowanPopulation, sigmoid  # noqa: E402


SIMPLE_PARAMS = WilsonCowanParams(
    tau_E=10.0, tau_I=20.0, w_EE=12.0, w_EI=10.0, w_IE=10.0, w_II=2.0,
    a_E=1.2, theta_E=4.0, a_I=1.0, theta_I=3.5, I_E=2.0,
)


def test_sigmoid_bounds_and_monotonicity():
    x = np.linspace(-50, 50, 500)
    s = sigmoid(x, a=1.3, theta=2.0)
    assert np.all(s >= 0) and np.all(s <= 1)
    assert np.all(np.diff(s) >= 0)  # monotonically increasing
    assert s[0] < 0.01 and s[-1] > 0.99


def test_fixed_point_is_a_true_equilibrium():
    """The (E*, I*) returned by fixed_point() should have (approximately)
    zero derivative under the deterministic ODE."""
    wc = WilsonCowanPopulation(SIMPLE_PARAMS)
    E_star, I_star = wc.fixed_point(I_I=3.0)
    dE, dI = wc._derivatives(E_star, I_star, I_I=3.0)
    assert abs(dE) < 1e-6
    assert abs(dI) < 1e-6


def test_jacobian_matches_finite_difference():
    """The analytic Jacobian must match a central finite-difference
    approximation of the same right-hand side, at an arbitrary
    (non-equilibrium) point -- this is the check that the hand-derived
    partial derivatives in models.py are correct."""
    wc = WilsonCowanPopulation(SIMPLE_PARAMS)
    E0, I0, I_I = 0.3, 0.2, 3.0
    h = 1e-6

    def f(E, I):
        return np.array(wc._derivatives(E, I, I_I))

    J_fd = np.zeros((2, 2))
    J_fd[:, 0] = (f(E0 + h, I0) - f(E0 - h, I0)) / (2 * h)
    J_fd[:, 1] = (f(E0, I0 + h) - f(E0, I0 - h)) / (2 * h)

    J_analytic = wc.jacobian(E0, I0, I_I)
    np.testing.assert_allclose(J_analytic, J_fd, atol=1e-5)


def test_jacobian_at_fixed_point_eigenvalues_are_finite():
    wc = WilsonCowanPopulation(SIMPLE_PARAMS)
    for I_I in [0.0, 2.0, 4.0, 6.0]:
        E_star, I_star = wc.fixed_point(I_I)
        J = wc.jacobian(E_star, I_star, I_I)
        eig = np.linalg.eigvals(J)
        assert np.all(np.isfinite(eig))


def test_deterministic_simulation_is_reproducible():
    wc = WilsonCowanPopulation(SIMPLE_PARAMS)
    t1, E1, I1 = wc.simulate_deterministic(I_I=3.0, t_end=200.0, dt=0.5)
    t2, E2, I2 = wc.simulate_deterministic(I_I=3.0, t_end=200.0, dt=0.5)
    np.testing.assert_array_equal(E1, E2)
    np.testing.assert_array_equal(I1, I2)


def test_deterministic_simulation_converges_to_fixed_point_when_stable():
    """For an I_I comfortably on the stable side, a long deterministic
    simulation started away from equilibrium should relax onto the fixed
    point found by the root-finder."""
    wc = WilsonCowanPopulation(SIMPLE_PARAMS)
    I_I = 6.0
    E_star, I_star = wc.fixed_point(I_I)
    J = wc.jacobian(E_star, I_star, I_I)
    assert np.all(np.linalg.eigvals(J).real < 0)  # confirm this I_I IS stable

    t, E, I = wc.simulate_deterministic(I_I, t_end=3000.0, dt=0.5, init=(0.5, 0.5))
    assert abs(E[-1] - E_star) < 1e-3
    assert abs(I[-1] - I_star) < 1e-3


def test_stochastic_simulation_fluctuates_around_fixed_point():
    """A stable condition's stochastic trajectory should stay concentrated
    near its fixed point rather than drifting away or blowing up."""
    wc = WilsonCowanPopulation(SIMPLE_PARAMS)
    I_I = 6.0
    E_star, I_star = wc.fixed_point(I_I)
    rng = np.random.default_rng(0)
    t, E, I = wc.simulate_stochastic(I_I, t_end=5000.0, dt=0.25, sigma=0.015,
                                      rng=rng, init=(E_star, I_star))
    assert abs(E.mean() - E_star) < 0.02
    assert abs(I.mean() - I_star) < 0.02
    assert E.std() < 0.1  # bounded, not blown up
