"""Unit tests for models.py: the exact OU transition, and the variogram
formula's derivation."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models import OUParams, OUProcess, variogram_formula  # noqa: E402


def test_stationary_variance_matches_formula():
    p = OUParams(tau=25.0, V_ss=-60.0, sigma_p=0.9)
    ou = OUProcess(p)
    assert ou.stationary_variance == pytest.approx(p.sigma_p ** 2 * p.tau / 2)


def test_transition_mean_decays_toward_V_ss():
    p = OUParams(tau=10.0, V_ss=-60.0, sigma_p=0.5)
    ou = OUProcess(p)
    # starting far from V_ss, the mean should move monotonically closer
    # to V_ss as dt increases
    dts = np.array([0.0, 5.0, 10.0, 50.0, 1000.0])
    means = ou.transition_mean(np.full_like(dts, -40.0), dts)
    assert means[0] == pytest.approx(-40.0)
    assert np.all(np.diff(means) < 0)  # -40 is above V_ss=-60, so decaying toward it means decreasing
    assert means[-1] == pytest.approx(p.V_ss, abs=1e-2)


def test_transition_variance_grows_toward_stationary_variance():
    p = OUParams(tau=10.0, V_ss=-60.0, sigma_p=0.5)
    ou = OUProcess(p)
    dts = np.array([0.0, 5.0, 10.0, 50.0, 1000.0])
    variances = ou.transition_variance(dts)
    assert variances[0] == pytest.approx(0.0, abs=1e-9)
    assert np.all(np.diff(variances) > 0)
    assert variances[-1] == pytest.approx(ou.stationary_variance, abs=1e-3)


def test_simulate_at_times_reproducible_with_same_rng_state():
    p = OUParams(tau=15.0, V_ss=-55.0, sigma_p=0.7)
    ou = OUProcess(p)
    times = np.sort(np.random.default_rng(0).uniform(0, 1000, size=50))
    V1 = ou.simulate_at_times(times, np.random.default_rng(42))
    V2 = ou.simulate_at_times(times, np.random.default_rng(42))
    np.testing.assert_array_equal(V1, V2)


def test_simulate_at_times_long_run_matches_stationary_moments():
    """A long simulation's empirical mean and variance should converge to
    the theoretical stationary mean (V_ss) and variance."""
    p = OUParams(tau=10.0, V_ss=-60.0, sigma_p=0.8)
    ou = OUProcess(p)
    rng = np.random.default_rng(1)
    times = np.arange(0, 200000, 2.0)  # densely sampled, long duration
    V = ou.simulate_at_times(times, rng)
    assert V.mean() == pytest.approx(p.V_ss, abs=0.1)
    assert V.var() == pytest.approx(ou.stationary_variance, rel=0.1)


def test_variogram_formula_zero_at_zero_lag():
    val = variogram_formula(np.array([0.0]), tau=25.0, sigma_p2_tau=20.0, obs_2sigma2=3.0)
    # at dt=0, the process contributes nothing (1-exp(0)=0); only the
    # obs_2sigma2 "nugget" term remains, since observation noise at the
    # same instant compared with itself would truly be zero -- but the
    # formula's nugget represents INDEPENDENT observation noise draws at
    # two different times, which is the only case the model is meant to
    # apply to (dt>0 in practice); confirm the algebraic value here.
    assert val[0] == pytest.approx(3.0)


def test_variogram_formula_approaches_plateau_at_large_lag():
    tau, sp2tau, obs2sig2 = 25.0, 20.0, 3.0
    val = variogram_formula(np.array([1e6]), tau, sp2tau, obs2sig2)
    assert val[0] == pytest.approx(obs2sig2 + sp2tau, rel=1e-3)


def test_variogram_formula_matches_hand_derivation_via_simulation():
    """Cross-check the closed-form variogram formula against a
    Monte Carlo estimate from many independent simulated pairs at a fixed
    lag -- this is the same check a group should perform on their own
    derivation before trusting it."""
    tau, V_ss, sigma_p, sigma_obs = 20.0, -60.0, 1.0, 0.8
    p = OUParams(tau=tau, V_ss=V_ss, sigma_p=sigma_p)
    ou = OUProcess(p)
    rng = np.random.default_rng(2)

    dt = 15.0
    n_pairs = 200_000
    V0 = rng.normal(V_ss, np.sqrt(ou.stationary_variance), size=n_pairs)
    mean1 = ou.transition_mean(V0, dt)
    var1 = ou.transition_variance(dt)
    V1 = rng.normal(mean1, np.sqrt(var1))
    # add independent observation noise at both endpoints
    V0_obs = V0 + rng.normal(0, sigma_obs, size=n_pairs)
    V1_obs = V1 + rng.normal(0, sigma_obs, size=n_pairs)

    empirical = np.mean((V1_obs - V0_obs) ** 2)
    predicted = variogram_formula(np.array([dt]), tau, sigma_p ** 2 * tau, 2 * sigma_obs ** 2)[0]
    assert empirical == pytest.approx(predicted, rel=0.02)
