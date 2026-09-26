"""Unit tests for analysis.py: the VAR(1) Jacobian estimator (including its
errors-in-variables correction) and the bifurcation-extrapolation pipeline,
checked against synthetic data generated directly from a KNOWN linear
system (NOT the Wilson-Cowan generator -- that is exercised by the
end-to-end smoke test instead)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import analysis  # noqa: E402


def _make_synthetic_linear_dataset(J_by_condition: dict, roles: dict, sigma_proc=0.02,
                                     sigma_obs=0.01, n_trials=8, T=10000.0, dt=0.5, seed=0):
    """Build a dataset dict shaped like analysis.load_dataset()'s output,
    but generated directly from KNOWN 2x2 linear systems
    dx = J x dt + sigma_proc dW, plus known additive observation noise --
    so the VAR(1) estimator's recovered J can be checked against an exact
    ground truth, and the errors-in-variables correction can be checked
    against a known sigma_obs."""
    rng = np.random.default_rng(seed)
    n = int(T / dt)
    t_axis = np.arange(n) * dt
    sqrt_dt = np.sqrt(dt)

    all_cond, all_trial, all_E, all_I, protocol_rows = [], [], [], [], []
    for cid, J in J_by_condition.items():
        for trial in range(n_trials):
            x = np.zeros((n, 2))
            for k in range(n - 1):
                x[k + 1] = x[k] + dt * (J @ x[k]) + sigma_proc * sqrt_dt * rng.normal(size=2)
            E_obs = x[:, 0] + rng.normal(0, sigma_obs, size=n)
            I_obs = x[:, 1] + rng.normal(0, sigma_obs, size=n)
            all_cond.append(cid)
            all_trial.append(trial)
            all_E.append(E_obs.astype(np.float32))
            all_I.append(I_obs.astype(np.float32))
        protocol_rows.append((cid, float(cid), n_trials, 0.0, T, dt, roles[cid]))

    protocol_arr = np.array(protocol_rows, dtype=[
        ("condition_id", "i8"), ("I_I", "f8"), ("n_trials", "i8"),
        ("burn_in_ms", "f8"), ("duration_ms", "f8"), ("dt_recorded_ms", "f8"),
        ("role", "U10")])

    return {
        "t_ms": t_axis, "E_obs": np.stack(all_E), "I_obs": np.stack(all_I),
        "condition_id": np.asarray(all_cond), "trial": np.asarray(all_trial),
        "protocol": protocol_arr, "sigma_obs": sigma_obs,
    }


# A simple known-stable linear system, same shape as the WC Jacobian.
J_STABLE = np.array([[-0.05, -0.03], [0.04, -0.09]])


def test_var1_estimator_recovers_known_stable_jacobian():
    data = _make_synthetic_linear_dataset({0: J_STABLE}, {0: "fit"}, seed=1)
    E_bar, I_bar = analysis.estimate_fixed_point(data, 0)
    assert abs(E_bar) < 0.05 and abs(I_bar) < 0.05  # should be ~0, the system's fixed point

    J_est = analysis.estimate_jacobian_var1(data, 0, E_bar, I_bar, data["sigma_obs"])
    np.testing.assert_allclose(J_est, J_STABLE, atol=0.02)


def test_var1_estimator_without_correction_is_biased_more_stable_than_truth():
    """Demonstrates why the errors-in-variables correction in
    estimate_jacobian_var1 is necessary: a naive (uncorrected) OLS fit on
    the same noisy data is measurably biased toward LOOKING MORE STABLE
    (more damped) than the true system, because attenuating the discrete
    transition matrix A (which sits close to the identity) toward zero
    makes J=(A-I)/dt more negative than truth -- the naive estimate is not
    just noisier, it is biased in a specific, wrong-for-safety direction.
    """
    J_weak = np.array([[-0.089, 0.0], [0.0, -0.089]])
    data = _make_synthetic_linear_dataset({0: J_weak}, {0: "fit"}, sigma_proc=0.015,
                                            sigma_obs=0.01, seed=2)
    E_bar, I_bar = analysis.estimate_fixed_point(data, 0)

    J_corrected = analysis.estimate_jacobian_var1(data, 0, E_bar, I_bar, data["sigma_obs"])
    J_naive = analysis.estimate_jacobian_var1(data, 0, E_bar, I_bar, sigma_obs=0.0)

    true_trace = np.trace(J_weak)
    # naive estimate's damping (|trace|) should be OVERSTATED relative to truth
    assert abs(np.trace(J_naive)) > abs(true_trace)
    # and the corrected estimate should be much closer to the true trace
    assert abs(np.trace(J_corrected) - true_trace) < abs(np.trace(J_naive) - true_trace)


def test_bifurcation_extrapolation_recovers_known_crossing():
    """Build 4 conditions whose KNOWN Jacobians have dominant-eigenvalue
    real part EXACTLY equal to a linear function of the control value
    (for this matrix family, Re(eigenvalue) = trace/2 exactly, since the
    two diagonal entries are equal and the discriminant is always
    negative -- verified below), giving a known crossing at control=3.0,
    and check that predict_bifurcation recovers it."""
    def J_at(control_value):
        re = -0.03 * (control_value - 3.0)
        return np.array([[re, -0.05], [0.05, re]])

    # sanity-check the assumed algebra before relying on it
    for cv in [5.0, 4.0, 3.5, 3.2]:
        J = J_at(cv)
        eig = np.linalg.eigvals(J)
        assert np.iscomplex(eig[0])
        assert eig[0].real == pytest.approx(-0.03 * (cv - 3.0), abs=1e-9)

    J_by_cond = {i: J_at(cv) for i, cv in enumerate([5.0, 4.0, 3.5, 3.2])}
    roles = {i: "fit" for i in J_by_cond}
    data = _make_synthetic_linear_dataset(J_by_cond, roles, sigma_proc=0.03, sigma_obs=0.01, seed=3)

    fit_results = {cid: analysis.fit_condition(data, cid, data["sigma_obs"])
                   for cid in J_by_cond}
    bp = analysis.predict_bifurcation(fit_results)
    assert bp.I_I_crit == pytest.approx(3.0, abs=0.3)


def test_dominant_eigenvalue_picks_larger_real_part():
    eig = np.array([-0.1 + 0j, 0.05 + 0j])
    assert analysis.dominant_eigenvalue(eig) == pytest.approx(0.05)


def test_predicted_stability_matches_sign_of_crossing():
    bp = analysis.BifurcationPrediction(I_I_crit=3.0, slope_re=-0.02, intercept_re=0.06,
                                          slope_im=0.01, intercept_im=0.05, fit_points={})
    assert analysis.predicted_stability(bp, 5.0) == "stable"
    assert analysis.predicted_stability(bp, 1.0) == "oscillatory"


def test_bootstrap_ci_bounds_are_ordered():
    J_by_cond = {i: J_STABLE for i in range(4)}
    roles = {i: "fit" for i in range(4)}
    data = _make_synthetic_linear_dataset(J_by_cond, roles, seed=5)
    # give conditions distinct I_I values so the linear regression is well-posed
    for i, row in enumerate(data["protocol"]):
        row["I_I"] = 3.0 + i
    summary, draws = analysis.bootstrap_ci(data, list(J_by_cond), n_boot=20, seed=0)
    for k, (lo, mid, hi) in summary.items():
        assert lo <= mid <= hi
        assert len(draws[k]) > 0
