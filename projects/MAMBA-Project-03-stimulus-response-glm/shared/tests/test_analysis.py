"""Unit tests for analysis.py, run against synthetic data with KNOWN
parameters (not the committed dataset)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models import (  # noqa: E402
    TrueEffects, DriftParams, simulate_session, build_design_matrix, drift_basis,
)
from analysis import (  # noqa: E402
    fit_ols, omitted_variable_bias, residual_autocorrelation, nested_f_test,
    evaluate_contrast, predict, holdout_metrics, residual_bootstrap_ci,
)


def test_fit_ols_recovers_known_coefficients_with_zero_noise():
    rng = np.random.default_rng(0)
    effects = TrueEffects(beta0=5, beta_45=1.2, beta_90=3.5, beta_135=0.5, beta_contrast=2.0, sigma_noise=0.0)
    drift = DriftParams(d_lin=0, d_quad=0, d_sin=0, d_cos=0)
    session = simulate_session(300, effects, drift, rng)
    X = build_design_matrix(session["orientation_deg"], session["contrast_pct"])
    fit = fit_ols(X, session["response"])
    expected = np.array([effects.beta0, effects.beta_45, effects.beta_90, effects.beta_135, effects.beta_contrast])
    np.testing.assert_allclose(fit.beta, expected, atol=1e-8)
    assert fit.rss == pytest.approx(0.0, abs=1e-12)


def test_fit_ols_se_shrinks_with_more_data():
    rng = np.random.default_rng(1)
    effects = TrueEffects(beta0=5, beta_45=1.2, beta_90=3.5, beta_135=0.5, beta_contrast=2.0, sigma_noise=2.0)
    drift = DriftParams(d_lin=0, d_quad=0, d_sin=0, d_cos=0)
    session_small = simulate_session(100, effects, drift, rng)
    session_large = simulate_session(4000, effects, drift, rng)
    X_small = build_design_matrix(session_small["orientation_deg"], session_small["contrast_pct"])
    X_large = build_design_matrix(session_large["orientation_deg"], session_large["contrast_pct"])
    fit_small = fit_ols(X_small, session_small["response"])
    fit_large = fit_ols(X_large, session_large["response"])
    assert np.all(fit_large.se < fit_small.se)


def test_omitted_variable_bias_matches_repeated_simulation_average():
    """The core cross-check required by the project workflow: verify the
    exact omitted-variable-bias formula against the AVERAGE bias observed
    across many repeated noise draws with the SAME design (isolating
    systematic bias from sampling noise)."""
    rng = np.random.default_rng(2)
    effects = TrueEffects(beta0=5, beta_45=1.2, beta_90=3.5, beta_135=0.5, beta_contrast=2.0, sigma_noise=1.5)
    drift = DriftParams(d_lin=-3.0, d_quad=2.0, d_sin=1.5, d_cos=-1.0)
    n = 400

    # fixed design (orientation/contrast schedule + trial times), redraw noise only
    design_rng = np.random.default_rng(3)
    session0 = simulate_session(n, effects, drift, design_rng)
    ori, contrast, t_frac = session0["orientation_deg"], session0["contrast_pct"], session0["t_frac"]

    X_naive = build_design_matrix(ori, contrast)
    X_drift = drift_basis(t_frac)
    beta_drift_true = np.array([drift.d_lin, drift.d_quad, drift.d_sin, drift.d_cos])
    predicted_bias = omitted_variable_bias(X_naive, X_drift, beta_drift_true)

    mean_response = session0["mean_response"]
    n_repeats = 500
    beta_repeats = np.zeros((n_repeats, X_naive.shape[1]))
    noise_rng = np.random.default_rng(4)
    for r in range(n_repeats):
        y_r = mean_response + noise_rng.normal(0, effects.sigma_noise, size=n)
        beta_repeats[r] = fit_ols(X_naive, y_r).beta

    true_beta = np.array([effects.beta0, effects.beta_45, effects.beta_90, effects.beta_135, effects.beta_contrast])
    empirical_bias = beta_repeats.mean(axis=0) - true_beta
    np.testing.assert_allclose(empirical_bias, predicted_bias, atol=0.03)


def test_omitted_variable_bias_is_zero_for_perfectly_orthogonal_omitted_regressor():
    rng = np.random.default_rng(5)
    n = 8
    X_included = np.column_stack([np.ones(n), np.array([1, -1, 1, -1, 1, -1, 1, -1.0])])
    # an omitted regressor exactly orthogonal to both columns of X_included
    X_omitted = np.array([1.0, 1, -1, -1, 1, 1, -1, -1]).reshape(-1, 1)
    assert np.allclose(X_included.T @ X_omitted, 0, atol=1e-10)
    bias = omitted_variable_bias(X_included, X_omitted, np.array([10.0]))
    np.testing.assert_allclose(bias, 0.0, atol=1e-10)


def test_omitted_variable_bias_hand_example():
    """The exact hand-checkable example from the handout Section 3.3."""
    X1 = np.array([[1.0], [2.0], [3.0], [4.0]])
    X2 = np.array([[1.0], [1.0], [2.0], [3.0]])
    beta2 = np.array([5.0])
    bias = omitted_variable_bias(X1, X2, beta2)
    assert bias[0] == pytest.approx(3.5, abs=1e-8)


def test_residual_autocorrelation_near_zero_for_white_noise():
    rng = np.random.default_rng(6)
    r = rng.normal(0, 1, size=5000)
    ac = residual_autocorrelation(r, lags=[1, 5, 10])
    for lag, val in ac.items():
        assert abs(val) < 0.05


def test_residual_autocorrelation_high_for_smooth_signal():
    t = np.linspace(0, 4 * np.pi, 2000)
    r = np.sin(t)  # smooth, strongly autocorrelated at short lags
    ac = residual_autocorrelation(r, lags=[1, 5])
    assert ac[1] > 0.9


def test_nested_f_test_detects_true_extra_signal():
    rng = np.random.default_rng(7)
    effects = TrueEffects(beta0=5, beta_45=1.2, beta_90=3.5, beta_135=0.5, beta_contrast=2.0, sigma_noise=1.5)
    drift = DriftParams(d_lin=-3.0, d_quad=2.0, d_sin=1.5, d_cos=-1.0)
    session = simulate_session(500, effects, drift, rng)
    X_naive = build_design_matrix(session["orientation_deg"], session["contrast_pct"])
    X_full = build_design_matrix(session["orientation_deg"], session["contrast_pct"],
                                  include_drift_basis=True, t_frac=session["t_frac"])
    fit_naive = fit_ols(X_naive, session["response"])
    fit_full = fit_ols(X_full, session["response"])
    result = nested_f_test(fit_naive, fit_full)
    assert result.pvalue < 1e-6


def test_nested_f_test_null_when_no_extra_signal():
    rng = np.random.default_rng(8)
    effects = TrueEffects(beta0=5, beta_45=1.2, beta_90=3.5, beta_135=0.5, beta_contrast=2.0, sigma_noise=1.5)
    drift = DriftParams(d_lin=0.0, d_quad=0.0, d_sin=0.0, d_cos=0.0)  # no true drift
    session = simulate_session(500, effects, drift, rng)
    X_naive = build_design_matrix(session["orientation_deg"], session["contrast_pct"])
    X_full = build_design_matrix(session["orientation_deg"], session["contrast_pct"],
                                  include_drift_basis=True, t_frac=session["t_frac"])
    fit_naive = fit_ols(X_naive, session["response"])
    fit_full = fit_ols(X_full, session["response"])
    result = nested_f_test(fit_naive, fit_full)
    assert result.pvalue > 0.05


def test_contrast_recovers_known_difference():
    rng = np.random.default_rng(9)
    effects = TrueEffects(beta0=5, beta_45=1.2, beta_90=3.5, beta_135=0.5, beta_contrast=2.0, sigma_noise=0.5)
    drift = DriftParams(d_lin=0, d_quad=0, d_sin=0, d_cos=0)
    session = simulate_session(3000, effects, drift, rng)
    X = build_design_matrix(session["orientation_deg"], session["contrast_pct"])
    fit = fit_ols(X, session["response"])
    c = np.array([0, 0, 1, 0, 0.0])  # D90 vs baseline
    result = evaluate_contrast(c, fit)
    assert result.estimate == pytest.approx(3.5, abs=0.1)
    assert result.pvalue < 1e-10


def test_holdout_metrics_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0])
    m = holdout_metrics(y, y)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)


def test_holdout_metrics_worse_than_mean_gives_negative_r2():
    y = np.array([1.0, 2.0, 3.0])
    bad_pred = np.array([10.0, -10.0, 20.0])
    m = holdout_metrics(y, bad_pred)
    assert m["r2"] < 0


def test_residual_bootstrap_ci_runs_and_brackets_point_estimate():
    rng = np.random.default_rng(10)
    effects = TrueEffects(beta0=5, beta_45=1.2, beta_90=3.5, beta_135=0.5, beta_contrast=2.0, sigma_noise=1.5)
    drift = DriftParams(d_lin=0, d_quad=0, d_sin=0, d_cos=0)
    session = simulate_session(400, effects, drift, rng)
    X = build_design_matrix(session["orientation_deg"], session["contrast_pct"])
    fit = fit_ols(X, session["response"])
    c = np.array([0, 0, 1, 0, 0.0])
    boot = residual_bootstrap_ci(X, fit, c, n_boot=500, rng=np.random.default_rng(11))
    assert len(boot) == 500
    point_estimate = c @ fit.beta
    lo, hi = np.percentile(boot, [2.5, 97.5])
    assert lo < point_estimate < hi
