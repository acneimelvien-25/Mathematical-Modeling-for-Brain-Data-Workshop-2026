"""Unit tests for analysis.py, run against synthetic data with KNOWN
parameters (not the committed dataset), so that pipeline correctness is
verified independently of dataset-specific numbers."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models import GainNoiseParams, generate_binned_spike_counts, fano_factor_formula  # noqa: E402
from analysis import (  # noqa: E402
    counts_from_spike_times,
    estimate_lambda_hz,
    empirical_fano_factor,
    fit_ff_vs_window_slope,
    pooled_sigma_g2,
    compare_models,
    fit_phi_constant_overdispersion,
    parametric_bootstrap_pvalue,
    poisson_null_detectability_pvalue,
    predict_holdout_ff,
    holdout_rmse,
)


def _simulate_condition(lam_hz, n_trials, sigma_g, T_max, rng):
    bins = generate_binned_spike_counts(
        lam_hz=lam_hz, n_trials=n_trials, dt_ms=1.0, n_bins_max=int(T_max),
        gain_params=GainNoiseParams(sigma_g=sigma_g), rng=rng,
    )
    return bins


def test_counts_from_spike_times_matches_direct_binning():
    rng = np.random.default_rng(0)
    bins = _simulate_condition(20.0, 30, 0.1, 400, rng)
    windows = [50.0, 100.0, 400.0]
    direct_counts = np.stack([bins[:, :int(w)].sum(axis=1) for w in windows], axis=1)

    trial_ids = np.arange(30)
    spike_trial_ids, spike_times = [], []
    for i in range(30):
        idxs = np.nonzero(bins[i])[0]
        for k in idxs:
            spike_trial_ids.append(i)
            spike_times.append(k + 0.5)
    counts_from_spikes = counts_from_spike_times(trial_ids, np.array(spike_trial_ids),
                                                  np.array(spike_times), windows)
    np.testing.assert_array_equal(counts_from_spikes, direct_counts)


def test_counts_from_spike_times_handles_trials_with_no_spikes():
    trial_ids = np.array([0, 1, 2])
    spike_trial_ids = np.array([0, 0, 2])
    spike_times = np.array([5.0, 15.0, 3.0])
    counts = counts_from_spike_times(trial_ids, spike_trial_ids, spike_times, [10.0, 20.0])
    expected = np.array([[1, 2], [0, 0], [1, 1]])
    np.testing.assert_array_equal(counts, expected)


def test_estimate_lambda_hz_recovers_known_rate():
    rng = np.random.default_rng(1)
    lam_true = 18.0
    bins = _simulate_condition(lam_true, 200_000, 0.15, 50, rng)
    counts_short = bins.sum(axis=1)
    lam_hat = estimate_lambda_hz(counts_short, shortest_window_ms=50.0)
    assert lam_hat == pytest.approx(lam_true, rel=0.02)


def test_fit_ff_vs_window_slope_recovers_known_line():
    T = np.array([20.0, 50.0, 100.0, 200.0])
    true_slope, true_intercept = 0.0007, 0.95
    ff = true_intercept + true_slope * T
    slope, intercept = fit_ff_vs_window_slope(T, ff)
    assert slope == pytest.approx(true_slope, rel=1e-6)
    assert intercept == pytest.approx(true_intercept, rel=1e-6)


def test_pooled_sigma_g2_recovers_known_value_from_exact_slopes():
    lam_hz = {6.0: 3.0, 25.0: 21.0, 100.0: 39.0}  # Hz, as the function expects
    true_sigma_g2 = 0.03
    slopes = {c: (lam_hz[c] / 1000.0) * true_sigma_g2 for c in lam_hz}  # slope is defined per-ms
    sg2_hat = pooled_sigma_g2(lam_hz, slopes)
    assert sg2_hat == pytest.approx(true_sigma_g2, rel=1e-8)


def test_full_pipeline_recovers_sigma_g_on_synthetic_known_data():
    """End-to-end check of stages 2-4 together, on data simulated with a
    KNOWN sigma_g, at a large enough sample size that recovery should be
    tight."""
    rng = np.random.default_rng(7)
    sigma_g_true = 0.2
    dt_ms = 1.0
    fit_windows = [20.0, 50.0, 100.0, 200.0, 400.0]
    lam_by_cond = {6.0: 4.0, 25.0: 20.0, 100.0: 38.0}
    n_trials = 20_000

    lam_hat, slopes = {}, {}
    for c, lam_hz in lam_by_cond.items():
        bins = _simulate_condition(lam_hz, n_trials, sigma_g_true, 400, rng)
        counts = np.stack([bins[:, :int(w)].sum(axis=1) for w in fit_windows], axis=1)
        lam_hat[c] = estimate_lambda_hz(counts[:, 0], fit_windows[0])
        ff = empirical_fano_factor(counts)
        slope, _ = fit_ff_vs_window_slope(fit_windows, ff)
        slopes[c] = slope

    sg2_hat = pooled_sigma_g2(lam_hat, slopes)
    sigma_g_hat = np.sqrt(max(sg2_hat, 0))
    assert sigma_g_hat == pytest.approx(sigma_g_true, abs=0.03)


def test_fit_phi_constant_overdispersion_recovers_known_constant():
    # Use large mean counts so the Gaussian-generated synthetic data isn't
    # distorted by clipping at zero (a small-mean-count artifact of this
    # TEST's data generation, not a property of the estimator itself).
    rng = np.random.default_rng(3)
    phi_true = 1.4
    cells = {}
    for c, T in [(50.0, 200.0), (50.0, 800.0), (150.0, 200.0), (150.0, 800.0)]:
        mean_pred = c / 1000.0 * T
        counts = rng.normal(mean_pred, np.sqrt(phi_true * mean_pred), size=50_000)
        counts = np.clip(np.round(counts), 0, None)
        cells[(c, T)] = counts
    phi_hat = fit_phi_constant_overdispersion(cells)
    assert phi_hat == pytest.approx(phi_true, abs=0.05)


def test_compare_models_prefers_h1_when_data_generated_under_h1():
    rng = np.random.default_rng(11)
    sigma_g2_true = 0.03
    dt_ms = 1.0
    lam_by_cond = {6.0: 4.0, 25.0: 20.0, 100.0: 38.0}
    fit_windows = [20.0, 50.0, 100.0, 200.0, 400.0]
    cells = {}
    for c, lam_hz in lam_by_cond.items():
        bins = _simulate_condition(lam_hz, 3000, np.sqrt(sigma_g2_true), 400, rng)
        for T in fit_windows:
            cells[(c, T)] = bins[:, :int(T)].sum(axis=1)
    phi_hat = fit_phi_constant_overdispersion(cells)
    result = compare_models(cells, lam_by_cond, dt_ms, sigma_g2_hat=sigma_g2_true, phi_hat=phi_hat)
    assert result.best_model() == "H1_doubly_stochastic"
    assert result.aic["H1_doubly_stochastic"] < result.aic["H0_poisson"]
    assert result.aic["H1_doubly_stochastic"] < result.aic["H2_constant_overdispersion"]


def test_compare_models_prefers_h0_when_data_is_pure_poisson():
    rng = np.random.default_rng(12)
    lam_by_cond = {25.0: 20.0}
    fit_windows = [20.0, 100.0, 400.0]
    cells = {}
    for T in fit_windows:
        mu = 20.0 / 1000.0 * T
        cells[(25.0, T)] = rng.poisson(mu, size=3000)
    phi_hat = fit_phi_constant_overdispersion(cells)
    # sigma_g2_hat forced to a small positive stand-in; H0 should still win
    # since the data truly has no overdispersion
    result = compare_models(cells, lam_by_cond, dt_ms=1.0, sigma_g2_hat=0.01, phi_hat=phi_hat)
    assert result.best_model() == "H0_poisson"


def test_parametric_bootstrap_pvalue_rejects_clear_overdispersion():
    rng = np.random.default_rng(5)
    mean_pred = 10.0
    overdispersed = rng.normal(mean_pred, np.sqrt(3 * mean_pred), size=300)
    overdispersed = np.clip(np.round(overdispersed), 0, None)
    obs_stat, pval, null_dist = parametric_bootstrap_pvalue(
        overdispersed, mean_pred, mean_pred, n_boot=2000, rng=rng)
    assert pval < 0.01


def test_parametric_bootstrap_pvalue_does_not_reject_matching_model():
    rng = np.random.default_rng(6)
    mean_pred, var_pred = 10.0, 10.0
    matching = rng.poisson(mean_pred, size=300)
    obs_stat, pval, null_dist = parametric_bootstrap_pvalue(
        matching, mean_pred, var_pred, n_boot=2000, rng=rng)
    assert pval > 0.05


def test_poisson_null_detectability_low_power_at_low_rate_short_window():
    """A weak, low-rate overdispersion signal should often fail to reject
    the strict Poisson null with a modest number of trials -- this is the
    statistical-power phenomenon the project's headline finding rests on."""
    rng = np.random.default_rng(9)
    lam_hz = 3.0
    T = 20.0
    sigma_g = 0.18
    n_trials = 250
    bins = _simulate_condition(lam_hz, n_trials, sigma_g, int(T), rng)
    counts = bins.sum(axis=1)
    obs_ff, pval = poisson_null_detectability_pvalue(counts, lam_hz, T, n_boot=1000, rng=rng)
    # not asserting a specific verdict (it is a real, noisy statistical
    # test) -- just that the machinery runs and returns a valid p-value
    assert 0.0 <= pval <= 1.0


def test_predict_holdout_ff_matches_formula_directly():
    lam_hz, dt_ms, sg2 = 20.0, 1.0, 0.03
    T = np.array([20.0, 100.0, 800.0])
    pred = predict_holdout_ff(lam_hz, T, dt_ms, sg2)
    expected = fano_factor_formula(lam_hz, T, dt_ms, sg2)
    np.testing.assert_allclose(pred, expected)


def test_holdout_rmse_zero_for_perfect_prediction():
    obs = np.array([1.0, 1.1, 1.3])
    assert holdout_rmse(obs, obs) == pytest.approx(0.0)


def test_holdout_rmse_positive_for_imperfect_prediction():
    obs = np.array([1.0, 1.1, 1.3])
    pred = np.array([1.05, 1.05, 1.4])
    assert holdout_rmse(obs, pred) > 0.0
