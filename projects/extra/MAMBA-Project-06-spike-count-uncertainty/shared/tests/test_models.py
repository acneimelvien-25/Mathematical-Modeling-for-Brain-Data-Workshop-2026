"""Unit tests for models.py: the tuning curve, trial-gain sampling, the
binned spike generator, and (most importantly) the exact Fano-factor
formula, cross-checked against brute-force Monte Carlo simulation."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models import (  # noqa: E402
    TuningCurveParams,
    GainNoiseParams,
    naka_rushton,
    sample_trial_gains,
    generate_binned_spike_counts,
    fano_factor_formula,
    mean_count_formula,
)


def test_naka_rushton_at_zero_contrast_is_baseline():
    p = TuningCurveParams(R0=2.0, Rmax=38.0, c50=25.0, n=2.5)
    assert naka_rushton(np.array([0.0]), p)[0] == pytest.approx(2.0)


def test_naka_rushton_at_c50_is_baseline_plus_half_range():
    p = TuningCurveParams(R0=2.0, Rmax=38.0, c50=25.0, n=2.5)
    assert naka_rushton(np.array([25.0]), p)[0] == pytest.approx(2.0 + 38.0 / 2)


def test_naka_rushton_saturates_toward_R0_plus_Rmax():
    p = TuningCurveParams(R0=2.0, Rmax=38.0, c50=25.0, n=2.5)
    val = naka_rushton(np.array([1e6]), p)[0]
    assert val == pytest.approx(2.0 + 38.0, rel=1e-4)


def test_naka_rushton_monotonically_increasing():
    p = TuningCurveParams(R0=2.0, Rmax=38.0, c50=25.0, n=2.5)
    cs = np.array([1.0, 5.0, 10.0, 25.0, 50.0, 100.0])
    vals = naka_rushton(cs, p)
    assert np.all(np.diff(vals) > 0)


def test_sample_trial_gains_mean_and_no_nonpositive():
    rng = np.random.default_rng(0)
    g = sample_trial_gains(200_000, GainNoiseParams(sigma_g=0.18), rng)
    assert np.all(g > 0)
    assert g.mean() == pytest.approx(1.0, abs=0.01)
    assert g.std() == pytest.approx(0.18, abs=0.01)


def test_generate_binned_spike_counts_shape_and_binary():
    rng = np.random.default_rng(1)
    bins = generate_binned_spike_counts(
        lam_hz=20.0, n_trials=50, dt_ms=1.0, n_bins_max=800,
        gain_params=GainNoiseParams(sigma_g=0.1), rng=rng,
    )
    assert bins.shape == (50, 800)
    assert set(np.unique(bins)).issubset({0, 1})


def test_mean_count_formula_matches_simulation():
    """E[N(T)] = lambda*T should hold regardless of sigma_g (mean is
    insensitive to gain variability, since E[g] = 1)."""
    rng = np.random.default_rng(2)
    lam_hz = 15.0
    T = 300.0
    n_trials = 400_000
    bins = generate_binned_spike_counts(
        lam_hz=lam_hz, n_trials=n_trials, dt_ms=1.0, n_bins_max=int(T),
        gain_params=GainNoiseParams(sigma_g=0.2), rng=rng,
    )
    counts = bins.sum(axis=1)
    predicted_mean = mean_count_formula(lam_hz, T)
    assert counts.mean() == pytest.approx(predicted_mean, rel=0.01)


@pytest.mark.parametrize("lam_hz,sigma_g,T", [
    (10.0, 0.15, 100.0),
    (10.0, 0.15, 500.0),
    (30.0, 0.25, 200.0),
    (5.0, 0.0, 400.0),   # zero gain variability: pure sub-Poisson binomial correction only
])
def test_fano_factor_formula_matches_monte_carlo(lam_hz, sigma_g, T):
    """The core cross-check required by the project workflow: verify the
    hand-derived closed-form Fano factor formula (law of total variance)
    against a large brute-force Monte Carlo simulation, independently of
    any fitting code."""
    rng = np.random.default_rng(hash((lam_hz, sigma_g, T)) % (2**32))
    n_trials = 300_000
    n_bins = int(T)
    bins = generate_binned_spike_counts(
        lam_hz=lam_hz, n_trials=n_trials, dt_ms=1.0, n_bins_max=n_bins,
        gain_params=GainNoiseParams(sigma_g=sigma_g), rng=rng,
    )
    counts = bins.sum(axis=1)
    empirical_ff = counts.var(ddof=1) / counts.mean()
    predicted_ff = fano_factor_formula(lam_hz, np.array([T]), dt_ms=1.0, sigma_g2=sigma_g ** 2)[0]
    assert empirical_ff == pytest.approx(predicted_ff, abs=0.02)


def test_fano_factor_formula_reduces_to_poisson_when_sigma_g_zero_and_dt_small():
    """As dt -> 0 with sigma_g2 = 0, FF(T) -> 1 (pure Poisson limit)."""
    ff = fano_factor_formula(lam_hz=10.0, T_ms=np.array([500.0]), dt_ms=1e-6, sigma_g2=0.0)[0]
    assert ff == pytest.approx(1.0, abs=1e-4)


def test_fano_factor_formula_increases_with_window_when_sigma_g_positive():
    T = np.array([20.0, 100.0, 400.0, 800.0])
    ff = fano_factor_formula(lam_hz=20.0, T_ms=T, dt_ms=1.0, sigma_g2=0.03)
    assert np.all(np.diff(ff) > 0)


def test_fano_factor_formula_reduces_to_pure_binomial_correction_when_sigma_g_zero():
    """With sigma_g2 = 0 (no gain fluctuation at all), the formula should
    reduce EXACTLY to FF(T) = 1 - lambda*dt (the ordinary sub-Poisson
    correction from counting a discrete-time Binomial rather than a
    continuous-time Poisson process), independent of T."""
    lam_hz, dt_ms = 24.0, 1.0
    lam_per_ms = lam_hz / 1000.0
    T = np.array([20.0, 100.0, 800.0])
    ff = fano_factor_formula(lam_hz, T, dt_ms, sigma_g2=0.0)
    expected = 1.0 - lam_per_ms * dt_ms
    assert np.allclose(ff, expected)
    # and it must NOT depend on T when sigma_g2 = 0
    assert np.allclose(ff, ff[0])
