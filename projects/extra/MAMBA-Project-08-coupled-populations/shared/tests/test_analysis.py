"""Unit tests for analysis.py, run against synthetic data with KNOWN
parameters (not the committed dataset)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models import RingParams, simulate_directed_ring, onset_angular_frequency  # noqa: E402
from analysis import (  # noqa: E402
    fit_local_linear_model, pooled_gain_estimate, dominant_angular_frequency,
    oscillation_amplitude, instantaneous_phase, mean_phase_lag,
    predicted_bifurcation_summary,
)


def test_fit_local_linear_model_recovers_known_coefficients_low_noise():
    rng = np.random.default_rng(0)
    params = RingParams(n_nodes=5, tau=1.0)
    w_true = -0.7
    x = simulate_directed_ring(w_true, params, T=2000.0, dt=0.01, sigma=0.03, rng=rng)
    x_sub = x[::5]
    dt_eff = 0.01 * 5
    fit = fit_local_linear_model(x_sub, dt_eff)
    assert fit.self_decay == pytest.approx(-1.0, abs=0.05)
    assert fit.coupling_coef == pytest.approx(w_true, abs=0.05)


def test_pooled_gain_estimate_recovers_known_gain_from_exact_coefficients():
    w_values = [-0.3, -0.5, -0.8, -1.0]
    g_true = 1.3
    coupling_coefs = [w * g_true for w in w_values]
    g_hat = pooled_gain_estimate(w_values, coupling_coefs)
    assert g_hat == pytest.approx(g_true, rel=1e-10)


def test_pooled_gain_estimate_robust_to_small_per_condition_noise():
    rng = np.random.default_rng(1)
    w_values = [-0.4, -0.6, -0.8, -1.0, -1.1]
    g_true = 0.9
    coupling_coefs = [w * g_true + rng.normal(0, 0.01) for w in w_values]
    g_hat = pooled_gain_estimate(w_values, coupling_coefs)
    assert g_hat == pytest.approx(g_true, abs=0.03)


def test_dominant_angular_frequency_matches_known_sine_wave():
    # duration chosen long enough for the FFT's frequency resolution
    # (~1/T) to resolve true_omega to within the test's tolerance
    dt = 0.05
    t = np.arange(0, 600, dt)
    true_omega = 0.7265
    x = np.sin(true_omega * t)
    freq = dominant_angular_frequency(x, dt)
    assert freq == pytest.approx(true_omega, abs=0.02)


def test_oscillation_amplitude_matches_known_sine_wave():
    t = np.arange(0, 50, 0.01)
    x = 2.5 * np.sin(0.5 * t)
    amp = oscillation_amplitude(x)
    assert amp == pytest.approx(5.0, abs=0.05)


def test_oscillation_amplitude_near_zero_for_decayed_signal():
    x = np.exp(-np.arange(0, 20, 0.01)) * 1e-8
    amp = oscillation_amplitude(x)
    assert amp < 1e-6


def test_instantaneous_phase_matches_known_sine_wave():
    dt = 0.01
    t = np.arange(0, 50, dt)
    x = np.sin(0.6 * t)
    phase = instantaneous_phase(x)
    # away from edges, phase should progress at rate ~0.6 rad per unit time
    mid = slice(len(t) // 4, 3 * len(t) // 4)
    unwrapped = np.unwrap(phase[mid])
    rate = np.polyfit(t[mid], unwrapped, 1)[0]
    assert rate == pytest.approx(0.6, abs=0.02)


def test_mean_phase_lag_matches_known_traveling_wave():
    """Construct an exact synthetic traveling wave with a KNOWN phase lag
    between adjacent nodes and confirm mean_phase_lag recovers it."""
    n_nodes = 5
    true_lag_rad = 2 * np.pi * 2 / 5  # k=2 mode, ~144 degrees
    dt = 0.01
    t = np.arange(0, 200, dt)
    omega = 0.7265
    x = np.zeros((len(t), n_nodes))
    for i in range(n_nodes):
        x[:, i] = np.sin(omega * t - i * true_lag_rad)
    lags = mean_phase_lag(x)
    # every node's lag relative to its upstream neighbor should match true_lag_rad
    # (mod 2*pi, allow either sign convention)
    for lag in lags:
        agreement = min(abs(lag - true_lag_rad), abs(lag + true_lag_rad),
                         abs(abs(lag) - true_lag_rad))
        assert agreement < 0.05


def test_predicted_bifurcation_summary_matches_hand_values_N5():
    summary = predicted_bifurcation_summary(g_hat=1.0, n_nodes=5, tau=1.0, mode_k=2)
    assert summary["critical_w"] == pytest.approx(-1.23607, abs=1e-4)
    assert summary["onset_angular_frequency"] == pytest.approx(0.72654, abs=1e-4)
    assert summary["phase_lag_deg"] == pytest.approx(144.0, abs=1e-8)
