"""Unit tests for models.py: block design, AR(1) noise generation, the
exact AR(1) covariance matrix, and multi-channel session simulation."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models import (  # noqa: E402
    N_CHANNELS, GRID_SHAPE, BlockDesignParams, NoiseParams,
    make_block_labels, expand_block_labels, ar1_noise, ar1_covariance_matrix,
    simulate_multichannel_session, build_design_matrix,
)


def test_make_block_labels_is_balanced():
    rng = np.random.default_rng(0)
    labels = make_block_labels(40, rng)
    assert len(labels) == 40
    assert np.sum(labels == 0) == 20
    assert np.sum(labels == 1) == 20


def test_make_block_labels_rejects_odd_n_blocks():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        make_block_labels(41, rng)


def test_expand_block_labels_shape_and_content():
    labels = np.array([0, 1, 0])
    expanded = expand_block_labels(labels, 3)
    np.testing.assert_array_equal(expanded, [0, 0, 0, 1, 1, 1, 0, 0, 0])


def test_ar1_noise_stationary_variance():
    """The stationary AR(1) parameterization should keep Var(e[t]) close
    to sigma^2 at every t, not just at t=0."""
    rng = np.random.default_rng(1)
    n_trials = 500
    sigma = 2.0
    params = NoiseParams(phi=0.7, sigma=sigma)
    traj = np.array([ar1_noise(n_trials, params, np.random.default_rng(s)) for s in range(3000)])
    empirical_var_by_t = traj.var(axis=0)
    # check late-time variance (well past any transient) matches sigma^2
    assert np.mean(empirical_var_by_t[100:]) == pytest.approx(sigma ** 2, rel=0.1)


def test_ar1_noise_autocorrelation_matches_phi():
    rng = np.random.default_rng(2)
    n_trials = 2000
    phi = 0.6
    params = NoiseParams(phi=phi, sigma=1.0)
    e = ar1_noise(n_trials, params, rng)
    e_centered = e - e.mean()
    lag1_corr = np.sum(e_centered[:-1] * e_centered[1:]) / np.sum(e_centered ** 2)
    assert lag1_corr == pytest.approx(phi, abs=0.08)


def test_ar1_noise_zero_phi_is_white_noise():
    rng = np.random.default_rng(3)
    params = NoiseParams(phi=0.0, sigma=1.0)
    e = ar1_noise(3000, params, rng)
    e_centered = e - e.mean()
    lag1_corr = np.sum(e_centered[:-1] * e_centered[1:]) / np.sum(e_centered ** 2)
    assert abs(lag1_corr) < 0.05


def test_ar1_covariance_matrix_matches_formula():
    Sigma = ar1_covariance_matrix(4, phi=0.5, sigma=2.0)
    expected = np.array([
        [4.0, 2.0, 1.0, 0.5],
        [2.0, 4.0, 2.0, 1.0],
        [1.0, 2.0, 4.0, 2.0],
        [0.5, 1.0, 2.0, 4.0],
    ])
    np.testing.assert_allclose(Sigma, expected)


def test_ar1_covariance_matrix_is_symmetric_positive_definite():
    Sigma = ar1_covariance_matrix(20, phi=0.6, sigma=1.0)
    np.testing.assert_allclose(Sigma, Sigma.T)
    eigvals = np.linalg.eigvalsh(Sigma)
    assert np.all(eigvals > 0)


def test_simulate_multichannel_session_shapes():
    rng = np.random.default_rng(4)
    design = BlockDesignParams(n_blocks=10, block_size=4)
    noise = NoiseParams(phi=0.5, sigma=1.0)
    session = simulate_multichannel_session(design, noise, np.array([0, 5]), 2.0, rng)
    assert session["Y"].shape == (design.n_trials, N_CHANNELS)
    assert len(session["condition"]) == design.n_trials
    assert len(session["block_labels"]) == design.n_blocks
    assert session["true_effect_by_channel"][0] == 2.0
    assert session["true_effect_by_channel"][5] == 2.0
    assert session["true_effect_by_channel"][1] == 0.0


def test_simulate_multichannel_session_zero_effect_zero_noise_is_flat():
    rng = np.random.default_rng(5)
    design = BlockDesignParams(n_blocks=6, block_size=2)
    noise = NoiseParams(phi=0.0, sigma=0.0)
    session = simulate_multichannel_session(design, noise, np.array([], dtype=int), 0.0, rng)
    np.testing.assert_allclose(session["Y"], 0.0, atol=1e-12)


def test_build_design_matrix_shape_and_intercept():
    condition = np.array([0, 1, 1, 0])
    X = build_design_matrix(condition)
    assert X.shape == (4, 2)
    np.testing.assert_allclose(X[:, 0], 1.0)
    np.testing.assert_allclose(X[:, 1], condition)


def test_grid_shape_matches_n_channels():
    assert GRID_SHAPE[0] * GRID_SHAPE[1] == N_CHANNELS
