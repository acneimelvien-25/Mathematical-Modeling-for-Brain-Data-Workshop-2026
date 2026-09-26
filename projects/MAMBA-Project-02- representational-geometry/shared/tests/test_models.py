"""Unit tests for models.py: the Gaussian-bump tuning model and the PCA
primitives."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models import (GaussianBumpPopulation, GaussianBumpPopulationParams,  # noqa: E402
                     fit_pca, pca_project, pca_reconstruct, variance_explained_ratio)


def _simple_population(seed=0):
    rng = np.random.default_rng(seed)
    M = 20
    params = GaussianBumpPopulationParams(
        preferred_xy=rng.uniform(-1.2, 1.2, size=(M, 2)),
        sigma=rng.uniform(0.4, 0.9, size=M),
        gain=rng.uniform(5, 15, size=M),
        baseline=rng.uniform(1, 3, size=M),
    )
    return GaussianBumpPopulation(params)


def test_mean_rate_peaks_at_preferred_location():
    """Each neuron's mean rate should be maximal exactly at its own
    preferred (x, y), since the Gaussian bump is centered there."""
    pop = _simple_population()
    for m in range(5):
        pref = pop.p.preferred_xy[m]
        r_at_pref = pop.mean_rate(pref[None, :])[0, m]
        # perturb slightly in a few directions; rate should be lower everywhere
        for dx, dy in [(0.1, 0), (-0.1, 0), (0, 0.1), (0, -0.1), (0.07, 0.07)]:
            r_nearby = pop.mean_rate((pref + np.array([dx, dy]))[None, :])[0, m]
            assert r_nearby < r_at_pref


def test_mean_rate_bounds():
    pop = _simple_population()
    xy = np.random.default_rng(1).uniform(-2, 2, size=(50, 2))
    rates = pop.mean_rate(xy)
    assert np.all(rates >= pop.p.baseline[None, :] - 1e-9)
    assert np.all(rates <= pop.p.baseline[None, :] + pop.p.gain[None, :] + 1e-9)


def test_sample_counts_are_nonnegative_integers_with_right_shape():
    pop = _simple_population()
    rng = np.random.default_rng(2)
    xy = np.array([[0.0, 0.0], [0.5, -0.5]])
    counts = pop.sample_counts(xy, n_trials=10, integration_time_s=1.0, rng=rng)
    assert counts.shape == (2, 10, pop.n_neurons)
    assert np.all(counts >= 0)
    assert np.issubdtype(counts.dtype, np.integer)


def test_sample_counts_mean_matches_rate_at_large_n():
    """A Poisson process's sample mean should converge to its rate
    parameter -- checked here with enough trials for a loose tolerance."""
    pop = _simple_population()
    rng = np.random.default_rng(3)
    xy = np.array([[0.2, 0.3]])
    n_trials = 20000
    counts = pop.sample_counts(xy, n_trials=n_trials, integration_time_s=1.0, rng=rng)
    empirical_mean = counts[0].mean(axis=0)
    true_rate = pop.mean_rate(xy)[0]
    # Poisson: std of the sample mean is sqrt(rate/n_trials); rates are a
    # few Hz, so this tolerance comfortably covers sampling noise
    assert np.max(np.abs(empirical_mean - true_rate)) < 0.5


def test_poisson_mean_variance_relationship():
    """A defining property of a Poisson process: variance should be
    approximately equal to the mean (checked per-neuron, one high-rate
    neuron, with enough trials for the estimate to be reasonably tight)."""
    pop = _simple_population()
    rng = np.random.default_rng(4)
    xy = np.array([[0.0, 0.0]])
    # pick whichever neuron has the highest baseline+gain for a clear signal
    m = np.argmax(pop.p.baseline + pop.p.gain)
    counts = pop.sample_counts(xy, n_trials=20000, integration_time_s=1.0, rng=rng)[0, :, m]
    mean_est, var_est = counts.mean(), counts.var()
    assert var_est == pytest.approx(mean_est, rel=0.15)


def test_pca_reconstruction_is_exact_with_all_components():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(30, 8))
    basis = fit_pca(X)
    scores = pca_project(basis, X, k=8)
    X_hat = pca_reconstruct(basis, scores, k=8)
    np.testing.assert_allclose(X_hat, X, atol=1e-8)


def test_pca_components_are_orthonormal():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(40, 10))
    basis = fit_pca(X)
    gram = basis.components.T @ basis.components
    np.testing.assert_allclose(gram, np.eye(10), atol=1e-8)


def test_pca_variance_explained_sums_to_one_and_is_decreasing():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(50, 12))
    basis = fit_pca(X)
    ratio = variance_explained_ratio(basis)
    assert ratio.sum() == pytest.approx(1.0, abs=1e-8)
    assert np.all(np.diff(ratio) <= 1e-12)  # non-increasing


def test_pca_matches_independent_svd_computation():
    """The covariance-eigendecomposition-based fit_pca should agree with
    an independent SVD-based computation of the same PCA, up to sign
    flips of each component."""
    rng = np.random.default_rng(8)
    X = rng.normal(size=(60, 15)) @ rng.normal(size=(15, 15))  # correlated features
    basis = fit_pca(X)

    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    eigvals_svd = (S ** 2) / (X.shape[0] - 1)

    np.testing.assert_allclose(basis.variance, eigvals_svd, atol=1e-6)
    for i in range(15):
        # components can differ by an overall sign
        agree_pos = np.allclose(basis.components[:, i], Vt[i], atol=1e-6)
        agree_neg = np.allclose(basis.components[:, i], -Vt[i], atol=1e-6)
        assert agree_pos or agree_neg


def test_low_rank_data_needs_few_components():
    """A sanity check on fit_pca's core promise: data with only 2 true
    degrees of freedom (embedded linearly into a higher-dimensional space)
    should have ~100% of its variance captured by 2 components."""
    rng = np.random.default_rng(9)
    latent = rng.normal(size=(100, 2))
    embedding = rng.normal(size=(2, 20))
    X = latent @ embedding + rng.normal(scale=1e-6, size=(100, 20))
    basis = fit_pca(X)
    ratio = variance_explained_ratio(basis)
    assert np.cumsum(ratio)[1] > 0.999
