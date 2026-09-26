"""Unit tests for analysis.py, run against synthetic data with KNOWN
parameters (not the committed dataset)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models import (  # noqa: E402
    BlockDesignParams, NoiseParams, make_block_labels, expand_block_labels,
    ar1_covariance_matrix, simulate_multichannel_session, build_design_matrix,
)
from analysis import (  # noqa: E402
    fit_mass_univariate, naive_pvalues, sandwich_variance, naive_variance,
    bonferroni_significant, benjamini_hochberg_significant,
    permutation_null_max_stat, permutation_pvalue_single_channel,
    max_stat_significant, replication_summary,
)


def test_fit_mass_univariate_recovers_known_effect_zero_noise():
    rng = np.random.default_rng(0)
    design = BlockDesignParams(n_blocks=10, block_size=4)
    noise = NoiseParams(phi=0.0, sigma=0.0)
    session = simulate_multichannel_session(design, noise, np.array([3, 7]), 2.5, rng)
    X = build_design_matrix(session["condition"])
    fit = fit_mass_univariate(X, session["Y"])
    assert fit.beta[3] == pytest.approx(2.5, abs=1e-10)
    assert fit.beta[7] == pytest.approx(2.5, abs=1e-10)
    assert fit.beta[0] == pytest.approx(0.0, abs=1e-10)


def test_naive_pvalues_small_for_large_effect_large_n():
    rng = np.random.default_rng(1)
    design = BlockDesignParams(n_blocks=40, block_size=5)
    noise = NoiseParams(phi=0.0, sigma=1.0)  # no autocorrelation -> naive test is valid here
    session = simulate_multichannel_session(design, noise, np.array([0]), 3.0, rng)
    X = build_design_matrix(session["condition"])
    fit = fit_mass_univariate(X, session["Y"])
    pvals = naive_pvalues(fit)
    assert pvals[0] < 0.001


def test_sandwich_variance_equals_naive_variance_when_sigma_is_spherical():
    rng = np.random.default_rng(2)
    n = 40
    X = np.column_stack([np.ones(n), rng.integers(0, 2, size=n)])
    sigma2 = 2.0
    Sigma = sigma2 * np.eye(n)
    sw = sandwich_variance(X, Sigma)
    nv = naive_variance(X, sigma2)
    assert sw == pytest.approx(nv, rel=1e-10)


def test_sandwich_variance_hand_example():
    """The exact hand-checkable example from the handout Section 3.3:
    an AAAABBBB (contrast-coded) design, n=8, phi=0.5 (an illustrative
    value, deliberately different from this project's true phi=0.6, so
    working the hand example does not incidentally reveal it), sigma=1."""
    X = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=float).reshape(-1, 1)
    Sigma = ar1_covariance_matrix(8, phi=0.5, sigma=1.0)
    sw = sandwich_variance(X, Sigma, contrast_index=0)
    nv = naive_variance(X, sigma2=1.0, contrast_index=0)
    assert sw == pytest.approx(0.2029, abs=1e-3)
    assert nv == pytest.approx(0.125, abs=1e-10)
    assert sw > nv  # naive UNDERESTIMATES the true variance here


def test_sandwich_variance_averages_higher_than_naive_for_actual_block_design():
    """The core cross-check required by the project workflow: verify,
    via averaging over many random block-label draws of the STUDY's
    actual design, that the naive formula is anti-conservative on
    average for this specific (blocked, autocorrelated) design."""
    rng = np.random.default_rng(3)
    n_blocks, block_size = 20, 4
    n = n_blocks * block_size
    phi, sigma = 0.6, 1.0
    Sigma = ar1_covariance_matrix(n, phi, sigma)
    ratios = []
    for _ in range(200):
        labels = make_block_labels(n_blocks, rng)
        cond = expand_block_labels(labels, block_size)
        X = (2 * cond - 1.0).reshape(-1, 1).astype(float)  # +-1 contrast coding
        sw = sandwich_variance(X, Sigma, contrast_index=0)
        nv = naive_variance(X, sigma2=sigma ** 2, contrast_index=0)
        ratios.append(sw / nv)
    assert np.mean(ratios) > 1.5  # substantially anti-conservative on average


def test_bonferroni_significant_matches_hand_calculation():
    pvals = np.array([0.001, 0.02, 0.03, 0.04])
    sig = bonferroni_significant(pvals, alpha=0.05)
    # alpha/n = 0.0125 -> only the first p-value survives
    np.testing.assert_array_equal(sig, [0])


def test_benjamini_hochberg_matches_hand_calculation():
    """The exact worked example from the handout Section 3.5:
    p = [0.001, 0.008, 0.02, 0.04, 0.12], alpha = 0.05 -> reject the
    first four (thresholds 0.01, 0.02, 0.03, 0.04, 0.05)."""
    pvals = np.array([0.001, 0.008, 0.02, 0.04, 0.12])
    sig = benjamini_hochberg_significant(pvals, alpha=0.05)
    np.testing.assert_array_equal(np.sort(sig), [0, 1, 2, 3])


def test_benjamini_hochberg_is_at_least_as_liberal_as_bonferroni():
    rng = np.random.default_rng(4)
    pvals = rng.uniform(0, 0.1, size=50)
    bonf = set(bonferroni_significant(pvals).tolist())
    fdr = set(benjamini_hochberg_significant(pvals).tolist())
    assert bonf.issubset(fdr)


def test_benjamini_hochberg_empty_when_nothing_significant():
    pvals = np.array([0.5, 0.6, 0.8, 0.9])
    sig = benjamini_hochberg_significant(pvals, alpha=0.05)
    assert len(sig) == 0


def test_permutation_null_max_stat_controls_type1_error_under_global_null():
    """Under a global null (no true effect anywhere), the max-stat
    permutation threshold should reject at close to the nominal alpha
    rate across independent simulated null datasets."""
    rng = np.random.default_rng(5)
    design = BlockDesignParams(n_blocks=20, block_size=4)
    noise = NoiseParams(phi=0.6, sigma=1.0)
    n_channels_test = 10
    n_sims = 60
    rejections = 0
    for _ in range(n_sims):
        session = simulate_multichannel_session(design, noise, np.array([], dtype=int), 0.0, rng)
        Y = session["Y"][:, :n_channels_test]
        X = build_design_matrix(session["condition"])
        fit = fit_mass_univariate(X, Y)
        max_null = permutation_null_max_stat(session["block_labels"], design.block_size, Y,
                                              n_perm=200, rng=rng)
        sig = max_stat_significant(fit, max_null, alpha=0.05)
        if len(sig) > 0:
            rejections += 1
    # family-wise error rate should be roughly 0.05, not dramatically higher
    assert rejections / n_sims < 0.20


def test_permutation_pvalue_single_channel_detects_strong_effect():
    rng = np.random.default_rng(6)
    design = BlockDesignParams(n_blocks=20, block_size=4)
    noise = NoiseParams(phi=0.3, sigma=1.0)
    session = simulate_multichannel_session(design, noise, np.array([0]), 3.0, rng)
    t_obs, pval = permutation_pvalue_single_channel(session["block_labels"], design.block_size,
                                                     session["Y"][:, 0], n_perm=500, rng=rng)
    assert pval < 0.05


def test_replication_summary_counts():
    disc = np.array([1, 2, 3, 4])
    repl = np.array([2, 3, 5])
    summary = replication_summary(disc, repl)
    assert summary["n_discovery_significant"] == 4
    assert summary["n_replication_significant"] == 3
    assert summary["n_replicated"] == 2
    assert summary["n_discovery_only"] == 2
    assert summary["n_replication_only"] == 1
