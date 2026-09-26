"""
analysis.py
===========
Reference analysis pipeline for MAMBA Project 07: Neuroimaging-Style GLM.

Pipeline stages (see student handout Sections 2 and 5):

1. Mass-univariate OLS: fit the SAME design matrix to every channel's
   response column, extract the condition-effect coefficient, its naive
   (i.i.d.-assumption) standard error, and a t-statistic, for every
   channel at once.
2. The "sandwich" variance formula for OLS under an arbitrary (e.g.
   autocorrelated) error covariance matrix, and a comparison against the
   naive formula on the study's actual design.
3. Classical multiple-comparisons corrections (Bonferroni, Benjamini-
   Hochberg FDR) applied to the naive per-channel p-values.
4. A block-label permutation test: build an empirical null distribution
   for each channel's test statistic by permuting which blocks receive
   which condition label (preserving the noise's own temporal/blocked
   structure exactly, whatever it is), and a max-statistic extension
   that additionally controls the family-wise error rate across all
   channels at once, without assuming any particular cross-channel
   independence structure.
5. Discovery/replication comparison utilities.

Every public function takes plain arrays in and returns plain arrays/
dicts/floats out, so it can be unit-tested against data with KNOWN
parameters independently of the committed dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
from scipy import stats

from models import expand_block_labels


# ---------------------------------------------------------------------------
# Stage 1: mass-univariate OLS
# ---------------------------------------------------------------------------

@dataclass
class MassUnivariateFit:
    beta: np.ndarray        # (n_channels,) condition-effect coefficient
    se_naive: np.ndarray    # (n_channels,) naive (i.i.d.-assumption) SE
    t_naive: np.ndarray     # (n_channels,) t = beta / se_naive
    residuals: np.ndarray   # (n_trials, n_channels)
    dof: int
    XtX_inv: np.ndarray     # (p, p), shared across channels


def fit_mass_univariate(X: np.ndarray, Y: np.ndarray) -> MassUnivariateFit:
    """Fit the SAME design matrix X to every column of Y (one OLS fit per
    channel), using the standard (i.i.d.-noise-assumption) OLS variance
    formula for each channel's standard error. Assumes the condition
    effect is the LAST column of X (index -1 / index 1 for a 2-column X).
    """
    n, p = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta_all = XtX_inv @ X.T @ Y          # (p, n_channels)
    fitted = X @ beta_all
    residuals = Y - fitted
    dof = n - p
    sigma2_hat = np.sum(residuals ** 2, axis=0) / dof   # (n_channels,)
    se_all = np.sqrt(XtX_inv[-1, -1] * sigma2_hat)       # (n_channels,)
    beta_effect = beta_all[-1, :]
    t_stat = beta_effect / se_all
    return MassUnivariateFit(beta=beta_effect, se_naive=se_all, t_naive=t_stat,
                              residuals=residuals, dof=dof, XtX_inv=XtX_inv)


def naive_pvalues(fit: MassUnivariateFit) -> np.ndarray:
    return 2 * (1 - stats.t.cdf(np.abs(fit.t_naive), fit.dof))


# ---------------------------------------------------------------------------
# Stage 2: the sandwich variance formula
# ---------------------------------------------------------------------------

def sandwich_variance(X: np.ndarray, Sigma: np.ndarray, contrast_index: int = -1) -> float:
    """Exact variance of a single OLS coefficient (default: the LAST
    column of X) under an arbitrary error covariance matrix Sigma (not
    necessarily sigma^2 * I):

        Var(beta_hat) = (X'X)^-1 X' Sigma X (X'X)^-1

    Returns only the [contrast_index, contrast_index] diagonal entry of
    this p x p matrix (the variance of one specific coefficient).
    """
    XtX_inv = np.linalg.inv(X.T @ X)
    full_cov = XtX_inv @ X.T @ Sigma @ X @ XtX_inv
    return float(full_cov[contrast_index, contrast_index])


def naive_variance(X: np.ndarray, sigma2: float, contrast_index: int = -1) -> float:
    """The (potentially wrong) i.i.d.-noise-assumption variance formula:
    sigma^2 * (X'X)^-1, diagonal entry for one coefficient.
    """
    XtX_inv = np.linalg.inv(X.T @ X)
    return float(sigma2 * XtX_inv[contrast_index, contrast_index])


# ---------------------------------------------------------------------------
# Stage 3: classical multiple-comparisons corrections
# ---------------------------------------------------------------------------

def bonferroni_significant(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Indices of channels significant after a Bonferroni correction
    (reject if p < alpha / n_tests)."""
    n = len(pvalues)
    return np.where(pvalues < alpha / n)[0]


def benjamini_hochberg_significant(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Indices of channels significant after Benjamini-Hochberg FDR
    control at level alpha: sort p-values ascending, find the largest
    rank k such that p_(k) <= alpha * k / n, and reject all hypotheses
    with rank <= k.
    """
    n = len(pvalues)
    order = np.argsort(pvalues)
    sorted_p = pvalues[order]
    thresholds = alpha * (np.arange(1, n + 1) / n)
    below = sorted_p <= thresholds
    if not np.any(below):
        return np.array([], dtype=int)
    k = np.max(np.where(below)[0])
    return np.sort(order[:k + 1])


# ---------------------------------------------------------------------------
# Stage 4: block-label permutation testing
# ---------------------------------------------------------------------------

def permutation_null_max_stat(block_labels: np.ndarray, block_size: int, Y: np.ndarray,
                               n_perm: int, rng: np.random.Generator) -> np.ndarray:
    """Build a null distribution of the MAXIMUM (across all channels) of
    |t| under random re-labelings of which blocks receive which
    condition. Each re-labeling preserves the block structure exactly
    (same block boundaries, same noise realizations, same number of
    blocks per condition) -- only WHICH blocks got which label is
    shuffled -- so the noise's own autocorrelation and any cross-channel
    structure are automatically preserved under every permutation,
    without the analyst needing to specify or estimate them.

    Returns an array of length n_perm of maximum-|t|-across-channels values.
    """
    n_blocks = len(block_labels)
    max_null = np.empty(n_perm)
    for i in range(n_perm):
        perm_labels = rng.permutation(block_labels)
        cond_perm = expand_block_labels(perm_labels, block_size)
        X_perm = np.column_stack([np.ones(len(cond_perm)), cond_perm])
        fit_perm = fit_mass_univariate(X_perm, Y)
        max_null[i] = np.max(np.abs(fit_perm.t_naive))
    return max_null


def permutation_pvalue_single_channel(block_labels: np.ndarray, block_size: int, y: np.ndarray,
                                       n_perm: int, rng: np.random.Generator) -> tuple:
    """Single-channel block-permutation p-value (no multiple-comparisons
    correction) -- for comparison against the naive formula's p-value on
    one channel at a time.
    """
    n_trials = len(y)
    cond_obs = expand_block_labels(block_labels, block_size)
    X_obs = np.column_stack([np.ones(n_trials), cond_obs])
    fit_obs = fit_mass_univariate(X_obs, y[:, None])
    t_obs = fit_obs.t_naive[0]

    n_blocks = len(block_labels)
    null_t = np.empty(n_perm)
    for i in range(n_perm):
        perm_labels = rng.permutation(block_labels)
        cond_p = expand_block_labels(perm_labels, block_size)
        X_p = np.column_stack([np.ones(n_trials), cond_p])
        fit_p = fit_mass_univariate(X_p, y[:, None])
        null_t[i] = fit_p.t_naive[0]
    pval = (np.sum(np.abs(null_t) >= np.abs(t_obs)) + 1) / (n_perm + 1)
    return float(t_obs), float(pval)


def max_stat_significant(fit: MassUnivariateFit, max_null: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Indices of channels whose observed |t| exceeds the (1 - alpha)
    quantile of the max-statistic permutation null -- family-wise error
    rate control across all channels simultaneously.
    """
    threshold = np.percentile(max_null, 100 * (1 - alpha))
    return np.where(np.abs(fit.t_naive) > threshold)[0]


# ---------------------------------------------------------------------------
# Stage 5: discovery/replication comparison
# ---------------------------------------------------------------------------

def replication_summary(discovery_sig: np.ndarray, replication_sig: np.ndarray) -> Dict[str, int]:
    disc = set(discovery_sig.tolist())
    repl = set(replication_sig.tolist())
    return {
        "n_discovery_significant": len(disc),
        "n_replication_significant": len(repl),
        "n_replicated": len(disc & repl),
        "n_discovery_only": len(disc - repl),
        "n_replication_only": len(repl - disc),
    }
