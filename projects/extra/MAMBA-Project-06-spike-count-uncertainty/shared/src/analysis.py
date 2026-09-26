"""
analysis.py
===========
Reference analysis pipeline for MAMBA Project 05: Spike-Count Uncertainty.

Pipeline stages (see student handout Sections 2 and 5 for the full task
description; this module implements one complete reference path through
it):

1. Load spike times / trial protocol; build a (trial x window) count
   matrix per condition from the raw spike times (Section 2).
2. Estimate each condition's mean rate lambda_hat(c) from the SHORTEST
   window only (Section 4.1) -- unbiased regardless of gain variability,
   since E[N(T)] = lambda*T exactly for any sigma_g.
3. For each FIT condition, regress the empirical Fano factor FF(T) against
   T (over FIT windows only) to get a per-condition slope and intercept
   (Section 4.2).
4. Pool the per-condition slopes across FIT conditions via a second,
   through-the-origin regression against lambda_hat(c) to obtain the
   single shared sigma_g^2 estimate (Section 4.3).
5. Model comparison: fit and compare three candidate explanations of the
   count variance structure (H0 pure Poisson, H1 doubly stochastic, H2
   constant overdispersion) via a Gaussian quasi-likelihood AIC (Section
   5).
6. Simulation-based goodness-of-fit check of the winning model against
   the real data via a parametric bootstrap (Section 5.3).
7. Per-cell detectability: for every (condition, window) cell, a
   parametric-bootstrap p-value against the strict H0 null (Section 6).
8. Holdout validation: use ONLY the FIT-derived sigma_g2_hat plus each
   holdout condition's own lambda_hat to predict FF(T), and compare to
   observed FF(T) on holdout conditions and/or the holdout window
   (Section 5.4).

Every public function here takes plain arrays/dicts in and returns plain
arrays/dicts/floats out, so it can be unit-tested against data with KNOWN
parameters (see tests/test_analysis.py) independently of the committed
dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from models import fano_factor_formula, mean_count_formula


# ---------------------------------------------------------------------------
# Stage 1: build count matrices from raw spike times
# ---------------------------------------------------------------------------

def counts_from_spike_times(trial_ids: np.ndarray, spike_trial_ids: np.ndarray,
                             spike_times_ms: np.ndarray, windows_ms: Sequence[float]) -> np.ndarray:
    """Build an (n_trials, n_windows) integer count matrix.

    ``trial_ids`` is the ordered list of trial ids for one condition (so
    trials with zero spikes are still represented). ``spike_trial_ids``
    and ``spike_times_ms`` are the (possibly much longer) full spike
    table, already filtered to this condition or not -- rows for other
    trial ids are ignored automatically.
    """
    windows_ms = np.asarray(windows_ms, dtype=float)
    n_trials = len(trial_ids)
    n_windows = len(windows_ms)
    counts = np.zeros((n_trials, n_windows), dtype=int)
    trial_index = {tid: i for i, tid in enumerate(trial_ids)}
    for tid, t in zip(spike_trial_ids, spike_times_ms):
        i = trial_index.get(int(tid))
        if i is None:
            continue
        # increment every window whose duration covers this spike time
        counts[i, :] += (t < windows_ms)
    return counts


# ---------------------------------------------------------------------------
# Stage 2: rate estimation
# ---------------------------------------------------------------------------

def estimate_lambda_hz(counts_shortest_window: np.ndarray, shortest_window_ms: float) -> float:
    """lambda_hat (Hz) = mean count at the shortest available window,
    divided by that window's duration, converted from per-ms to per-s.
    Unbiased for lambda regardless of sigma_g, since E[N(T)] = lambda*T
    exactly (see models.mean_count_formula).
    """
    mean_count = float(np.mean(counts_shortest_window))
    lam_per_ms = mean_count / shortest_window_ms
    return lam_per_ms * 1000.0


# ---------------------------------------------------------------------------
# Stage 3-4: Fano-factor regression and pooled sigma_g^2 estimate
# ---------------------------------------------------------------------------

def empirical_fano_factor(counts: np.ndarray) -> np.ndarray:
    """Sample Fano factor (unbiased sample variance / sample mean) for
    each column (window) of a (n_trials, n_windows) count matrix.
    """
    means = counts.mean(axis=0)
    variances = counts.var(axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return variances / means


def fit_ff_vs_window_slope(windows_ms: Sequence[float], ff: Sequence[float]) -> tuple:
    """Ordinary least squares fit of FF(T) = intercept + slope * T.
    Returns (slope, intercept).
    """
    T = np.asarray(windows_ms, dtype=float)
    y = np.asarray(ff, dtype=float)
    A = np.vstack([T, np.ones_like(T)]).T
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(sol[0]), float(sol[1])


def pooled_sigma_g2(lambda_hz_by_condition: Dict[float, float],
                     slope_by_condition: Dict[float, float]) -> float:
    """Second-stage, through-the-origin OLS of slope_c against
    lambda_hat(c) (both in the SAME units -- lambda in spikes/ms, to match
    how slope was produced against T in ms): the model predicts
    slope_c = lambda(c) * sigma_g^2 exactly, i.e. a line through the
    origin with slope sigma_g^2. Fitting through the origin (rather than
    with a free intercept) uses the theoretical constraint directly and
    is the appropriate regression when the model specifies the intercept.
    """
    conditions = sorted(lambda_hz_by_condition.keys())
    x = np.array([lambda_hz_by_condition[c] / 1000.0 for c in conditions])  # spikes/ms
    y = np.array([slope_by_condition[c] for c in conditions])
    return float(np.sum(x * y) / np.sum(x * x))


# ---------------------------------------------------------------------------
# Stage 5: model comparison (H0 / H1 / H2) via Gaussian quasi-likelihood AIC
# ---------------------------------------------------------------------------

def _quasi_loglik(counts: np.ndarray, mean_pred: float, var_pred: float) -> float:
    var_pred = max(float(var_pred), 1e-6)
    resid2 = (counts - mean_pred) ** 2
    ll = -0.5 * np.log(2 * np.pi * var_pred) - 0.5 * resid2 / var_pred
    return float(np.sum(ll))


@dataclass
class ModelComparisonResult:
    loglik: Dict[str, float] = field(default_factory=dict)
    n_params: Dict[str, int] = field(default_factory=dict)
    aic: Dict[str, float] = field(default_factory=dict)

    def best_model(self) -> str:
        return min(self.aic, key=self.aic.get)


def compare_models(counts_by_condition_window: Dict[tuple, np.ndarray],
                    lambda_hz_by_condition: Dict[float, float],
                    dt_ms: float,
                    sigma_g2_hat: float,
                    phi_hat: float) -> ModelComparisonResult:
    """Compare three candidate explanations of the count variance
    structure using a Gaussian quasi-likelihood (a large-count Normal
    approximation to the true count distribution; justified here because
    the smallest cells still have several trials per expected spike, and
    is what makes an apples-to-apples AIC comparison possible across
    models whose EXACT likelihoods are not all in closed form).

    ``counts_by_condition_window`` maps (condition_pct, window_ms) ->
    array of per-trial counts for that cell (typically FIT cells only).

    - H0 (pure Poisson): variance = mean, 0 free variance-shape params.
    - H1 (doubly stochastic / this project's model): variance predicted
      by fano_factor_formula(lambda(c), T, dt, sigma_g2_hat) * mean,
      1 free variance-shape parameter (sigma_g2_hat, already estimated
      elsewhere and passed in).
    - H2 (constant overdispersion): variance = phi_hat * mean for a
      single shared phi_hat regardless of window length, 1 free
      variance-shape parameter.

    All three models use the SAME mean_pred = lambda_hat(c) * T for every
    cell, so the comparison isolates the variance-structure assumption.
    """
    ll = {"H0_poisson": 0.0, "H1_doubly_stochastic": 0.0, "H2_constant_overdispersion": 0.0}
    for (c, T), counts in counts_by_condition_window.items():
        lam_c = lambda_hz_by_condition[c]
        mean_pred = mean_count_formula(lam_c, T)
        ll["H0_poisson"] += _quasi_loglik(counts, mean_pred, mean_pred)
        ff1 = float(fano_factor_formula(lam_c, np.array([T]), dt_ms, sigma_g2_hat)[0])
        ll["H1_doubly_stochastic"] += _quasi_loglik(counts, mean_pred, ff1 * mean_pred)
        ll["H2_constant_overdispersion"] += _quasi_loglik(counts, mean_pred, phi_hat * mean_pred)

    n_params = {"H0_poisson": 0, "H1_doubly_stochastic": 1, "H2_constant_overdispersion": 1}
    aic = {m: -2 * ll[m] + 2 * n_params[m] for m in ll}
    return ModelComparisonResult(loglik=ll, n_params=n_params, aic=aic)


def fit_phi_constant_overdispersion(counts_by_condition_window: Dict[tuple, np.ndarray]) -> float:
    """Method-of-moments estimate of a single shared constant Fano factor
    phi across all supplied cells, weighted by each cell's total expected
    information (mean count x number of trials) -- cells with more
    spikes and more trials get more weight, matching how their sampling
    variance in FF actually scales.
    """
    ffs, weights = [], []
    for (c, T), counts in counts_by_condition_window.items():
        mean_c = counts.mean()
        var_c = counts.var(ddof=1)
        if mean_c <= 0:
            continue
        ffs.append(var_c / mean_c)
        weights.append(mean_c * len(counts))
    ffs = np.array(ffs)
    weights = np.array(weights)
    return float(np.sum(ffs * weights) / np.sum(weights))


# ---------------------------------------------------------------------------
# Stage 6: simulation-based goodness-of-fit (parametric bootstrap)
# ---------------------------------------------------------------------------

def parametric_bootstrap_pvalue(observed_counts: np.ndarray, mean_pred: float, var_pred: float,
                                 n_boot: int, rng: np.random.Generator,
                                 statistic: str = "fano_factor") -> tuple:
    """Simulate n_boot replicate datasets of the same size as
    ``observed_counts`` from a Normal(mean_pred, var_pred) approximation
    (clipped at 0, since counts cannot be negative -- a minor
    approximation that matters only for very small mean_pred), compute
    the chosen summary statistic for each replicate, and return
    (observed_statistic, two_sided-ish p-value, null_distribution).

    Using a Gaussian generative approximation here (rather than resampling
    directly from a Poisson or Binomial with the candidate model's
    variance) keeps this function usable for ANY candidate mean/variance
    pair, including ones (like H1's) that do not correspond to a single
    named textbook distribution -- exactly the situation a real simulation
    -based check is for.
    """
    n = len(observed_counts)
    sd_pred = np.sqrt(max(var_pred, 1e-8))
    sims = rng.normal(mean_pred, sd_pred, size=(n_boot, n))
    sims = np.clip(sims, 0, None)

    def stat(x):
        m = x.mean(axis=-1)
        v = x.var(axis=-1, ddof=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return v / m

    if statistic != "fano_factor":
        raise ValueError("only 'fano_factor' is implemented")

    obs_stat = float(observed_counts.var(ddof=1) / observed_counts.mean())
    null_stats = stat(sims)
    null_stats = null_stats[np.isfinite(null_stats)]
    pval = (np.sum(null_stats >= obs_stat) + 1) / (len(null_stats) + 1)
    return obs_stat, float(pval), null_stats


def poisson_null_detectability_pvalue(observed_counts: np.ndarray, lam_hz: float, T_ms: float,
                                       n_boot: int, rng: np.random.Generator) -> tuple:
    """The per-cell detectability test used in handout Section 6: is this
    cell's observed Fano factor significantly ABOVE what a strict Poisson
    process at rate lambda_hat would produce, given the same number of
    trials? Uses an exact Poisson parametric bootstrap (not the Gaussian
    approximation above) since H0 is a single, exactly simulable named
    distribution.
    """
    n = len(observed_counts)
    mu = mean_count_formula(lam_hz, T_ms)
    obs_ff = float(observed_counts.var(ddof=1) / observed_counts.mean())
    sims = rng.poisson(mu, size=(n_boot, n))
    with np.errstate(invalid="ignore", divide="ignore"):
        null_ff = sims.var(axis=1, ddof=1) / sims.mean(axis=1)
    null_ff = null_ff[np.isfinite(null_ff)]
    pval = (np.sum(null_ff >= obs_ff) + 1) / (len(null_ff) + 1)
    return obs_ff, float(pval)


# ---------------------------------------------------------------------------
# Stage 8: holdout prediction
# ---------------------------------------------------------------------------

def predict_holdout_ff(lambda_hz: float, windows_ms: Sequence[float], dt_ms: float,
                        sigma_g2_hat: float) -> np.ndarray:
    """Predicted FF(T) for a condition using ONLY its own (independently
    estimated) lambda and the FIT-derived shared sigma_g2_hat.
    """
    return fano_factor_formula(lambda_hz, np.asarray(windows_ms, dtype=float), dt_ms, sigma_g2_hat)


def holdout_rmse(observed_ff: np.ndarray, predicted_ff: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(observed_ff) - np.asarray(predicted_ff)) ** 2)))


# ---------------------------------------------------------------------------
# Uncertainty quantification: trial-resampling bootstrap for sigma_g_hat
# ---------------------------------------------------------------------------

def bootstrap_sigma_g_ci(counts_by_condition: Dict[float, np.ndarray], fit_contrasts: Sequence[float],
                          fit_window_idx: Sequence[int], fit_windows_ms: Sequence[float],
                          lambda_hz_by_condition: Dict[float, float],
                          n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Resample trials WITH replacement independently within each FIT
    condition, n_boot times, and re-run stages 2-4 of the pipeline on each
    resample to build a bootstrap distribution of sigma_g_hat. Returns an
    array of length n_boot of sigma_g_hat values (not sigma_g^2, for
    direct use in a percentile CI on the natural, interpretable scale).
    """
    out = np.empty(n_boot)
    for b in range(n_boot):
        slopes = {}
        lam_boot = {}
        for c in fit_contrasts:
            counts = counts_by_condition[c]
            n_trials = counts.shape[0]
            idx = rng.integers(0, n_trials, size=n_trials)
            resampled = counts[idx][:, fit_window_idx]
            lam_boot[c] = estimate_lambda_hz(resampled[:, 0], fit_windows_ms[0])
            ff = empirical_fano_factor(resampled)
            slope, _ = fit_ff_vs_window_slope(fit_windows_ms, ff)
            slopes[c] = slope
        out[b] = np.sqrt(max(pooled_sigma_g2(lam_boot, slopes), 0.0))
    return out
