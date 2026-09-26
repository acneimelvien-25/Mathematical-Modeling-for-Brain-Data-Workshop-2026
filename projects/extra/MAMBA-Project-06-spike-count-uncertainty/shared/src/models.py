"""
models.py
=========
Model implementation for MAMBA Project 05: Spike-Count Uncertainty.

The generative model is a single cortical neuron whose instantaneous firing
rate on trial i, for stimulus condition c, is

    lambda_i(c) = lambda(c) * g_i

where ``lambda(c)`` is a fixed, condition-dependent mean rate (a contrast
tuning curve) and ``g_i`` is a trial-to-trial multiplicative gain drawn
once per trial from a shared, condition-INDEPENDENT distribution

    g_i ~ Normal(1, sigma_g^2)      (resampled if g_i <= 0)

representing a slowly varying, trial-level excitability/arousal state that
is common to the whole recording session. Given g_i, spikes within the
trial are generated as a memoryless point process: time is discretized
into bins of width ``dt`` (small enough that at most one spike can occur
per bin), and each bin is an independent Bernoulli trial with success
probability ``p_i = lambda_i(c) * dt``. The spike count in a window of
duration T (an integer number of bins, n = T/dt) is therefore, CONDITIONAL
on g_i, exactly

    N | g_i ~ Binomial(n, lambda(c) * g_i * dt)

which is the discrete-time process actually used to generate every trial
in this package. As dt -> 0 with n*dt = T fixed, Binomial(n, lambda*dt)
converges to Poisson(lambda*T) -- the classical Poisson-as-a-limit-of-
binomial correspondence -- so this is also, to an excellent approximation
(dt = 1 ms here, and lambda*dt <= 0.04 always), a doubly stochastic
("Cox") Poisson process with Gaussian-distributed rate gain.

Because the conditional distribution is exactly Binomial, the marginal
mean and variance of N (marginalizing over g_i) can be derived EXACTLY
via the law of total expectation and the law of total variance -- no
approximation is needed anywhere in ``fano_factor_formula`` below. See
the student handout Section 3 for the derivation; ``fano_factor_formula``
is the resulting closed-form prediction that the whole analysis pipeline
is built around.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Contrast tuning curve
# ---------------------------------------------------------------------------

@dataclass
class TuningCurveParams:
    R0: float      # baseline (spontaneous) rate, Hz
    Rmax: float    # maximum rate above baseline, Hz
    c50: float     # semi-saturation contrast, %
    n: float       # Naka-Rushton exponent (dimensionless)


def naka_rushton(contrast_pct: np.ndarray, params: TuningCurveParams) -> np.ndarray:
    """Naka-Rushton contrast-response function:

        lambda(c) = R0 + Rmax * c^n / (c^n + c50^n)

    ``contrast_pct`` and ``c50`` must be in the same units (percent
    Michelson contrast here). Returns a rate in Hz (spikes per second).
    This is a standard, smooth, saturating tuning-curve shape used
    throughout visual neuroscience; its specific parameter values are
    part of the hidden ground truth and are NOT given to participants.
    """
    c = np.asarray(contrast_pct, dtype=float)
    return params.R0 + params.Rmax * c ** params.n / (c ** params.n + params.c50 ** params.n)


# ---------------------------------------------------------------------------
# Trial-level spike generation
# ---------------------------------------------------------------------------

@dataclass
class GainNoiseParams:
    sigma_g: float   # SD of the trial-to-trial multiplicative gain, dimensionless


def sample_trial_gains(n_trials: int, params: GainNoiseParams, rng: np.random.Generator) -> np.ndarray:
    """Draw n_trials i.i.d. gains g_i ~ Normal(1, sigma_g^2), resampling any
    non-positive draws (a firing rate cannot be negative). For the sigma_g
    values used in this package (<= ~0.25), resampling affects a
    negligible fraction of draws (P(g <= 0) < 1e-8), so it does not
    materially bias E[g] away from 1.
    """
    g = rng.normal(1.0, params.sigma_g, size=n_trials)
    bad = g <= 0
    while np.any(bad):
        g[bad] = rng.normal(1.0, params.sigma_g, size=int(bad.sum()))
        bad = g <= 0
    return g


def generate_binned_spike_counts(lam_hz: float, n_trials: int, dt_ms: float, n_bins_max: int,
                                  gain_params: GainNoiseParams, rng: np.random.Generator) -> np.ndarray:
    """Generate the full binned spike-count trajectory for n_trials trials
    of a single condition with mean rate ``lam_hz``.

    Returns an array of shape (n_trials, n_bins_max) of 0/1 bin outcomes.
    Cumulative sums along axis 1, sliced at the appropriate bin index,
    give the spike count in any window T <= n_bins_max * dt_ms starting
    at trial onset -- because it is the SAME realized spike train being
    counted for different durations, counts at different T for the same
    trial are correlated (nested), exactly as in a real spike train.
    """
    g = sample_trial_gains(n_trials, gain_params, rng)
    lam_per_ms = lam_hz / 1000.0
    p = np.clip(lam_per_ms * g * dt_ms, 0.0, 1.0)
    bins = rng.binomial(1, p[:, None], size=(n_trials, n_bins_max))
    return bins


# ---------------------------------------------------------------------------
# The Fano-factor formula: the key analysis primitive
# ---------------------------------------------------------------------------

def fano_factor_formula(lam_hz: float, T_ms: np.ndarray, dt_ms: float, sigma_g2: float) -> np.ndarray:
    """Exact predicted Fano factor Var[N(T)] / E[N(T)] of the spike count
    in a window of duration T (ms), under the doubly stochastic
    Binomial-Gaussian-gain model above, as a function of the condition's
    mean rate ``lam_hz`` (Hz), the bin width ``dt_ms`` (ms), and the
    (shared, condition-independent) trial-gain variance ``sigma_g2``.

        FF(T) = 1 - lambda*dt*(1 + sigma_g^2) + lambda*T*sigma_g^2

    with lambda in spikes/ms (= lam_hz / 1000). Derive this yourself
    (handout Section 3.2) via the law of total variance applied to
    N | g ~ Binomial(T/dt, lambda*g*dt), before using it here -- this
    function does not derive it for you.

    Two regimes to note (see handout Section 3.4):
      - As dt -> 0 (or more precisely, whenever lambda*dt << 1), the
        constant term -lambda*dt*(1+sigma_g^2) is small, and
        FF(T) ~= 1 + lambda*T*sigma_g^2 -- pure Poisson-Gamma-like
        overdispersion, GROWING linearly with the window length T.
      - When sigma_g2 = 0 (no gain fluctuation at all), this reduces
        exactly to FF(T) = 1 - lambda*dt, the ordinary (very small,
        here always < 0.04) sub-Poisson correction from counting a
        Binomial rather than a continuous-time Poisson process.
    """
    lam_per_ms = lam_hz / 1000.0
    T = np.asarray(T_ms, dtype=float)
    return 1.0 - lam_per_ms * dt_ms * (1.0 + sigma_g2) + lam_per_ms * T * sigma_g2


def mean_count_formula(lam_hz: float, T_ms: np.ndarray) -> np.ndarray:
    """E[N(T)] = lambda * T exactly (regardless of sigma_g2, since
    E[g] = 1): the mean count is NOT informative about gain variability
    on its own -- only the variance (equivalently the Fano factor) is.
    """
    lam_per_ms = lam_hz / 1000.0
    return lam_per_ms * np.asarray(T_ms, dtype=float)
