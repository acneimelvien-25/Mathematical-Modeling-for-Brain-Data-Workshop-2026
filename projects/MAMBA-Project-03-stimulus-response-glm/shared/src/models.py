"""
models.py
=========
Model implementation for MAMBA Project 06: Stimulus-to-Response GLM.

Scenario: a single 2-photon-imaged visual cortex neuron's trial-evoked
response amplitude (a continuous measure, e.g. peak dF/F in arbitrary
units) is recorded across a session of trials. On each trial the animal
is shown a drifting grating at one of four orientations and one of four
contrasts. The neuron's true (noiseless) mean response on trial t is

    mean_response(t) = beta0
                       + beta_45  * D45(t)  + beta_90 * D90(t) + beta_135 * D135(t)
                       + beta_contrast * log2(contrast(t) / 12.5)
                       + drift(t_frac)
                       + noise

where D45/D90/D135 are treatment-coded orientation dummy variables
(baseline = 0 degrees), the contrast regressor is expressed as
log2(contrast_pct / 12.5) (0 at the lowest tested contrast, +1 per
doubling), and ``drift`` is a smooth, deterministic function of the
trial's fractional position in the session (t_frac = trial_index /
n_trials), representing a slow change in the neuron's baseline
excitability across the recording session (e.g. adaptation, electrode
drift, or slow state changes) -- NOT tied to any stimulus.

Critically, the STIMULUS SCHEDULE is not fully randomized with respect to
session time: orientation is block-randomized (uncorrelated with trial
number by construction), but contrast is drawn from a distribution whose
mean increases across the session (see ``generate_contrast_schedule``) --
producing a real confound between the contrast regressor and the omitted
drift term if a model does not include a drift regressor. See the student
handout Section 3 for the resulting omitted-variable-bias analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ORIENTATIONS_DEG = (0.0, 45.0, 90.0, 135.0)
CONTRASTS_PCT = (12.5, 25.0, 50.0, 100.0)


@dataclass
class TrueEffects:
    beta0: float
    beta_45: float
    beta_90: float
    beta_135: float
    beta_contrast: float
    sigma_noise: float


@dataclass
class DriftParams:
    d_lin: float
    d_quad: float
    d_sin: float
    d_cos: float


def drift_function(t_frac: np.ndarray, drift_params: DriftParams) -> np.ndarray:
    """The TRUE (hidden) session-drift function, evaluated at fractional
    trial position(s) t_frac in [0, 1). A smooth combination of a linear
    trend, a quadratic term, and one full slow oscillation cycle over the
    session -- deliberately not representable exactly by a single linear
    trend term alone (see handout Section 3.1).
    """
    t_frac = np.asarray(t_frac, dtype=float)
    p = drift_params
    return (p.d_lin * t_frac + p.d_quad * t_frac ** 2
            + p.d_sin * np.sin(2 * np.pi * t_frac) + p.d_cos * np.cos(2 * np.pi * t_frac))


def drift_basis(t_frac: np.ndarray) -> np.ndarray:
    """One reasonable basis for approximating ``drift_function`` in a
    regression: [t_frac, t_frac^2, sin(2*pi*t_frac), cos(2*pi*t_frac)].
    This EXACTLY spans the true drift function's functional form (by
    construction, for this synthetic dataset) -- a real dataset would not
    come with this guarantee, which is exactly why the handout asks
    participants to justify their own choice of basis rather than simply
    handing them this one.
    """
    t_frac = np.asarray(t_frac, dtype=float)
    return np.column_stack([t_frac, t_frac ** 2, np.sin(2 * np.pi * t_frac), np.cos(2 * np.pi * t_frac)])


def generate_orientation_sequence(n_trials: int, rng: np.random.Generator) -> np.ndarray:
    """Block-randomized orientation sequence: each consecutive block of 4
    trials contains one of each of the 4 orientations, in random order.
    This guarantees the orientation dummies are (very nearly) exactly
    uncorrelated with trial number / any smooth slow function of it,
    regardless of the drift's shape.
    """
    n_blocks = int(np.ceil(n_trials / len(ORIENTATIONS_DEG)))
    seq = []
    for _ in range(n_blocks):
        seq.extend(rng.permutation(ORIENTATIONS_DEG).tolist())
    return np.array(seq[:n_trials])


def generate_contrast_schedule(n_trials: int, rng: np.random.Generator,
                                block_probs=None) -> np.ndarray:
    """Contrast sequence drawn independently per trial, but from a
    distribution over the 4 contrast levels whose MEAN increases across
    four successive session blocks -- a real, if moderate, confound
    between contrast and trial number (NOT perfect collinearity: there is
    substantial contrast variability within every block).
    """
    if block_probs is None:
        block_probs = [
            [0.40, 0.30, 0.20, 0.10],
            [0.30, 0.30, 0.25, 0.15],
            [0.20, 0.25, 0.30, 0.25],
            [0.10, 0.15, 0.30, 0.45],
        ]
    n_blocks = len(block_probs)
    per_block = n_trials // n_blocks
    seq = []
    for b, probs in enumerate(block_probs):
        n_this = per_block if b < n_blocks - 1 else n_trials - per_block * (n_blocks - 1)
        seq.extend(rng.choice(CONTRASTS_PCT, size=n_this, p=probs).tolist())
    return np.array(seq[:n_trials])


def orientation_dummies(orientation_deg: np.ndarray) -> tuple:
    """Treatment-coded dummy variables (baseline = 0 degrees)."""
    D45 = (orientation_deg == 45.0).astype(float)
    D90 = (orientation_deg == 90.0).astype(float)
    D135 = (orientation_deg == 135.0).astype(float)
    return D45, D90, D135


def log_contrast_regressor(contrast_pct: np.ndarray) -> np.ndarray:
    """log2(contrast / 12.5): 0 at the lowest tested contrast, +1 per
    doubling of contrast."""
    return np.log2(np.asarray(contrast_pct, dtype=float) / 12.5)


def simulate_session(n_trials: int, effects: TrueEffects, drift_params: DriftParams,
                      rng: np.random.Generator) -> dict:
    """Simulate one full session. Returns a dict of arrays (all length
    n_trials): trial_index, orientation_deg, contrast_pct, t_frac,
    response, plus the noiseless mean_response and the true drift value
    (both instructor-only).
    """
    orientation_deg = generate_orientation_sequence(n_trials, rng)
    contrast_pct = generate_contrast_schedule(n_trials, rng)
    trial_index = np.arange(n_trials)
    t_frac = trial_index / n_trials

    D45, D90, D135 = orientation_dummies(orientation_deg)
    logC = log_contrast_regressor(contrast_pct)
    drift_true = drift_function(t_frac, drift_params)

    mean_response = (effects.beta0 + effects.beta_45 * D45 + effects.beta_90 * D90
                     + effects.beta_135 * D135 + effects.beta_contrast * logC + drift_true)
    response = mean_response + rng.normal(0.0, effects.sigma_noise, size=n_trials)

    return {
        "trial_index": trial_index,
        "orientation_deg": orientation_deg,
        "contrast_pct": contrast_pct,
        "t_frac": t_frac,
        "response": response,
        "mean_response": mean_response,   # instructor-only
        "drift_true": drift_true,          # instructor-only
    }


def build_design_matrix(orientation_deg: np.ndarray, contrast_pct: np.ndarray,
                         include_drift_basis: bool = False, t_frac: np.ndarray = None) -> np.ndarray:
    """Build the (n_trials x p) design matrix: intercept, 3 orientation
    dummies, 1 log-contrast regressor, and OPTIONALLY the 4-column drift
    basis. Column order: [1, D45, D90, D135, log_contrast, (drift basis...)].
    """
    D45, D90, D135 = orientation_dummies(orientation_deg)
    logC = log_contrast_regressor(contrast_pct)
    n = len(orientation_deg)
    X = np.column_stack([np.ones(n), D45, D90, D135, logC])
    if include_drift_basis:
        if t_frac is None:
            raise ValueError("t_frac is required when include_drift_basis=True")
        X = np.column_stack([X, drift_basis(t_frac)])
    return X


MAIN_EFFECT_NAMES = ["intercept", "D45", "D90", "D135", "log_contrast"]
DRIFT_BASIS_NAMES = ["t_frac", "t_frac_sq", "sin_2pi_t", "cos_2pi_t"]
