"""
analysis.py
===========
Reference analysis pipeline for MAMBA Project 02: Population Rhythms.

Implements one complete, defensible path from the raw observations in
``shared/data/`` to: (1) an estimated fixed point and local (linearized)
Jacobian at each FIT condition, (2) a linear extrapolation of the
Jacobian's dominant eigenvalue to predict the critical control value
I_I_crit at which the system loses stability (a Hopf bifurcation), (3) a
predicted oscillation ("quasi-cycle") frequency trend, and (4) a
validation report comparing these predictions against every condition,
including the two HOLDOUT conditions that were never used for fitting.

Participants are NOT required to follow this exact path -- see the
handout's "what is intentionally left for the group to decide" section.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass

import numpy as np
from scipy.signal import welch

from models import WilsonCowanParams, WilsonCowanPopulation


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(data_dir: str):
    npz = np.load(os.path.join(data_dir, "population_traces.npz"))
    protocol = np.genfromtxt(
        os.path.join(data_dir, "stimulus_protocol.csv"), delimiter=",",
        names=True, dtype=None, encoding="utf-8",
    )
    with open(os.path.join(data_dir, "noise_calibration.json")) as f:
        calibration = json.load(f)
    return {
        "t_ms": npz["t_ms"], "E_obs": npz["E_obs"], "I_obs": npz["I_obs"],
        "condition_id": npz["condition_id"], "trial": npz["trial"],
        "protocol": protocol, "sigma_obs": float(calibration["sigma_obs"]),
    }


def condition_control_values(data) -> dict:
    """Map condition_id -> commanded I_I."""
    return {int(row["condition_id"]): float(row["I_I"]) for row in data["protocol"]}


def condition_roles(data) -> dict:
    return {int(row["condition_id"]): str(row["role"]) for row in data["protocol"]}


def trials_for_condition(data, condition_id: int, trial_subset=None):
    mask = data["condition_id"] == condition_id
    trials_here = data["trial"][mask]
    E_here = data["E_obs"][mask]
    I_here = data["I_obs"][mask]
    if trial_subset is not None:
        trial_to_row = {int(t): i for i, t in enumerate(trials_here)}
        order = [trial_to_row[int(t)] for t in trial_subset]
        return E_here[order], I_here[order], np.asarray(trial_subset)
    return E_here, I_here, trials_here


# ---------------------------------------------------------------------------
# Fixed point + Jacobian (VAR(1)) estimation
# ---------------------------------------------------------------------------

@dataclass
class ConditionFit:
    I_I: float
    E_bar: float
    I_bar: float
    J: np.ndarray            # estimated 2x2 Jacobian
    eig: np.ndarray          # its 2 eigenvalues (complex)


def estimate_fixed_point(data, condition_id: int):
    """Sample mean of (E, I) across all recorded time points and trials of
    this condition -- the natural estimator of the equilibrium, since the
    stochastic simulation was built to fluctuate AROUND the true fixed
    point once burn-in has elapsed."""
    E, I, _ = trials_for_condition(data, condition_id)
    return float(E.mean()), float(I.mean())


def estimate_jacobian_var1(data, condition_id: int, E_bar: float, I_bar: float,
                            sigma_obs: float, trial_subset=None) -> np.ndarray:
    """Estimate the local Jacobian via a discrete-time VAR(1) fit to the
    mean-subtracted fluctuations:

        delta_x(t+dt) = A delta_x(t) + noise,   delta_x = (E-E_bar, I-I_bar)

    A naive ordinary-least-squares fit of this regression, pooling every
    trial of this condition, is SUBSTANTIALLY BIASED here: delta_x(t) is
    used as the REGRESSOR, but it is itself observed with independent
    measurement noise (amplitude `sigma_obs`, disclosed in
    shared/data/README.md) on top of the true fluctuation. This is a
    textbook errors-in-variables problem ("regression dilution"): the
    naive regression's design matrix has inflated variance on its diagonal
    (from the added noise) without a corresponding inflation of its
    covariance with the next time step, ATTENUATING the fitted transition
    matrix A toward zero. Because A itself sits close to the identity (the
    system barely moves from one 0.5 ms step to the next), shrinking A
    toward zero makes J = (A-I)/dt MORE NEGATIVE than the truth -- i.e.
    the naive estimate makes the population look MORE heavily damped
    (falsely MORE stable, not less) than it really is. At the most
    strongly damped conditions (where the true fluctuation amplitude is
    smallest relative to the fixed observation-noise floor), this bias is
    severe enough to also corrupt the off-diagonal terms and turn a
    genuinely complex eigenvalue pair into a spurious real pair. This
    matters beyond arithmetic: a naive analysis would lead a group to
    conclude the real system stays stable further than it actually does --
    exactly the wrong direction of error for a "how close is this circuit
    to losing stability" question.

    The correction: since observation noise is independent across time,
    E[X^T X] = E[X_true^T X_true] + n * sigma_obs^2 * I, so subtracting
    n * sigma_obs^2 * I from the sample X^T X before solving the normal
    equations removes the bias (to leading order):

        A = (X^T X - n sigma_obs^2 I)^{-1} X^T Y

    Pairs are built WITHIN each trial only; the last sample of one trial
    is never paired with the first sample of the next trial (they are not
    temporally adjacent). J is recovered from J = (A - I) / dt.
    """
    E, I, trial_ids = trials_for_condition(data, condition_id, trial_subset)
    dt = float(data["t_ms"][1] - data["t_ms"][0])

    X_rows, Y_rows = [], []
    for row in range(E.shape[0]):
        dE = E[row] - E_bar
        dI = I[row] - I_bar
        X_rows.append(np.stack([dE[:-1], dI[:-1]], axis=1))
        Y_rows.append(np.stack([dE[1:], dI[1:]], axis=1))
    X = np.concatenate(X_rows, axis=0)
    Y = np.concatenate(Y_rows, axis=0)
    n = X.shape[0]

    XtX_corrected = X.T @ X - n * sigma_obs ** 2 * np.eye(2)
    XtY = X.T @ Y
    A = np.linalg.solve(XtX_corrected, XtY).T  # x_{t+1} = A x_t
    J = (A - np.eye(2)) / dt
    return J


def fit_condition(data, condition_id: int, sigma_obs: float, trial_subset=None) -> ConditionFit:
    E_bar, I_bar = estimate_fixed_point(data, condition_id)
    J = estimate_jacobian_var1(data, condition_id, E_bar, I_bar, sigma_obs, trial_subset)
    eig = np.linalg.eigvals(J)
    I_I = condition_control_values(data)[condition_id]
    return ConditionFit(I_I=I_I, E_bar=E_bar, I_bar=I_bar, J=J, eig=eig)


def dominant_eigenvalue(eig: np.ndarray) -> complex:
    """The eigenvalue with the larger (least negative / most positive) real
    part -- the one that determines stability and, near the bifurcation,
    the emergent oscillation frequency."""
    return eig[np.argmax(eig.real)]


# ---------------------------------------------------------------------------
# Extrapolation to the bifurcation
# ---------------------------------------------------------------------------

@dataclass
class BifurcationPrediction:
    I_I_crit: float
    slope_re: float
    intercept_re: float
    slope_im: float
    intercept_im: float
    fit_points: dict  # condition_id -> ConditionFit


def predict_bifurcation(fit_results: dict) -> BifurcationPrediction:
    """Linearly regress Re(dominant eigenvalue) and Im(dominant eigenvalue)
    against I_I across the fit conditions, then solve Re(I_I) = 0 for the
    predicted critical I_I (the Hopf bifurcation location). This is an
    extrapolation OUTSIDE the range of I_I actually used for fitting --
    state that explicitly when reporting the prediction.
    """
    I_vals = np.array([fr.I_I for fr in fit_results.values()])
    re_vals = np.array([dominant_eigenvalue(fr.eig).real for fr in fit_results.values()])
    im_vals = np.array([abs(dominant_eigenvalue(fr.eig).imag) for fr in fit_results.values()])

    X = np.column_stack([np.ones_like(I_vals), I_vals])
    coeffs_re, *_ = np.linalg.lstsq(X, re_vals, rcond=None)
    intercept_re, slope_re = coeffs_re
    coeffs_im, *_ = np.linalg.lstsq(X, im_vals, rcond=None)
    intercept_im, slope_im = coeffs_im

    I_I_crit = -intercept_re / slope_re
    return BifurcationPrediction(I_I_crit=float(I_I_crit), slope_re=float(slope_re),
                                  intercept_re=float(intercept_re), slope_im=float(slope_im),
                                  intercept_im=float(intercept_im), fit_points=fit_results)


def predicted_frequency_hz(bp: BifurcationPrediction, I_I: float) -> float:
    """Predicted quasi-cycle / onset oscillation frequency at a given I_I,
    from the linear trend fitted to Im(dominant eigenvalue) vs I_I.
    Units: eigenvalues are in rad/ms, so Hz = |Im|/(2 pi) * 1000."""
    im_pred = bp.intercept_im + bp.slope_im * I_I
    return abs(im_pred) / (2 * np.pi) * 1000.0


def predicted_stability(bp: BifurcationPrediction, I_I: float) -> str:
    return "stable" if I_I > bp.I_I_crit else "oscillatory"


# ---------------------------------------------------------------------------
# Observed stability / frequency, directly from data (model-free)
# ---------------------------------------------------------------------------

@dataclass
class ObservedSummary:
    peak_freq_hz: float
    narrowband_fraction: float
    std_E: float


def observed_summary(data, condition_id: int, min_freq_hz: float = 1.0,
                      band_halfwidth_hz: float = 1.5) -> ObservedSummary:
    """Compute a MODEL-FREE summary of a condition's actual recorded
    dynamics, used to VALIDATE the model's predictions (so it must not use
    the fitted model at all):

    - peak_freq_hz: the frequency of the tallest peak in the trial-averaged
      Welch power spectral density of E, searched only above
      ``min_freq_hz`` (very-low-frequency / near-DC power reflects slow
      drift and finite-sample mean estimation, not oscillatory dynamics,
      and would otherwise dominate the search).
    - narrowband_fraction: the fraction of total power (again excluding
      near-DC) falling within +/- band_halfwidth_hz of that peak -- large
      for a sharp spectral line (a genuine sustained oscillation), small
      for a broad Lorentzian-like resonance (a noise-driven quasi-cycle).
    - std_E: the raw standard deviation of E across all samples and
      trials -- a simple, robust amplitude measure that should grow
      smoothly approaching a Hopf bifurcation and then grow sharply once
      a genuine (nonlinearly-saturated) limit cycle emerges past it.
    """
    E, I, _ = trials_for_condition(data, condition_id)
    dt = float(data["t_ms"][1] - data["t_ms"][0])
    fs = 1000.0 / dt

    psd_accum = None
    for row in range(E.shape[0]):
        x = E[row] - E[row].mean()
        freqs, psd = welch(x, fs=fs, nperseg=min(4096, len(x)))
        psd_accum = psd if psd_accum is None else psd_accum + psd
    psd_mean = psd_accum / E.shape[0]

    mask = freqs >= min_freq_hz
    f_search, p_search = freqs[mask], psd_mean[mask]
    peak_idx = np.argmax(p_search)
    peak_freq = float(f_search[peak_idx])

    band = (f_search >= peak_freq - band_halfwidth_hz) & (f_search <= peak_freq + band_halfwidth_hz)
    narrowband_fraction = float(p_search[band].sum() / p_search.sum())
    std_E = float(E.std())
    return ObservedSummary(peak_freq_hz=peak_freq, narrowband_fraction=narrowband_fraction,
                            std_E=std_E)


# ---------------------------------------------------------------------------
# Full validation report
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    per_condition: dict  # condition_id -> dict of predicted/observed values


def validate(data, bp: BifurcationPrediction) -> ValidationReport:
    """Compare the model's predictions against every condition (fit AND
    holdout). Two predictions are checked:

    1. STABILITY / bifurcation location: is I_I_crit_hat consistent with
       where the model-free indicators (std_E, narrowband_fraction) show
       the system's behavior actually change character? This is reported
       as numbers to plot (Section 8 of the handout), not reduced to a
       single pass/fail -- a real Hopf bifurcation in a NOISY system is a
       gradual-looking transition in these proxies, not a sharp step, so
       judge it from the figure, the same way you would with real data.
    2. FREQUENCY: the predicted quasi-cycle/oscillation frequency (from
       the Im(eigenvalue) trend, evaluated at each condition's own I_I,
       extrapolated for holdout conditions) vs. the observed PSD peak
       frequency, as a relative-error table exactly analogous to
       Project 01's F-I validation table.
    """
    currents = condition_control_values(data)
    roles = condition_roles(data)
    per_condition = {}
    for cid, I_I in currents.items():
        pred_freq = predicted_frequency_hz(bp, I_I)
        obs = observed_summary(data, cid)
        freq_rel_err = (abs(pred_freq - obs.peak_freq_hz) / obs.peak_freq_hz
                         if obs.peak_freq_hz > 0 else float("nan"))
        per_condition[cid] = {
            "I_I": I_I, "role": roles[cid],
            "predicted_stability": predicted_stability(bp, I_I),
            "predicted_freq_hz": pred_freq, "observed_freq_hz": obs.peak_freq_hz,
            "freq_relative_error": freq_rel_err,
            "narrowband_fraction": obs.narrowband_fraction, "std_E": obs.std_E,
        }
    return ValidationReport(per_condition=per_condition)


# ---------------------------------------------------------------------------
# Bootstrap over trials
# ---------------------------------------------------------------------------

def bootstrap_ci(data, fit_condition_ids: list, n_boot: int = 300, seed: int = 0,
                  ci: float = 0.90):
    rng = np.random.default_rng(seed)
    n_trials = int(data["protocol"][0]["n_trials"])
    sigma_obs = data["sigma_obs"]

    draws = {"I_I_crit": [], "slope_re": []}
    for b in range(n_boot):
        fit_results = {}
        for cid in fit_condition_ids:
            boot_trials = rng.integers(0, n_trials, size=n_trials)
            E_bar, I_bar = estimate_fixed_point_subset(data, cid, boot_trials)
            J = estimate_jacobian_var1(data, cid, E_bar, I_bar, sigma_obs, trial_subset=boot_trials)
            eig = np.linalg.eigvals(J)
            I_I = condition_control_values(data)[cid]
            fit_results[cid] = ConditionFit(I_I=I_I, E_bar=E_bar, I_bar=I_bar, J=J, eig=eig)
        try:
            bp = predict_bifurcation(fit_results)
        except Exception:
            continue
        if not np.isfinite(bp.I_I_crit):
            continue
        draws["I_I_crit"].append(bp.I_I_crit)
        draws["slope_re"].append(bp.slope_re)

    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    summary = {}
    for k, vals in draws.items():
        vals = np.asarray(vals)
        summary[k] = (float(np.quantile(vals, lo_q)), float(np.median(vals)),
                      float(np.quantile(vals, hi_q)))
    return summary, draws


def estimate_fixed_point_subset(data, condition_id, trial_subset):
    E, I, _ = trials_for_condition(data, condition_id, trial_subset)
    return float(E.mean()), float(I.mean())
