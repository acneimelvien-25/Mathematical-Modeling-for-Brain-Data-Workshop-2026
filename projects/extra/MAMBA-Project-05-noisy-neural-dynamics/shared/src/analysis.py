"""
analysis.py
===========
Reference analysis pipeline for MAMBA Project 04: Noisy Neural Dynamics.

Implements one complete, defensible path from the raw observations in
``shared/data/`` to: (1) a per-session mean-voltage (V_ss) estimate, (2) a
pooled, irregular-sampling-aware estimate of the SHARED (tau, sigma_p,
sigma_obs) via nonlinear least squares on the empirical variogram
(structure function), (3) a per-session INDEPENDENT re-estimate of the
same three parameters, used to diagnose where recovery breaks down as a
function of sampling density, (4) bootstrap uncertainty quantification,
and (5) a validation report predicting each HOLDOUT session's own
variogram from the FIT-derived pooled parameters.

Participants are NOT required to follow this exact path -- see the
handout's "what is intentionally left for the group to decide" section.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

from models import variogram_formula


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(data_dir: str):
    npz = np.load(os.path.join(data_dir, "voltage_sessions.npz"))
    import csv as _csv
    with open(os.path.join(data_dir, "session_protocol.csv"), newline="") as f:
        rows = list(_csv.DictReader(f))
    protocol = [
        {"session_id": int(r["session_id"]), "mean_dt_ms": float(r["mean_dt_ms"]),
         "n_samples": int(r["n_samples"]), "duration_ms": float(r["duration_ms"]),
         "gamma_cv": float(r["gamma_cv"]), "role": r["role"]}
        for r in rows
    ]
    return {"session_id": npz["session_id"], "time_ms": npz["time_ms"],
            "V_obs_mV": npz["V_obs_mV"], "protocol": protocol}


def session_roles(data) -> dict:
    return {p["session_id"]: p["role"] for p in data["protocol"]}


def fit_session_ids(data) -> list:
    return sorted(p["session_id"] for p in data["protocol"] if p["role"] == "fit")


def holdout_session_ids(data) -> list:
    return sorted(p["session_id"] for p in data["protocol"] if p["role"] == "holdout")


def session_data(data, session_id: int):
    mask = data["session_id"] == session_id
    return data["time_ms"][mask], data["V_obs_mV"][mask]


# ---------------------------------------------------------------------------
# Per-session V_ss estimate
# ---------------------------------------------------------------------------

def estimate_V_ss(data, session_id: int) -> float:
    """The sample mean of a session's observations is an unbiased
    estimator of its true V_ss: the stationary distribution has mean
    V_ss, and observation noise is zero-mean, independent of the
    process."""
    _, V = session_data(data, session_id)
    return float(V.mean())


# ---------------------------------------------------------------------------
# Empirical variogram
# ---------------------------------------------------------------------------

@dataclass
class Variogram:
    dt_centers: np.ndarray
    values: np.ndarray
    counts: np.ndarray


def empirical_variogram(times: np.ndarray, V: np.ndarray, max_lag: float,
                          n_bins: int = 25, min_pairs_per_bin: int = 5) -> Variogram:
    """Bin ALL pairwise (not just consecutive) differences by their
    elapsed time, using log-spaced bins (concentrating resolution at
    small lags, where the interesting dynamics -- and the least
    identifiability, at large lags -- live). Using all pairs, not just
    consecutive ones, is valid because the OU process is stationary: the
    distribution of V(t+dt)-V(t) depends only on dt, not on t or on which
    samples happen to be adjacent in the recording.
    """
    n = len(times)
    idx = np.triu_indices(n, k=1)
    dt_all = times[idx[1]] - times[idx[0]]
    dV_all = V[idx[1]] - V[idx[0]]
    mask = (dt_all > 0) & (dt_all <= max_lag)
    dt_all, dV_all = dt_all[mask], dV_all[mask]

    bins = np.geomspace(dt_all.min(), max_lag, n_bins + 1)
    bin_idx = np.digitize(dt_all, bins)
    centers, values, counts = [], [], []
    for b in range(1, n_bins + 1):
        sel = bin_idx == b
        if sel.sum() < min_pairs_per_bin:
            continue
        centers.append(dt_all[sel].mean())
        values.append(np.mean(dV_all[sel] ** 2))
        counts.append(int(sel.sum()))
    return Variogram(dt_centers=np.array(centers), values=np.array(values),
                      counts=np.array(counts))


def pooled_variogram(data, session_ids: list, max_lag: float, n_bins: int = 25,
                       min_pairs_per_bin: int = 5) -> Variogram:
    """Combine pairwise differences from MULTIPLE sessions before binning.
    This is valid without first subtracting each session's own V_ss,
    because differencing (V(t+dt)-V(t)) automatically cancels any
    constant offset -- a session's own mean never appears in the
    variogram."""
    all_dt, all_dV2 = [], []
    for sid in session_ids:
        times, V = session_data(data, sid)
        n = len(times)
        idx = np.triu_indices(n, k=1)
        dt = times[idx[1]] - times[idx[0]]
        dV = V[idx[1]] - V[idx[0]]
        mask = (dt > 0) & (dt <= max_lag)
        all_dt.append(dt[mask])
        all_dV2.append(dV[mask] ** 2)
    dt_all = np.concatenate(all_dt)
    dV2_all = np.concatenate(all_dV2)

    bins = np.geomspace(dt_all.min(), max_lag, n_bins + 1)
    bin_idx = np.digitize(dt_all, bins)
    centers, values, counts = [], [], []
    for b in range(1, n_bins + 1):
        sel = bin_idx == b
        if sel.sum() < min_pairs_per_bin:
            continue
        centers.append(dt_all[sel].mean())
        values.append(dV2_all[sel].mean())
        counts.append(int(sel.sum()))
    return Variogram(dt_centers=np.array(centers), values=np.array(values),
                      counts=np.array(counts))


# ---------------------------------------------------------------------------
# Fitting the variogram
# ---------------------------------------------------------------------------

@dataclass
class DynamicsFit:
    tau: float
    sigma_p: float
    sigma_obs: float
    success: bool


def fit_variogram(vgram: Variogram, tau_init: float = 20.0) -> DynamicsFit:
    """Nonlinear least squares fit of models.variogram_formula to an
    empirical variogram, weighted by each bin's pair count (more pairs ->
    a more precisely estimated bin mean -> more weight)."""
    if len(vgram.dt_centers) < 4:
        return DynamicsFit(tau=np.nan, sigma_p=np.nan, sigma_obs=np.nan, success=False)
    try:
        sigma = 1.0 / np.sqrt(np.maximum(vgram.counts, 1))
        popt, _ = curve_fit(
            variogram_formula, vgram.dt_centers, vgram.values,
            p0=[tau_init, 10.0, 4.0], sigma=sigma,
            bounds=([0.5, 0.01, 0.001], [2000.0, 1000.0, 500.0]), maxfev=20000,
        )
        tau_hat, sp2tau_hat, obs2x2_hat = popt
        sigma_p_hat = np.sqrt(sp2tau_hat / tau_hat)
        sigma_obs_hat = np.sqrt(obs2x2_hat / 2.0)
        return DynamicsFit(tau=float(tau_hat), sigma_p=float(sigma_p_hat),
                             sigma_obs=float(sigma_obs_hat), success=True)
    except Exception:
        return DynamicsFit(tau=np.nan, sigma_p=np.nan, sigma_obs=np.nan, success=False)


def fit_pooled(data, session_ids: list, max_lag: float = 200.0, n_bins: int = 25) -> DynamicsFit:
    vgram = pooled_variogram(data, session_ids, max_lag=max_lag, n_bins=n_bins)
    return fit_variogram(vgram)


def fit_per_session(data, session_ids: list, max_lag: float = 200.0,
                      n_bins: int = 20) -> dict:
    """Fit EACH session independently (no pooling) -- used to diagnose
    how recovery quality depends on that session's own sampling
    density."""
    results = {}
    for sid in session_ids:
        times, V = session_data(data, sid)
        vgram = empirical_variogram(times, V, max_lag=max_lag, n_bins=n_bins)
        results[sid] = fit_variogram(vgram)
    return results


# ---------------------------------------------------------------------------
# Validation against holdout sessions
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    per_session: dict  # session_id -> dict with predicted/observed variogram values


def validate(data, pooled_fit: DynamicsFit, session_ids: list, max_lag: float = 400.0,
              n_bins: int = 20) -> ValidationReport:
    """For each given session (typically the holdout sessions, but this
    also works on fit sessions for a sanity check), compute its own
    empirical variogram and compare pointwise against the prediction from
    the pooled fit parameters."""
    per_session = {}
    for sid in session_ids:
        times, V = session_data(data, sid)
        vgram = empirical_variogram(times, V, max_lag=max_lag, n_bins=n_bins)
        predicted = variogram_formula(
            vgram.dt_centers, pooled_fit.tau,
            pooled_fit.sigma_p ** 2 * pooled_fit.tau, 2 * pooled_fit.sigma_obs ** 2,
        )
        rel_err = np.abs(predicted - vgram.values) / np.maximum(vgram.values, 1e-9)
        per_session[sid] = {
            "dt_centers": vgram.dt_centers, "observed": vgram.values,
            "predicted": predicted, "counts": vgram.counts,
            "mean_relative_error": float(np.average(rel_err, weights=vgram.counts)),
        }
    return ValidationReport(per_session=per_session)


# ---------------------------------------------------------------------------
# Bootstrap over sessions/pairs
# ---------------------------------------------------------------------------

def bootstrap_pooled_tau(data, session_ids: list, n_boot: int = 300, seed: int = 0,
                           ci: float = 0.90, max_lag: float = 200.0, n_bins: int = 25):
    """Resample WITHIN each session (bootstrap over that session's own
    samples, preserving each sample's own time -- i.e. resample which
    time points are included, not the times themselves) and refit the
    pooled variogram each time."""
    rng = np.random.default_rng(seed)
    draws = {"tau": [], "sigma_p": [], "sigma_obs": []}

    session_arrays = {sid: session_data(data, sid) for sid in session_ids}

    for b in range(n_boot):
        all_dt, all_dV2 = [], []
        for sid in session_ids:
            times, V = session_arrays[sid]
            n = len(times)
            boot_idx = rng.integers(0, n, size=n)
            t_b, V_b = times[boot_idx], V[boot_idx]
            order = np.argsort(t_b)
            t_b, V_b = t_b[order], V_b[order]
            idx = np.triu_indices(n, k=1)
            dt = t_b[idx[1]] - t_b[idx[0]]
            dV = V_b[idx[1]] - V_b[idx[0]]
            mask = (dt > 0) & (dt <= max_lag)
            all_dt.append(dt[mask])
            all_dV2.append(dV[mask] ** 2)
        dt_all = np.concatenate(all_dt)
        dV2_all = np.concatenate(all_dV2)
        bins = np.geomspace(dt_all.min(), max_lag, n_bins + 1)
        bin_idx = np.digitize(dt_all, bins)
        centers, values, counts = [], [], []
        for bi in range(1, n_bins + 1):
            sel = bin_idx == bi
            if sel.sum() < 5:
                continue
            centers.append(dt_all[sel].mean())
            values.append(dV2_all[sel].mean())
            counts.append(int(sel.sum()))
        vgram = Variogram(dt_centers=np.array(centers), values=np.array(values),
                            counts=np.array(counts))
        fit = fit_variogram(vgram)
        if fit.success:
            draws["tau"].append(fit.tau)
            draws["sigma_p"].append(fit.sigma_p)
            draws["sigma_obs"].append(fit.sigma_obs)

    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    summary = {}
    for k, vals in draws.items():
        vals = np.asarray(vals)
        if len(vals) == 0:
            summary[k] = (np.nan, np.nan, np.nan)
        else:
            summary[k] = (float(np.quantile(vals, lo_q)), float(np.median(vals)),
                          float(np.quantile(vals, hi_q)))
    return summary, draws
