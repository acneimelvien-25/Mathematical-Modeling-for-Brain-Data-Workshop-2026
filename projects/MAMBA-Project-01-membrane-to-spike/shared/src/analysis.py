"""
analysis.py
===========
Reference analysis pipeline for MAMBA Project 01: Membrane to Spike.

This module implements one complete, defensible path from the raw
observations in ``shared/data/`` to fitted LIF parameters, their bootstrap
uncertainty, and a validation report against the two held-out current
levels. It is used by:

  * ``shared/tests/`` to check the core equations and the end-to-end pipeline,
  * ``instructor/`` to produce the reference outputs and answer key numbers.

Participants are NOT required to follow this exact path (see the handout's
"what is intentionally left for the group to decide" section) -- alternative
choices of numerical method, windowing, or optimization routine are equally
acceptable if justified and validated. This module documents ONE reference
route so the instructor team has reproducible target numbers and tolerance
bands to grade against.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar

from models import LIFParams, LeakyIntegrateFire


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(data_dir: str):
    """Load the three data files into simple in-memory structures.

    Returns a dict with keys: t_ms, V_obs_mV, condition_id, trial (from the
    npz), spikes (list of (condition_id, trial, spike_time_ms) tuples), and
    protocol (list of dict rows from stimulus_protocol.csv).
    """
    npz = np.load(os.path.join(data_dir, "voltage_traces.npz"))
    spikes = np.genfromtxt(
        os.path.join(data_dir, "spike_times.csv"), delimiter=",",
        names=True, dtype=None, encoding="utf-8",
    )
    protocol = np.genfromtxt(
        os.path.join(data_dir, "stimulus_protocol.csv"), delimiter=",",
        names=True, dtype=None, encoding="utf-8",
    )
    return {
        "t_ms": npz["t_ms"],
        "V_obs_mV": npz["V_obs_mV"],
        "condition_id": npz["condition_id"],
        "trial": npz["trial"],
        "spikes": spikes,
        "protocol": protocol,
    }


def condition_currents(data) -> dict:
    """Map condition_id -> commanded I_dc (uA/cm^2)."""
    proto = data["protocol"]
    return {int(row["condition_id"]): float(row["I_dc_uA_per_cm2"]) for row in proto}


def condition_roles(data) -> dict:
    proto = data["protocol"]
    return {int(row["condition_id"]): str(row["role"]) for row in proto}


def trials_for_condition(data, condition_id: int, trial_subset: Optional[list] = None):
    """Return the (n_trials, T) voltage sub-matrix for one condition, plus
    the trial numbers (in row order), optionally restricted to
    ``trial_subset`` (used by the bootstrap)."""
    mask = data["condition_id"] == condition_id
    trials_here = data["trial"][mask]
    V_here = data["V_obs_mV"][mask]
    if trial_subset is not None:
        keep = np.isin(trials_here, trial_subset)
        # preserve resampling multiplicity for bootstrap: rebuild by index
        order = []
        trial_to_row = {t: i for i, t in enumerate(trials_here)}
        for t in trial_subset:
            order.append(trial_to_row[t])
        return V_here[order], np.asarray(trial_subset)
    return V_here, trials_here


def spikes_for_condition(data, condition_id: int, trial_subset: Optional[list] = None):
    sp = data["spikes"]
    mask = sp["condition_id"] == condition_id
    rows = sp[mask]
    if trial_subset is not None:
        out = []
        for t in trial_subset:
            out.append(rows[rows["trial"] == t]["spike_time_ms"])
        return out  # list of arrays, one per (possibly repeated) trial
    # group by trial
    by_trial = {}
    for row in rows:
        by_trial.setdefault(int(row["trial"]), []).append(float(row["spike_time_ms"]))
    return by_trial


# ---------------------------------------------------------------------------
# Subthreshold fit: E_L, R, tau_m
# ---------------------------------------------------------------------------

@dataclass
class SubthresholdFit:
    E_L: float
    R: float
    tau_m: float
    V_ss_by_condition: dict  # condition_id -> (I_dc, V_ss_mean, V_ss_sem, n_clean_trials)


def _clean_trial_mask(data, condition_id: int, trial_ids: np.ndarray) -> np.ndarray:
    """A trial is 'clean' (usable for subthreshold analysis) if it has ZERO
    detected spikes anywhere in its recording. Even conditions chosen to sit
    below the deterministic rheobase occasionally cross threshold once
    input-current noise is added (a real phenomenon: noise-driven excitable
    excursions), so any window-based subthreshold estimator must screen
    trials for contamination before averaging."""
    by_trial = spikes_for_condition(data, condition_id)
    return np.array([len(by_trial.get(int(tid), [])) == 0 for tid in trial_ids])


def fit_subthreshold(data, subthreshold_condition_ids: list, ss_window=(100.0, 200.0),
                      tau_window=(0.0, 8.0), tau_bounds=(0.05, 50.0)) -> SubthresholdFit:
    """Estimate (E_L, R) from steady-state depolarization vs. current
    (linear regression), and tau_m from a JOINT nonlinear least-squares fit
    of the trial-averaged onset transient to the analytic RC solution, using
    only the caller-supplied SUBTHRESHOLD conditions (no repetitive spiking
    expected).

    Steady state: V_ss(I) = E_L + R * I  -> ordinary least squares on
    (I, V_ss) pairs (this is the two-column design matrix [1, I]).

    Time constant: the model's own analytic solution for a current step
    from rest is V(t) = V_ss - (V_ss - E_L) exp(-t/tau_m). tau_m is
    recovered by minimizing the summed squared error between this curve and
    the trial-averaged onset trace, POOLED across every non-zero-current
    subthreshold condition with a single shared tau_m. This nonlinear
    least-squares fit (rather than log-linearizing the exponential) avoids
    amplifying observation noise in the tail of the transient, where the
    true deviation from V_ss is already small.

    Because the reference neuron's effective membrane time constant is only
    a few hundred microseconds to a couple of milliseconds (fast compared to
    the observation noise and sample interval), tau_m is the least
    identifiable parameter in this whole pipeline -- expect the recovered
    value to be correct only to within roughly a factor of 2-3, and expect
    the bootstrap interval on tau_m to be wide. Reporting that spread
    honestly is itself part of the deliverable (handout Section 8).
    """
    t = data["t_ms"]
    currents = condition_currents(data)

    I_list, Vss_list = [], []
    Vss_by_cond = {}
    onset_curves = {}  # condition_id -> (t_window, mean_V_window over clean trials)

    for cid in subthreshold_condition_ids:
        I_dc = currents[cid]
        V, trial_ids = trials_for_condition(data, cid)
        clean = _clean_trial_mask(data, cid, trial_ids)
        V_clean = V[clean]

        ss_mask = (t >= ss_window[0]) & (t <= ss_window[1])
        per_trial_ss = V_clean[:, ss_mask].mean(axis=1)
        Vss_mean = per_trial_ss.mean()
        Vss_sem = per_trial_ss.std(ddof=1) / np.sqrt(len(per_trial_ss)) if len(per_trial_ss) > 1 else np.nan
        Vss_by_cond[cid] = (I_dc, float(Vss_mean), float(Vss_sem), int(clean.sum()))
        I_list.append(I_dc)
        Vss_list.append(Vss_mean)

        onset_mask = (t >= tau_window[0]) & (t <= tau_window[1])
        onset_curves[cid] = (t[onset_mask], V_clean[:, onset_mask].mean(axis=0))

    I_arr = np.asarray(I_list)
    Vss_arr = np.asarray(Vss_list)
    X = np.column_stack([np.ones_like(I_arr), I_arr])
    coeffs, *_ = np.linalg.lstsq(X, Vss_arr, rcond=None)
    E_L_hat, R_hat = coeffs

    # tau_m: joint nonlinear fit, pooled across every non-zero-current
    # subthreshold condition, sharing one tau_m but each condition's own
    # (already-estimated) Vss.
    curves = []
    for cid in subthreshold_condition_ids:
        I_dc, Vss_mean, _, _ = Vss_by_cond[cid]
        if abs(I_dc) < 1e-9:
            continue
        tt, vv = onset_curves[cid]
        curves.append((tt, vv, Vss_mean))

    def sse(tau_m):
        total = 0.0
        for tt, vv, Vss in curves:
            pred = Vss - (Vss - E_L_hat) * np.exp(-tt / tau_m)
            total += np.sum((vv - pred) ** 2)
        return total

    result = minimize_scalar(sse, bounds=tau_bounds, method="bounded")
    tau_m_hat = float(result.x)

    return SubthresholdFit(E_L=float(E_L_hat), R=float(R_hat), tau_m=tau_m_hat,
                            V_ss_by_condition=Vss_by_cond)


# ---------------------------------------------------------------------------
# Spiking fit: V_th, V_reset, t_ref
# ---------------------------------------------------------------------------

@dataclass
class SpikingFit:
    V_th: float
    V_reset: float
    t_ref: float
    isi_by_condition: dict  # condition_id -> array of ISIs (ms)
    rate_by_condition: dict  # condition_id -> (I_dc, observed_rate_hz)


def fit_spiking(data, spiking_condition_ids: list, subthreshold_fit: SubthresholdFit,
                 post_search_ms=6.0) -> SpikingFit:
    """Estimate V_reset, V_th, and t_ref from the spiking conditions.

    V_reset: the MINIMUM observed voltage within ``post_search_ms`` after
    each spike (the after-spike undershoot trough), averaged across spikes.
    Unlike an idealized LIF, the reference neuron has no flat post-spike
    plateau, so the trough is the best analog of "the voltage the model
    should reset to."

    V_th and t_ref: a naive approach would estimate V_th from a fixed
    window of voltage immediately before each spike -- but the
    Hodgkin-Huxley upstroke is a fast regenerative event with no clean
    "threshold voltage" in that sense (see handout Section 6), and
    plugging a window-based V_th estimate directly into the analytic
    rate formula together with the independently-fitted subthreshold R
    typically implies a rheobase far outside the range where the neuron
    is actually observed to fire -- a self-inconsistent model. Instead,
    V_th and t_ref are solved for JOINTLY and EXACTLY from the two
    spiking fit conditions' empirical firing rates, by inverting the
    analytic ISI formula (handout Section 4) at both currents
    simultaneously:

        isi(I) = t_ref + tau_m * ln( (R I - (V_reset-E_L)) / (R I - (V_th-E_L)) )

    Writing D = V_th - E_L, a_i = R*I_i, b = V_reset - E_L, this is two
    equations in the two unknowns (D, t_ref); eliminating t_ref gives a
    closed-form solution for D (derived in instructor_guide.md), hence
    V_th, and then t_ref follows by back-substitution. This guarantees the
    fitted model reproduces the two FIT firing rates exactly by
    construction -- the real test of the model is then whether it predicts
    the two HELD-OUT conditions, which it was never forced to match.
    """
    t = data["t_ms"]
    currents = condition_currents(data)

    post_mins = []
    isi_by_cond = {}
    rate_by_cond = {}

    for cid in spiking_condition_ids:
        I_dc = currents[cid]
        by_trial_spikes = spikes_for_condition(data, cid)
        V, trial_ids = trials_for_condition(data, cid)
        trial_to_row = {int(tr): i for i, tr in enumerate(trial_ids)}

        isis = []
        for trial, spk_list in by_trial_spikes.items():
            spk_arr = np.sort(np.asarray(spk_list))
            isis.extend(np.diff(spk_arr).tolist())
            row = trial_to_row[trial]
            v_row = V[row]
            for s in spk_arr:
                post_mask = (t > s) & (t <= s + post_search_ms)
                if post_mask.sum() > 0:
                    post_mins.append(v_row[post_mask].min())
        isi_by_cond[cid] = np.asarray(isis)
        rate_by_cond[cid] = (I_dc, empirical_firing_rate_hz(data, cid))

    V_reset_hat = float(np.mean(post_mins))
    b = V_reset_hat - subthreshold_fit.E_L
    tau_m = subthreshold_fit.tau_m
    R = subthreshold_fit.R

    if len(spiking_condition_ids) != 2:
        raise ValueError("fit_spiking's closed-form solve requires exactly "
                          "two spiking conditions; got %d" % len(spiking_condition_ids))
    (I1, rate1), (I2, rate2) = (rate_by_cond[cid] for cid in spiking_condition_ids)
    a1, a2 = R * I1, R * I2
    isi1, isi2 = 1000.0 / rate1, 1000.0 / rate2

    K = (isi2 - isi1) / tau_m - np.log((a2 - b) / (a1 - b))
    E = np.exp(K)
    D = (a1 - E * a2) / (1.0 - E)
    V_th_hat = subthreshold_fit.E_L + D
    t_ref_hat = isi1 - tau_m * np.log((a1 - b) / (a1 - D))

    return SpikingFit(V_th=float(V_th_hat), V_reset=V_reset_hat, t_ref=float(t_ref_hat),
                       isi_by_condition=isi_by_cond, rate_by_condition=rate_by_cond)


# ---------------------------------------------------------------------------
# Firing rate summaries + validation against holdout conditions
# ---------------------------------------------------------------------------

def empirical_firing_rate_hz(data, condition_id: int) -> float:
    by_trial = spikes_for_condition(data, condition_id)
    proto_row = [r for r in data["protocol"] if int(r["condition_id"]) == condition_id][0]
    duration_s = float(proto_row["duration_ms"]) / 1000.0
    n_trials = float(proto_row["n_trials"])
    total_spikes = sum(len(v) for v in by_trial.values())
    return total_spikes / (duration_s * n_trials)


@dataclass
class ValidationReport:
    predicted_rate_hz: dict
    observed_rate_hz: dict
    relative_error: dict
    worst_condition: int
    worst_relative_error: float


def validate(data, lif_params: LIFParams) -> ValidationReport:
    """Compare the analytic LIF firing rate against the empirical firing
    rate at EVERY condition (fit and holdout), so the group can see both how
    well the fit describes its own training data and where it fails to
    extrapolate."""
    currents = condition_currents(data)
    predicted, observed, rel_err = {}, {}, {}
    for cid, I_dc in currents.items():
        f_pred = float(LeakyIntegrateFire.analytic_firing_rate(
            np.array([I_dc]), lif_params.tau_m, lif_params.R, lif_params.E_L,
            lif_params.V_th, lif_params.V_reset, lif_params.t_ref)[0])
        f_obs = empirical_firing_rate_hz(data, cid)
        predicted[cid] = f_pred
        observed[cid] = f_obs
        denom = max(f_obs, 1e-9)
        rel_err[cid] = abs(f_pred - f_obs) / denom if f_obs > 0 or f_pred > 0 else 0.0

    worst_cid = max(rel_err, key=rel_err.get)
    return ValidationReport(predicted, observed, rel_err, worst_cid, rel_err[worst_cid])


# ---------------------------------------------------------------------------
# Full fit pipeline + bootstrap
# ---------------------------------------------------------------------------

def full_fit(data, subthreshold_condition_ids: list, spiking_condition_ids: list) -> LIFParams:
    sub = fit_subthreshold(data, subthreshold_condition_ids)
    spk = fit_spiking(data, spiking_condition_ids, sub)
    return LIFParams(tau_m=sub.tau_m, R=sub.R, E_L=sub.E_L,
                      V_th=spk.V_th, V_reset=spk.V_reset, t_ref=spk.t_ref)


def bootstrap_ci(data, subthreshold_condition_ids: list, spiking_condition_ids: list,
                  n_boot: int = 500, seed: int = 0, ci: float = 0.90):
    """Resample trials WITH REPLACEMENT within each fit condition and rerun
    the full fit pipeline. Returns a dict of parameter -> (low, mid, high)
    at the requested central interval, plus the raw bootstrap draws."""
    rng = np.random.default_rng(seed)
    n_trials = int(data["protocol"][0]["n_trials"])  # same for every condition
    fit_condition_ids = list(subthreshold_condition_ids) + list(spiking_condition_ids)

    draws = {k: [] for k in ("tau_m", "R", "E_L", "V_th", "V_reset", "t_ref")}

    for b in range(n_boot):
        boot_trial_ids = {
            cid: rng.integers(0, n_trials, size=n_trials) for cid in fit_condition_ids
        }
        resampled_data = _build_resampled(data, boot_trial_ids)
        try:
            params = full_fit(resampled_data, subthreshold_condition_ids, spiking_condition_ids)
        except Exception:
            continue
        for k in draws:
            draws[k].append(getattr(params, k))

    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    summary = {}
    for k, vals in draws.items():
        vals = np.asarray(vals)
        summary[k] = (float(np.quantile(vals, lo_q)), float(np.median(vals)),
                      float(np.quantile(vals, hi_q)))
    return summary, draws


def _build_resampled(data, boot_trial_ids: dict):
    """Construct a resampled copy of the loaded dataset: for each condition
    in boot_trial_ids, duplicate the chosen trial rows (with replacement)
    and relabel them 0..n-1 so downstream code needs no special casing.
    Conditions not in boot_trial_ids are passed through unchanged."""
    t = data["t_ms"]
    all_cond, all_trial, all_V = [], [], []
    spike_rows = []

    present_conditions = np.unique(data["condition_id"])
    for cid in present_conditions:
        cid = int(cid)
        if cid in boot_trial_ids:
            V_here, trial_ids_here = trials_for_condition(data, cid)
            id_to_row = {int(tid): i for i, tid in enumerate(trial_ids_here)}
            by_trial_spk = spikes_for_condition(data, cid)
            for new_idx, old_trial in enumerate(boot_trial_ids[cid]):
                row = id_to_row[int(old_trial)]
                all_cond.append(cid)
                all_trial.append(new_idx)
                all_V.append(V_here[row])
                for s in by_trial_spk.get(int(old_trial), []):
                    spike_rows.append((cid, new_idx, s))
        else:
            V_here, trial_ids_here = trials_for_condition(data, cid)
            by_trial_spk = spikes_for_condition(data, cid)
            for i, tid in enumerate(trial_ids_here):
                all_cond.append(cid)
                all_trial.append(int(tid))
                all_V.append(V_here[i])
                for s in by_trial_spk.get(int(tid), []):
                    spike_rows.append((cid, int(tid), s))

    spikes_arr = np.array(
        spike_rows, dtype=[("condition_id", "i8"), ("trial", "i8"), ("spike_time_ms", "f8")]
    )
    return {
        "t_ms": t,
        "V_obs_mV": np.stack(all_V, axis=0),
        "condition_id": np.asarray(all_cond),
        "trial": np.asarray(all_trial),
        "spikes": spikes_arr,
        "protocol": data["protocol"],
    }
