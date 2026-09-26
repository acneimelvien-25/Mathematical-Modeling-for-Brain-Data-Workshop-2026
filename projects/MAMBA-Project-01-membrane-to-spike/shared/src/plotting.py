"""
plotting.py
===========
Figure-generation helpers for MAMBA Project 01: Membrane to Spike.

Every function returns the created ``matplotlib.figure.Figure`` so callers
can either ``fig.savefig(...)`` (used by ``instructor/build_reference.py``)
or display inline in a notebook. No function calls ``plt.show()``.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import LIFParams, LeakyIntegrateFire
from analysis import condition_currents, trials_for_condition, _clean_trial_mask


def plot_subthreshold_fit(data, sub_fit, subthreshold_condition_ids):
    """Two panels: (left) steady-state depolarization vs. current with the
    fitted line, (right) trial-averaged onset transients with the fitted
    exponential overlaid."""
    t = data["t_ms"]
    currents = condition_currents(data)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    I_vals = np.array([sub_fit.V_ss_by_condition[c][0] for c in subthreshold_condition_ids])
    Vss_vals = np.array([sub_fit.V_ss_by_condition[c][1] for c in subthreshold_condition_ids])
    Vss_sem = np.array([sub_fit.V_ss_by_condition[c][2] for c in subthreshold_condition_ids])
    ax.errorbar(I_vals, Vss_vals, yerr=Vss_sem, fmt="o", color="black", capsize=3,
                label="observed (mean $\\pm$ SEM)")
    I_line = np.linspace(0, max(I_vals) * 1.1, 50)
    ax.plot(I_line, sub_fit.E_L + sub_fit.R * I_line, "-", color="tab:blue",
            label=f"fit: $E_L$={sub_fit.E_L:.2f} mV, $R$={sub_fit.R:.3f}")
    ax.set_xlabel("Injected current $I$ (µA/cm²)")
    ax.set_ylabel("Steady-state $V$ (mV)")
    ax.set_title("Subthreshold steady state vs. current")
    ax.legend(fontsize=8)

    ax = axes[1]
    for cid in subthreshold_condition_ids:
        I_dc = currents[cid]
        if abs(I_dc) < 1e-9:
            continue
        V, trial_ids = trials_for_condition(data, cid)
        clean = _clean_trial_mask(data, cid, trial_ids)
        mean_trace = V[clean].mean(axis=0)
        mask = (t >= 0) & (t <= 8.0)
        ax.plot(t[mask], mean_trace[mask], alpha=0.6, label=f"I={I_dc:.1f} (data)")
        Vss = sub_fit.V_ss_by_condition[cid][1]
        pred = Vss - (Vss - sub_fit.E_L) * np.exp(-t[mask] / sub_fit.tau_m)
        ax.plot(t[mask], pred, "--", color="black", linewidth=1)
    ax.set_xlabel("Time since step onset (ms)")
    ax.set_ylabel("Trial-averaged $V$ (mV)")
    ax.set_title(f"Onset transient fit ($\\tau_m$={sub_fit.tau_m:.3f} ms)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


def plot_fi_curve(data, lif_params: LIFParams, validation_report):
    """F-I curve: observed firing rate (fit vs. holdout conditions marked
    differently) against the analytic LIF prediction curve."""
    currents = condition_currents(data)
    protocol = {int(r["condition_id"]): str(r["role"]) for r in data["protocol"]}

    fig, ax = plt.subplots(figsize=(6, 4.5))
    I_line = np.linspace(0, max(currents.values()) * 1.05, 200)
    rate_line = LeakyIntegrateFire.analytic_firing_rate(
        I_line, lif_params.tau_m, lif_params.R, lif_params.E_L,
        lif_params.V_th, lif_params.V_reset, lif_params.t_ref)
    ax.plot(I_line, rate_line, "-", color="tab:blue", label="fitted LIF (analytic)")

    for cid, I_dc in currents.items():
        role = protocol[cid]
        obs = validation_report.observed_rate_hz[cid]
        marker = "o" if role == "fit" else "^"
        color = "black" if role == "fit" else "tab:red"
        ax.scatter([I_dc], [obs], marker=marker, color=color, s=60, zorder=5)

    ax.scatter([], [], marker="o", color="black", label="observed (fit condition)")
    ax.scatter([], [], marker="^", color="tab:red", label="observed (holdout condition)")
    ax.set_xlabel("Injected current $I$ (µA/cm²)")
    ax.set_ylabel("Firing rate (spikes/s)")
    ax.set_title("F-I curve: fitted LIF vs. observations")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_example_traces(data, condition_id: int, trial: int = 0, window=(0, 60)):
    """Raw noisy voltage trace for one example trial, with detected spikes
    marked -- useful as a sanity-check / qualitative figure."""
    t = data["t_ms"]
    from analysis import spikes_for_condition
    V, trial_ids = trials_for_condition(data, condition_id)
    row = list(trial_ids).index(trial)
    v_row = V[row]
    by_trial = spikes_for_condition(data, condition_id)
    spikes = by_trial.get(trial, [])

    mask = (t >= window[0]) & (t <= window[1])
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(t[mask], v_row[mask], color="black", linewidth=0.8)
    for s in spikes:
        if window[0] <= s <= window[1]:
            ax.axvline(s, color="tab:red", alpha=0.4, linewidth=1)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("$V_{obs}$ (mV)")
    ax.set_title(f"Condition {condition_id}, trial {trial} (spikes marked in red)")
    fig.tight_layout()
    return fig


def plot_isi_distributions(spiking_fit):
    """Histogram of ISIs for each spiking fit condition, used to sanity
    check the refractory-period / firing-regularity story."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for cid, isis in spiking_fit.isi_by_condition.items():
        ax.hist(isis, bins=20, alpha=0.5, label=f"condition {cid}")
    ax.axvline(spiking_fit.t_ref, color="black", linestyle="--",
               label=f"fitted $t_{{ref}}$={spiking_fit.t_ref:.2f} ms")
    ax.set_xlabel("Inter-spike interval (ms)")
    ax.set_ylabel("Count")
    ax.set_title("ISI distributions, spiking fit conditions")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_bootstrap_distributions(draws):
    """Small multiples of the bootstrap parameter distributions."""
    keys = list(draws.keys())
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, k in zip(axes.flat, keys):
        ax.hist(draws[k], bins=30, color="tab:blue", alpha=0.75)
        ax.set_title(k)
        ax.set_ylabel("count")
    fig.suptitle("Bootstrap distributions of fitted LIF parameters (trial resampling)")
    fig.tight_layout()
    return fig
