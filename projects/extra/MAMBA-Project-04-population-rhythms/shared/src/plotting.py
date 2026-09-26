"""
plotting.py
===========
Figure-generation helpers for MAMBA Project 02: Population Rhythms.

Every function returns the created ``matplotlib.figure.Figure``; no
function calls ``plt.show()``.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis import (condition_control_values, condition_roles, trials_for_condition,
                       dominant_eigenvalue, predicted_frequency_hz, predicted_stability)


def plot_example_traces(data, condition_id: int, trial: int = 0, window=(0, 1000)):
    """Raw noisy (E, I) trace for one example trial."""
    t = data["t_ms"]
    E, I, trial_ids = trials_for_condition(data, condition_id)
    row = list(trial_ids).index(trial)
    mask = (t >= window[0]) & (t <= window[1])

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(t[mask], E[row][mask], label="E (excitatory)", color="tab:red", linewidth=0.8)
    ax.plot(t[mask], I[row][mask], label="I (inhibitory)", color="tab:blue", linewidth=0.8)
    I_I = condition_control_values(data)[condition_id]
    role = condition_roles(data)[condition_id]
    ax.set_xlabel("Time since recording start (ms)")
    ax.set_ylabel("Population activity")
    ax.set_title(f"Condition {condition_id} (I_I={I_I}, {role}), trial {trial}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_eigenvalue_trend(fit_results: dict, bp):
    """Two panels: Re(dominant eigenvalue) vs I_I with the linear
    extrapolation to the predicted bifurcation, and Im(dominant
    eigenvalue) vs I_I with its own linear trend."""
    I_vals = np.array([fr.I_I for fr in fit_results.values()])
    re_vals = np.array([dominant_eigenvalue(fr.eig).real for fr in fit_results.values()])
    im_vals = np.array([abs(dominant_eigenvalue(fr.eig).imag) for fr in fit_results.values()])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.scatter(I_vals, re_vals, color="black", zorder=5, label="estimated (fit conditions)")
    I_line = np.linspace(min(I_vals.min(), bp.I_I_crit) - 0.3, I_vals.max() + 0.3, 50)
    ax.plot(I_line, bp.intercept_re + bp.slope_re * I_line, "-", color="tab:blue",
            label="linear extrapolation")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.axvline(bp.I_I_crit, color="tab:red", linestyle="--",
               label=f"predicted $I_I^{{crit}}$={bp.I_I_crit:.2f}")
    ax.set_xlabel("$I_I$")
    ax.set_ylabel("Re(dominant eigenvalue) (1/ms)")
    ax.set_title("Stability trend and extrapolated bifurcation")
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.scatter(I_vals, im_vals, color="black", zorder=5)
    ax.plot(I_line, np.abs(bp.intercept_im + bp.slope_im * I_line), "-", color="tab:blue")
    ax.axvline(bp.I_I_crit, color="tab:red", linestyle="--")
    ax.set_xlabel("$I_I$")
    ax.set_ylabel("|Im(dominant eigenvalue)| (rad/ms)")
    ax.set_title("Quasi-cycle frequency trend")

    fig.tight_layout()
    return fig


def plot_phase_diagram(data, bp, validation_report):
    """The project's headline figure: a model-free indicator of oscillatory
    behavior (std of E) plotted against I_I for every condition (fit and
    holdout visually distinguished), with the model's predicted
    bifurcation location marked -- i.e. a validated phase diagram."""
    currents = condition_control_values(data)
    roles = condition_roles(data)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for cid, I_I in currents.items():
        role = roles[cid]
        std_E = validation_report.per_condition[cid]["std_E"]
        marker = "o" if role == "fit" else "^"
        color = "black" if role == "fit" else "tab:red"
        ax.scatter([I_I], [std_E], marker=marker, color=color, s=70, zorder=5)

    ax.scatter([], [], marker="o", color="black", label="fit condition")
    ax.scatter([], [], marker="^", color="tab:red", label="holdout condition")
    ax.axvline(bp.I_I_crit, color="tab:blue", linestyle="--",
               label=f"predicted $I_I^{{crit}}$={bp.I_I_crit:.2f}")
    ax.set_xlabel("$I_I$ (external drive to inhibitory population)")
    ax.set_ylabel("std(E) across all trials/samples")
    ax.set_title("Phase diagram: amplitude vs. control parameter")
    ax.invert_xaxis()  # I_I decreasing (toward instability) reads left-to-right
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_psd_comparison(data, condition_ids: list):
    """Overlaid power spectral densities (Welch, trial-averaged) for a
    handful of conditions, to visually compare spectral sharpness."""
    from scipy.signal import welch
    t = data["t_ms"]
    dt = float(t[1] - t[0])
    fs = 1000.0 / dt
    currents = condition_control_values(data)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cid in condition_ids:
        E, I, _ = trials_for_condition(data, cid)
        psd_accum = None
        for row in range(E.shape[0]):
            x = E[row] - E[row].mean()
            freqs, psd = welch(x, fs=fs, nperseg=min(4096, len(x)))
            psd_accum = psd if psd_accum is None else psd_accum + psd
        psd_mean = psd_accum / E.shape[0]
        ax.plot(freqs, psd_mean, label=f"I_I={currents[cid]}")
    ax.set_xlim(0, 30)
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD of E (log scale)")
    ax.set_title("Power spectral density by condition")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_bootstrap_distribution(draws):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].hist(draws["I_I_crit"], bins=25, color="tab:blue", alpha=0.8)
    axes[0].set_title("Bootstrap: predicted $I_I^{crit}$")
    axes[0].set_xlabel("$I_I^{crit}$")
    axes[1].hist(draws["slope_re"], bins=25, color="tab:orange", alpha=0.8)
    axes[1].set_title("Bootstrap: Re(eigenvalue) vs. $I_I$ slope")
    axes[1].set_xlabel("slope")
    fig.tight_layout()
    return fig
