"""
plotting.py
===========
Figure-generation helpers for MAMBA Project 08: Coupled Populations.

Every function returns the created ``matplotlib.figure.Figure``; no
function calls ``plt.show()``.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FIT_COLOR = "#1b6ca8"
HOLDOUT_COLOR = "#c1440e"


def plot_fit_timeseries_example(time, x, w, n_show=200):
    fig, ax = plt.subplots(figsize=(7, 4))
    for i in range(x.shape[1]):
        ax.plot(time[:n_show], x[:n_show, i], label=f"node {i}", linewidth=1)
    ax.set_xlabel("Time")
    ax.set_ylabel("Activity")
    ax.set_title(f"Example FIT-regime activity, w={w}")
    ax.legend(fontsize=7, ncol=5, loc="upper right")
    fig.tight_layout()
    return fig


def plot_eigenvalue_spectrum(eigs_by_w, w_values, mode_k=None):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    cmap = plt.get_cmap("viridis")
    for i, w in enumerate(w_values):
        color = cmap(i / max(len(w_values) - 1, 1))
        eigs = eigs_by_w[w]
        ax.scatter(eigs.real, eigs.imag, color=color, label=f"w={w}", s=40)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax.set_xlabel(r"Re($\lambda$)")
    ax.set_ylabel(r"Im($\lambda$)")
    ax.set_title("Predicted eigenvalue spectrum (all modes, all w)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_gain_regression(w_values, coupling_coefs, g_hat):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    w = np.array(w_values)
    ax.scatter(w, coupling_coefs, color=FIT_COLOR, s=50, label="estimated (per condition)")
    w_grid = np.linspace(min(w) * 1.1, 0, 100)
    ax.plot(w_grid, g_hat * w_grid, "--", color="black", label=f"fit: coef = {g_hat:.3f} * w")
    ax.set_xlabel("Coupling strength w (FIT conditions)")
    ax.set_ylabel("Estimated coupling coefficient")
    ax.set_title("Pooled gain estimate (through-origin regression)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_holdout_timeseries(time, x, w, n_show=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    n_show = n_show or len(time)
    for i in range(x.shape[1]):
        ax.plot(time[:n_show], x[:n_show, i], label=f"node {i}", linewidth=1.2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Activity")
    ax.set_title(f"HOLDOUT activity, w={w}")
    ax.legend(fontsize=7, ncol=5, loc="upper right")
    fig.tight_layout()
    return fig


def plot_amplitude_vs_w(w_values, amplitudes, w_crit_predicted):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(w_values, amplitudes, "o-", color=HOLDOUT_COLOR)
    ax.axvline(w_crit_predicted, color="black", linestyle="--",
               label=f"predicted critical w = {w_crit_predicted:.3f}")
    ax.set_xlabel("Coupling strength w (HOLDOUT conditions)")
    ax.set_ylabel("Oscillation amplitude (max-min)")
    ax.set_title("Emergent oscillation amplitude vs. coupling strength")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_phase_lag_comparison(observed_lags_deg, predicted_lag_deg):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    n = len(observed_lags_deg)
    x = np.arange(n)
    ax.bar(x, observed_lags_deg, color=FIT_COLOR, label="observed (per node)")
    ax.axhline(predicted_lag_deg, color="black", linestyle="--",
               label=f"predicted = {predicted_lag_deg:.1f} deg")
    ax.set_xlabel("Node")
    ax.set_ylabel("Phase lag vs. upstream neighbor (degrees)")
    ax.set_title("Traveling-wave phase pattern: observed vs. predicted")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_directed_vs_symmetric(w_values, directed_amplitudes, symmetric_temporal_std):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(w_values, directed_amplitudes, "o-", color=HOLDOUT_COLOR, label="directed ring: oscillation amplitude")
    ax.plot(w_values, symmetric_temporal_std, "s--", color="#555555", label="symmetric ring: temporal std (near 0 = static)")
    ax.set_xlabel("Coupling strength |w|")
    ax.set_ylabel("Amplitude / temporal std")
    ax.set_title("Directed (oscillatory) vs. symmetric (stationary) ring")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
