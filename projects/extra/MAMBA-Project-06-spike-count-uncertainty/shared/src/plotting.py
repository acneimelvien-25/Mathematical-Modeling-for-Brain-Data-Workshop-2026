"""
plotting.py
===========
Figure-generation helpers for MAMBA Project 05: Spike-Count Uncertainty.

Every function returns the created ``matplotlib.figure.Figure``; no
function calls ``plt.show()``.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import naka_rushton, fano_factor_formula


FIT_COLOR = "#1b6ca8"
HOLDOUT_COLOR = "#c1440e"
NEUTRAL_COLOR = "#555555"


def plot_tuning_curve(contrasts_pct, lambda_hat_by_condition, fit_contrasts, holdout_contrasts,
                       tuning_params=None):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    cs = np.array(sorted(contrasts_pct))
    for c in cs:
        color = FIT_COLOR if c in fit_contrasts else HOLDOUT_COLOR
        marker = "o" if c in fit_contrasts else "s"
        ax.scatter([c], [lambda_hat_by_condition[c]], color=color, marker=marker, s=70, zorder=3,
                   label=("fit condition" if c == fit_contrasts[0] else
                          ("holdout condition" if c == holdout_contrasts[0] else None)))
    if tuning_params is not None:
        c_grid = np.linspace(min(cs) * 0.8, max(cs) * 1.02, 200)
        ax.plot(c_grid, naka_rushton(c_grid, tuning_params), "--", color=NEUTRAL_COLOR, linewidth=1,
                label="reference tuning curve (instructor only)")
    ax.set_xscale("log")
    ax.set_xlabel("Contrast (%)")
    ax.set_ylabel(r"Estimated rate $\hat{\lambda}(c)$ (Hz)")
    ax.set_title("Contrast tuning curve\n(rate estimated from shortest window)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def plot_example_rasters(spike_lists_by_trial, condition_pct, T_max_ms=800.0, n_show=40):
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, spikes in enumerate(spike_lists_by_trial[:n_show]):
        ax.vlines(spikes, i, i + 0.8, color="black", linewidth=0.6)
    ax.set_xlim(0, T_max_ms)
    ax.set_ylim(0, n_show)
    ax.set_xlabel("Time from stimulus onset (ms)")
    ax.set_ylabel("Trial")
    ax.set_title(f"Example raster, contrast = {condition_pct:.0f}%  ({n_show} of many trials shown)")
    fig.tight_layout()
    return fig


def plot_fano_factor_vs_window(windows_ms, ff_by_condition, fit_contrasts, holdout_contrasts,
                                fit_windows_ms=None, holdout_window_ms=None,
                                predicted_ff_by_condition=None, dt_ms=1.0):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    windows_ms = np.asarray(windows_ms, dtype=float)
    cmap = plt.get_cmap("viridis")
    all_conditions = sorted(ff_by_condition.keys())
    for i, c in enumerate(all_conditions):
        color = cmap(i / max(len(all_conditions) - 1, 1))
        marker = "o" if c in fit_contrasts else "s"
        linestyle = "-" if c in fit_contrasts else "--"
        label = f"{c:.0f}% ({'fit' if c in fit_contrasts else 'holdout'})"
        ax.plot(windows_ms, ff_by_condition[c], marker=marker, linestyle="none", color=color,
                markersize=6, label=label)
        if predicted_ff_by_condition is not None and c in predicted_ff_by_condition:
            ax.plot(windows_ms, predicted_ff_by_condition[c], linestyle=linestyle, color=color,
                    linewidth=1.3, alpha=0.85)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":", label="Poisson (FF = 1)")
    if holdout_window_ms:
        for T in holdout_window_ms:
            ax.axvline(T, color=NEUTRAL_COLOR, linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Counting window T (ms)")
    ax.set_ylabel("Fano factor  Var[N(T)] / E[N(T)]")
    ax.set_title("Fano factor vs. counting-window length, by contrast")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()
    return fig


def plot_model_comparison_aic(aic_dict):
    fig, ax = plt.subplots(figsize=(5, 4))
    names = list(aic_dict.keys())
    values = np.array([aic_dict[k] for k in names])
    best = names[int(np.argmin(values))]
    delta = values - values.min()
    colors = ["#2a9d34" if k == best else "#a0a0a0" for k in names]
    ax.bar(range(len(names)), delta, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_ylabel(r"$\Delta$AIC relative to best model")
    ax.set_title("Model comparison on FIT data")
    for i, d in enumerate(delta):
        ax.text(i, d + 0.02 * delta.max(), f"{d:.1f}", ha="center", fontsize=8)
    fig.tight_layout()
    return fig


def plot_detectability_heatmap(contrasts_pct, windows_ms, pvalue_grid, fit_contrasts):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    contrasts_pct = list(contrasts_pct)
    windows_ms = list(windows_ms)
    grid = np.array(pvalue_grid)  # shape (len(contrasts), len(windows))
    im = ax.imshow(np.log10(grid), aspect="auto", cmap="RdYlBu_r", vmin=-4, vmax=0)
    ax.set_xticks(range(len(windows_ms)))
    ax.set_xticklabels([f"{int(w)}" for w in windows_ms])
    ax.set_yticks(range(len(contrasts_pct)))
    ylabels = [f"{c:.0f}%" + (" (fit)" if c in fit_contrasts else " (holdout)") for c in contrasts_pct]
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Counting window T (ms)")
    ax.set_ylabel("Contrast condition")
    ax.set_title("Detectability of overdispersion: log10(p-value)\n(vs. strict Poisson null, parametric bootstrap)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10(p-value)")
    for i in range(len(contrasts_pct)):
        for j in range(len(windows_ms)):
            p = grid[i, j]
            marker = "*" if p < 0.05 else ""
            ax.text(j, i, marker, ha="center", va="center", color="black", fontsize=12)
    fig.tight_layout()
    return fig


def plot_bootstrap_sigma_g(sigma_g_boot, sigma_g_true=None, sigma_g_hat=None):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(sigma_g_boot, bins=30, color=FIT_COLOR, alpha=0.8)
    if sigma_g_hat is not None:
        ax.axvline(sigma_g_hat, color="black", linewidth=1.5, label=r"$\hat{\sigma}_g$ (point estimate)")
    if sigma_g_true is not None:
        ax.axvline(sigma_g_true, color=HOLDOUT_COLOR, linewidth=1.5, linestyle="--",
                   label=r"true $\sigma_g$ (instructor only)")
    ax.set_xlabel(r"$\sigma_g$")
    ax.set_ylabel("Bootstrap count")
    ax.set_title("Trial-resampling bootstrap distribution of " + r"$\hat{\sigma}_g$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
