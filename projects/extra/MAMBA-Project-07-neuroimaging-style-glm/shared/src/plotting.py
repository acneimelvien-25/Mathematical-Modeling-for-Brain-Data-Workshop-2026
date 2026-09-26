"""
plotting.py
===========
Figure-generation helpers for MAMBA Project 07: Neuroimaging-Style GLM.

Every function returns the created ``matplotlib.figure.Figure``; no
function calls ``plt.show()``.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import GRID_SHAPE


def _grid(values, fill=np.nan):
    n_rows, n_cols = GRID_SHAPE
    grid = np.full((n_rows, n_cols), fill, dtype=float)
    for c, v in enumerate(values):
        grid[c // n_cols, c % n_cols] = v
    return grid


def plot_channel_grid(values, title, signal_channels=None, cmap="RdBu_r", vsym=True, label=""):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    grid = _grid(values)
    vmax = np.nanmax(np.abs(grid)) if vsym else np.nanmax(grid)
    vmin = -vmax if vsym else np.nanmin(grid)
    im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax)
    if signal_channels is not None:
        n_rows, n_cols = GRID_SHAPE
        for c in signal_channels:
            ax.add_patch(plt.Rectangle((c % n_cols - 0.5, c // n_cols - 0.5), 1, 1,
                                        fill=False, edgecolor="black", linewidth=2))
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax)
    if label:
        cbar.set_label(label)
    fig.tight_layout()
    return fig


def plot_significance_comparison(signal_channels, naive_sig, bonf_sig, fdr_sig, maxstat_sig, n_channels):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    methods = ["naive\nuncorrected", "Bonferroni", "BH-FDR", "max-stat\npermutation"]
    sig_sets = [set(naive_sig.tolist()), set(bonf_sig.tolist()), set(fdr_sig.tolist()), set(maxstat_sig.tolist())]
    true_set = set(np.asarray(signal_channels).tolist())

    tp = [len(s & true_set) for s in sig_sets]
    fp = [len(s - true_set) for s in sig_sets]

    x = np.arange(len(methods))
    ax.bar(x, tp, color="#1b6ca8", label="true positives")
    ax.bar(x, fp, bottom=tp, color="#c1440e", label="false positives")
    ax.axhline(len(true_set), color="black", linestyle=":", linewidth=1, label=f"{len(true_set)} true signal channels")
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_ylabel("Number of channels flagged significant")
    ax.set_title("Discoveries by correction method")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_null_distribution(max_null, observed_thresh_channels_t, alpha=0.05):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(max_null, bins=40, color="#888888", alpha=0.8, label="max |t| across channels\n(permutation null)")
    thresh = np.percentile(max_null, 100 * (1 - alpha))
    ax.axvline(thresh, color="black", linewidth=1.5, label=f"{100*(1-alpha):.0f}th percentile = {thresh:.2f}")
    for t in observed_thresh_channels_t:
        ax.axvline(t, color="#1b6ca8", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("max |t| across all 64 channels")
    ax.set_ylabel("Permutation count")
    ax.set_title("Max-statistic permutation null distribution")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_sandwich_vs_naive(channel_ids, sandwich_var, naive_var):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(len(channel_ids))
    ax.scatter(x, sandwich_var, s=15, color="#1b6ca8", label="sandwich (true) variance")
    ax.scatter(x, naive_var, s=15, color="#c1440e", label="naive (i.i.d.) variance", marker="x")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Estimated Var(beta_hat)")
    ax.set_title("Sandwich vs. naive variance, per channel")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_discovery_replication_scatter(t_discovery, t_replication, discovery_sig, replication_sig, signal_channels):
    fig, ax = plt.subplots(figsize=(6, 6))
    n = len(t_discovery)
    colors = []
    for c in range(n):
        in_d = c in set(discovery_sig.tolist())
        in_r = c in set(replication_sig.tolist())
        if in_d and in_r:
            colors.append("#1b6ca8")
        elif in_d:
            colors.append("#c1440e")
        elif in_r:
            colors.append("#e8a33d")
        else:
            colors.append("#bbbbbb")
    ax.scatter(t_discovery, t_replication, c=colors, s=30)
    for c in signal_channels:
        ax.annotate(str(c), (t_discovery[c], t_replication[c]), fontsize=7, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Discovery-half t-statistic")
    ax.set_ylabel("Replication-half t-statistic")
    ax.set_title("Discovery vs. replication t-statistics, all channels")
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1b6ca8", label="sig. in both", markersize=8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#c1440e", label="discovery only", markersize=8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e8a33d", label="replication only", markersize=8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#bbbbbb", label="neither", markersize=8),
    ]
    ax.legend(handles=legend_elements, fontsize=8)
    fig.tight_layout()
    return fig
