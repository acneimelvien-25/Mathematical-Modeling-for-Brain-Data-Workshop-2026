"""
plotting.py
===========
Figure-generation helpers for MAMBA Project 03: Representational Geometry.

Every function returns the created ``matplotlib.figure.Figure``; no
function calls ``plt.show()``.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis import (stimulus_features, stimulus_roles, trial_averaged_responses,
                       fit_stimulus_ids, holdout_stimulus_ids)
from models import pca_project


def plot_variance_explained(dim_report):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.bar(np.arange(1, 16), dim_report.variance_ratio[:15], color="tab:blue")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Fraction of variance explained")
    ax.set_title("Variance explained per component")

    ax = axes[1]
    ax.plot(np.arange(1, 21), dim_report.cumulative_variance[:20], "o-", color="tab:blue")
    ax.axhline(0.90, color="gray", linestyle=":", label="90%")
    ax.axhline(0.95, color="gray", linestyle="--", label="95%")
    ax.axvline(2, color="tab:red", linestyle="--", label="true stimulus dimensionality (2)")
    ax.axvline(dim_report.n_components_for_90pct, color="tab:blue", linestyle=":",
               label=f"{dim_report.n_components_for_90pct} PCs for 90%")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative variance explained")
    ax.set_title("Cumulative variance vs. true dimensionality")
    ax.legend(fontsize=7)

    fig.tight_layout()
    return fig


def plot_pc_projection(data, dim_report, stimulus_ids=None, pcs=(0, 1)):
    """Scatter of stimuli projected onto two chosen PCs, colored by their
    TRUE feature_x value -- a visual check of how (or whether) the
    dominant PCA axes track the true stimulus structure."""
    if stimulus_ids is None:
        stimulus_ids = fit_stimulus_ids(data)
    R = trial_averaged_responses(data, stimulus_ids)
    scores = pca_project(dim_report.basis, R, max(pcs) + 1)
    feats = stimulus_features(data)
    fx = np.array([feats[sid][0] for sid in stimulus_ids])

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sc = ax.scatter(scores[:, pcs[0]], scores[:, pcs[1]], c=fx, cmap="coolwarm", s=60)
    fig.colorbar(sc, ax=ax, label="true feature_x")
    ax.set_xlabel(f"PC{pcs[0]+1}")
    ax.set_ylabel(f"PC{pcs[1]+1}")
    ax.set_title(f"Stimuli projected onto PC{pcs[0]+1}/PC{pcs[1]+1}")
    fig.tight_layout()
    return fig


def plot_geometry_scatter(geometry_report, title="Representational geometry"):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(geometry_report.stimulus_distances, geometry_report.neural_distances,
               s=8, alpha=0.4, color="tab:blue")
    ax.set_xlabel("Stimulus-feature distance (true)")
    ax.set_ylabel("Neural response distance")
    ax.set_title(f"{title}\n(Pearson r={geometry_report.pearson_r:.3f}, "
                 f"Spearman r={geometry_report.spearman_r:.3f})")
    fig.tight_layout()
    return fig


def plot_cv_curve(cv_curve, chosen_k, fit_error=None, holdout_error=None):
    ks = sorted(cv_curve.keys())
    errs = [cv_curve[k] for k in ks]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(ks, errs, "-", color="tab:blue", label="cross-validated error (within fit set)")
    ax.axvline(chosen_k, color="tab:red", linestyle="--", label=f"chosen k={chosen_k}")
    if fit_error is not None:
        ax.scatter([chosen_k], [fit_error], color="black", zorder=5, label="final fit error")
    if holdout_error is not None:
        ax.scatter([chosen_k], [holdout_error], color="tab:orange", marker="^", zorder=5,
                   label="final holdout error")
    ax.set_xlabel("Number of principal components used (k)")
    ax.set_ylabel("Mean decoding error (Euclidean, feature units)")
    ax.set_title("Choosing k: bias-variance trade-off")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_decoded_vs_true(data, decoder, decode_fn, stimulus_ids=None):
    """Scatter of decoded (x, y) vs. true (x, y), fit and holdout stimuli
    distinguished, for a visual sense of decoder accuracy and any
    systematic bias."""
    from analysis import trials_for_stimulus
    fit_ids = fit_stimulus_ids(data)
    holdout_ids = holdout_stimulus_ids(data)
    feats = stimulus_features(data)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for ids, color, marker, label in [(fit_ids, "black", "o", "fit"),
                                        (holdout_ids, "tab:red", "^", "holdout")]:
        for sid in ids:
            counts, _ = trials_for_stimulus(data, sid)
            preds = decode_fn(decoder, counts)
            true_xy = np.array(feats[sid])
            ax.scatter(preds[:, 0], preds[:, 1], color=color, marker=marker, alpha=0.25, s=15)
            ax.scatter([true_xy[0]], [true_xy[1]], color=color, marker="x", s=80, linewidths=2)
    ax.scatter([], [], color="black", marker="o", label="fit stimulus (predictions)")
    ax.scatter([], [], color="tab:red", marker="^", label="holdout stimulus (predictions)")
    ax.scatter([], [], color="gray", marker="x", label="true stimulus location")
    ax.set_xlabel("Decoded feature_x")
    ax.set_ylabel("Decoded feature_y")
    ax.set_title("Decoded vs. true stimulus location")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig
