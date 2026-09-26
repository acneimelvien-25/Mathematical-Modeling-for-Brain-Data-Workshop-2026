"""
plotting.py
===========
Figure-generation helpers for MAMBA Project 06: Stimulus-to-Response GLM.

Every function returns the created ``matplotlib.figure.Figure``; no
function calls ``plt.show()``.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


NAIVE_COLOR = "#c1440e"
FULL_COLOR = "#1b6ca8"
FIT_COLOR = "#1b6ca8"
HOLDOUT_COLOR = "#c1440e"


def plot_response_vs_trial(trial_index, response, trial_role, drift_true=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    trial_index = np.asarray(trial_index)
    response = np.asarray(response)
    is_fit = np.asarray(trial_role) == "fit"
    ax.scatter(trial_index[is_fit], response[is_fit], s=8, color=FIT_COLOR, alpha=0.5, label="fit")
    ax.scatter(trial_index[~is_fit], response[~is_fit], s=8, color=HOLDOUT_COLOR, alpha=0.6, label="holdout")
    if drift_true is not None:
        ax.plot(trial_index, drift_true, color="black", linewidth=1.2, alpha=0.6,
                label="true drift component (instructor only)")
    ax.set_xlabel("Trial index (session time)")
    ax.set_ylabel("Response (a.u.)")
    ax.set_title("Raw response across the session")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def plot_coefficient_comparison(names, true_beta, beta_naive, beta_full, se_naive=None, se_full=None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(names))
    width = 0.3
    ax.bar(x - width, true_beta, width, label="true", color="#888888")
    ax.bar(x, beta_naive, width, yerr=se_naive, label="naive (no drift)", color=NAIVE_COLOR, capsize=3)
    ax.bar(x + width, beta_full, width, yerr=se_full, label="full (with drift)", color=FULL_COLOR, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Coefficient estimate")
    ax.set_title("Fitted coefficients: naive vs. full model")
    ax.legend(fontsize=8)
    ax.axhline(0, color="black", linewidth=0.6)
    fig.tight_layout()
    return fig


def plot_bias_prediction_vs_observed(names, predicted_bias, observed_bias):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, predicted_bias, width, label="predicted (formula)", color="#555555")
    ax.bar(x + width / 2, observed_bias, width, label="observed (naive fit - true)", color=NAIVE_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Bias")
    ax.set_title("Omitted-variable bias: formula vs. observed")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_residual_autocorrelation(lags, ac_naive, ac_full):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    lags = list(lags)
    x = np.arange(len(lags))
    width = 0.35
    ax.bar(x - width / 2, [ac_naive[l] for l in lags], width, label="naive (no drift)", color=NAIVE_COLOR)
    ax.bar(x + width / 2, [ac_full[l] for l in lags], width, label="full (with drift)", color=FULL_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in lags])
    ax.set_xlabel("Lag (trials)")
    ax.set_ylabel("Residual autocorrelation")
    ax.set_title("Residual autocorrelation: naive vs. full model")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_residuals_vs_trial(trial_index, resid_naive, resid_full):
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].scatter(trial_index, resid_naive, s=8, color=NAIVE_COLOR, alpha=0.6)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Naive model residuals vs. trial index")
    axes[1].scatter(trial_index, resid_full, s=8, color=FULL_COLOR, alpha=0.6)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Residual")
    axes[1].set_xlabel("Trial index (session time)")
    axes[1].set_title("Full model residuals vs. trial index")
    fig.tight_layout()
    return fig


def plot_holdout_prediction(trial_index_holdout, y_true, pred_naive, pred_full):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    order = np.argsort(trial_index_holdout)
    ti = np.asarray(trial_index_holdout)[order]
    ax.plot(ti, np.asarray(y_true)[order], "o", color="black", markersize=4, label="observed")
    ax.plot(ti, np.asarray(pred_naive)[order], "-", color=NAIVE_COLOR, linewidth=1.5, label="naive prediction")
    ax.plot(ti, np.asarray(pred_full)[order], "-", color=FULL_COLOR, linewidth=1.5, label="full prediction")
    ax.set_xlabel("Trial index (holdout trials, session time)")
    ax.set_ylabel("Response (a.u.)")
    ax.set_title("Holdout prediction: naive vs. full model")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_contrast_forest(contrast_names, estimates_naive, ci_naive, estimates_full, ci_full):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    y = np.arange(len(contrast_names))
    for i, name in enumerate(contrast_names):
        ax.errorbar(estimates_naive[i], y[i] + 0.15,
                    xerr=[[estimates_naive[i] - ci_naive[i][0]], [ci_naive[i][1] - estimates_naive[i]]],
                    fmt="o", color=NAIVE_COLOR, capsize=3, label="naive" if i == 0 else None)
        ax.errorbar(estimates_full[i], y[i] - 0.15,
                    xerr=[[estimates_full[i] - ci_full[i][0]], [ci_full[i][1] - estimates_full[i]]],
                    fmt="o", color=FULL_COLOR, capsize=3, label="full" if i == 0 else None)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(contrast_names)
    ax.set_xlabel("Contrast estimate (with 95% CI)")
    ax.set_title("Contrasts: naive vs. full model")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
