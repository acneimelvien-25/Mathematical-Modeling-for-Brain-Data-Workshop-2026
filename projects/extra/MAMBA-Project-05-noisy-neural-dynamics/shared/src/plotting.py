"""
plotting.py
===========
Figure-generation helpers for MAMBA Project 04: Noisy Neural Dynamics.

Every function returns the created ``matplotlib.figure.Figure``; no
function calls ``plt.show()``.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis import session_data, session_roles, pooled_variogram
from models import variogram_formula


def plot_example_session(data, session_id: int, window=(0, 500)):
    times, V = session_data(data, session_id)
    mask = (times >= window[0]) & (times <= window[1])
    role = session_roles(data)[session_id]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(times[mask], V[mask], "o-", color="black", markersize=3, linewidth=0.8)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("$V_{obs}$ (mV)")
    ax.set_title(f"Session {session_id} ({role}) -- note the irregular sample spacing")
    fig.tight_layout()
    return fig


def plot_per_session_identifiability(per_session_fits: dict, session_mean_dt: dict, true_tau=None):
    """Scatter of each session's INDEPENDENTLY fitted tau against that
    session's mean sampling interval -- the core identifiability-diagnosis
    figure."""
    sids = sorted(per_session_fits.keys())
    mean_dts = [session_mean_dt[s] for s in sids]
    taus = [per_session_fits[s].tau for s in sids]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(mean_dts, taus, s=70, color="tab:blue", zorder=5)
    for s, x, y in zip(sids, mean_dts, taus):
        ax.annotate(f"session {s}", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
    if true_tau is not None:
        ax.axhline(true_tau, color="tab:red", linestyle="--", label=f"true $\\tau$={true_tau}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Session mean sampling interval (ms, log scale)")
    ax.set_ylabel("Independently fitted $\\tau$ (ms, log scale)")
    ax.set_title("Identifiability of $\\tau$ vs. sampling density")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_pooled_variogram_fit(data, fit_ids, pooled_fit, max_lag=200.0, n_bins=25):
    vgram = pooled_variogram(data, fit_ids, max_lag=max_lag, n_bins=n_bins)
    dt_line = np.geomspace(vgram.dt_centers.min(), max_lag, 200)
    predicted = variogram_formula(dt_line, pooled_fit.tau, pooled_fit.sigma_p ** 2 * pooled_fit.tau,
                                    2 * pooled_fit.sigma_obs ** 2)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(vgram.dt_centers, vgram.values, s=np.clip(vgram.counts / 20, 5, 100),
               color="black", alpha=0.6, label="empirical (pooled across fit sessions)")
    ax.plot(dt_line, predicted, "-", color="tab:blue",
            label=f"fitted model ($\\tau$={pooled_fit.tau:.1f} ms)")
    ax.set_xscale("log")
    ax.set_xlabel("Elapsed time $\\Delta t$ (ms, log scale)")
    ax.set_ylabel("Var[$V(t+\\Delta t) - V(t)$] (mV$^2$)")
    ax.set_title("Pooled variogram fit (fit sessions only)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_holdout_validation(validation_report, session_roles_map=None):
    n = len(validation_report.per_session)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5))
    if n == 1:
        axes = [axes]
    for ax, (sid, d) in zip(axes, sorted(validation_report.per_session.items())):
        ax.scatter(d["dt_centers"], d["observed"], s=np.clip(d["counts"] / 5, 5, 100),
                   color="black", alpha=0.6, label="observed (this session)")
        ax.plot(d["dt_centers"], d["predicted"], "-", color="tab:red",
                label="predicted (pooled fit-session parameters)")
        ax.set_xscale("log")
        ax.set_xlabel("Elapsed time $\\Delta t$ (ms, log scale)")
        ax.set_ylabel("Var[$V(t+\\Delta t)-V(t)$] (mV$^2$)")
        ax.set_title(f"Session {sid} (mean rel. err={d['mean_relative_error']:.2f})")
        ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_bootstrap_distributions(draws):
    keys = list(draws.keys())
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4))
    for ax, k in zip(axes, keys):
        ax.hist(draws[k], bins=25, color="tab:blue", alpha=0.8)
        ax.set_title(k)
        ax.set_xlabel(k)
        ax.set_ylabel("count")
    fig.suptitle("Bootstrap distributions of pooled parameters (resampled within sessions)")
    fig.tight_layout()
    return fig
