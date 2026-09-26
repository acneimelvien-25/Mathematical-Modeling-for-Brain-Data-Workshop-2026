"""
analysis.py
===========
Reference analysis pipeline for MAMBA Project 08: Coupled Populations.

Pipeline stages (see student handout Sections 2 and 5):

1. From FIT (subthreshold, noisy) time series at several coupling
   strengths w, estimate the effective self-decay rate and the
   effective coupling coefficient (w*g) via a discretized linear
   (VAR(1)-like) regression, pooling across nodes and time.
2. Pool the per-condition coupling-coefficient estimates across FIT w
   values into a single shared effective gain g_hat (through-the-origin
   regression, since the model predicts coupling_coefficient = w * g
   exactly).
3. Use g_hat and the closed-form circulant eigenvalue formulas
   (models.circulant_ring_eigenvalues) to predict the critical coupling
   strength, the mode that goes unstable first, and the oscillation
   frequency at onset.
4. Diagnose emergent oscillations in HOLDOUT time series: detect
   whether sustained oscillation is present, estimate its frequency,
   and estimate the phase lag between neighboring nodes.
5. Compare the symmetric-ring case (real eigenvalues only) against the
   directed-ring case.

Every public function takes plain arrays in and returns plain arrays/
dicts/floats out, so it can be unit-tested against data with KNOWN
parameters independently of the committed dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
from scipy.signal import hilbert


# ---------------------------------------------------------------------------
# Stage 1: per-condition self-decay / coupling-coefficient estimation
# ---------------------------------------------------------------------------

@dataclass
class LinearFit:
    self_decay: float      # estimated coefficient on x_i(t) (should be about -1/tau)
    coupling_coef: float   # estimated coefficient on x_{i-1}(t) (should be about w*g/tau)


def fit_local_linear_model(x: np.ndarray, dt_eff: float) -> LinearFit:
    """Fit dx_i/dt ~ a * x_i(t) + b * x_{i-1}(t) by ordinary least squares,
    pooling every node and every consecutive time step in x (shape
    (n_times, n_nodes)) into one regression. Assumes x is already
    subsampled to its saved effective time step dt_eff (the spacing
    between consecutive ROWS of x), and that node i's upstream neighbor
    is node (i-1) mod n_nodes (models.py's convention).
    """
    x_self = x[:-1]
    x_upstream = np.roll(x[:-1], 1, axis=1)
    dx = (x[1:] - x[:-1]) / dt_eff

    A = np.column_stack([x_self.flatten(), x_upstream.flatten()])
    y = dx.flatten()
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return LinearFit(self_decay=float(coef[0]), coupling_coef=float(coef[1]))


def pooled_gain_estimate(w_values: Sequence[float], coupling_coefs: Sequence[float]) -> float:
    """Through-the-origin regression of coupling_coef against w (the
    model predicts coupling_coef = w * g_true / tau exactly, so the
    correct second-stage fit forces a zero intercept). Returns g_hat
    (assuming tau = 1; divide by an independently-estimated tau if
    tau != 1 -- see Section 3 of the handout for why using -self_decay
    as a per-condition estimate of 1/tau first is the right order of
    operations here).
    """
    w = np.asarray(w_values, dtype=float)
    c = np.asarray(coupling_coefs, dtype=float)
    return float(np.sum(w * c) / np.sum(w * w))


# ---------------------------------------------------------------------------
# Stage 4: oscillation diagnostics on holdout data
# ---------------------------------------------------------------------------

def dominant_angular_frequency(x_node: np.ndarray, dt_eff: float) -> float:
    """Estimate a single node's dominant oscillation angular frequency
    via the peak of its (mean-removed) power spectrum.
    """
    centered = x_node - x_node.mean()
    freqs = np.fft.rfftfreq(len(centered), d=dt_eff)
    spectrum = np.abs(np.fft.rfft(centered))
    peak_idx = np.argmax(spectrum[1:]) + 1  # skip the DC (zero-frequency) bin
    return float(2 * np.pi * freqs[peak_idx])


def oscillation_amplitude(x_node: np.ndarray) -> float:
    return float(x_node.max() - x_node.min())


def instantaneous_phase(x_node: np.ndarray) -> np.ndarray:
    """Hilbert-transform-based instantaneous phase of a (mean-removed)
    oscillatory signal."""
    centered = x_node - x_node.mean()
    analytic = hilbert(centered)
    return np.angle(analytic)


def mean_phase_lag(x: np.ndarray) -> np.ndarray:
    """For each node i, the circular mean of (phase_i - phase_{i-1})
    across time, in RADIANS in (-pi, pi]. x has shape (n_times, n_nodes).
    """
    n_nodes = x.shape[1]
    phases = np.stack([instantaneous_phase(x[:, i]) for i in range(n_nodes)], axis=1)
    lags = np.zeros(n_nodes)
    for i in range(n_nodes):
        upstream = (i - 1) % n_nodes
        diff = np.angle(np.exp(1j * (phases[:, i] - phases[:, upstream])))
        lags[i] = np.angle(np.mean(np.exp(1j * diff)))  # circular mean
    return lags


# ---------------------------------------------------------------------------
# Stage 5: full pipeline summary helper
# ---------------------------------------------------------------------------

def predicted_bifurcation_summary(g_hat: float, n_nodes: int, tau: float, mode_k: int) -> Dict[str, float]:
    from models import RingParams, critical_coupling, onset_angular_frequency
    params = RingParams(n_nodes=n_nodes, tau=tau)
    w_crit = critical_coupling(g_hat, params, mode_k)
    freq = onset_angular_frequency(params, mode_k)
    phase_lag_deg = 360.0 * mode_k / n_nodes
    return {"critical_w": w_crit, "onset_angular_frequency": freq, "phase_lag_deg": phase_lag_deg}
