"""
models.py
=========
Model implementation for MAMBA Project 08: Coupled Populations.

Scenario: N identical rate-model populations (nodes) arranged on a
RING, each obeying

    tau * dx_i/dt = -x_i + w * S(x_{i-1}) [+ noise]

where S is a sigmoidal gain function (tanh here) and node i receives
input from exactly ONE upstream neighbor, i-1 (indices mod N) -- a
unidirectional (directed) ring, the discrete analogue of a cyclic
negative-feedback loop (e.g. a "repressilator"-style architecture, or a
simplified central-pattern-generator circuit). With S = tanh and no
external bias, x* = 0 is an exact fixed point for every node,
regardless of w or N, since tanh(0) = 0 -- this is what makes the
linear stability analysis below exact and clean.

Linearizing around x* = 0, the Jacobian is

    J = (-1/tau) I + (w*g/tau) P

where g = S'(0) = 1 (exactly, for tanh) and P is the CYCLIC PERMUTATION
matrix (P_{i, i-1 mod N} = 1, zero elsewhere) -- i.e. J is a CIRCULANT
matrix. Circulant matrices have an exact, closed-form eigen-decomposition
via the discrete Fourier modes: see ``circulant_ring_eigenvalues`` and
the student handout Section 3 for the derivation.

``build_symmetric_ring_dynamics`` implements the contrasting case where
each node is coupled to BOTH neighbors (an undirected/symmetric ring) --
a genuinely different coupling matrix (symmetric, hence guaranteed real
eigenvalues by the spectral theorem) used for the required comparison
in the handout (Task T5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RingParams:
    n_nodes: int
    tau: float


def circulant_ring_eigenvalues(w: float, g: float, params: RingParams) -> np.ndarray:
    """Exact eigenvalues of the DIRECTED (unidirectional) ring's Jacobian,
    for every Fourier mode k = 0, ..., N-1:

        lambda_k = (-1 + w*g*exp(2*pi*i*k/N)) / tau

    Returned in order of increasing k.
    """
    N = params.n_nodes
    k = np.arange(N)
    omega_k = np.exp(2j * np.pi * k / N)
    return (-1.0 + w * g * omega_k) / params.tau


def symmetric_ring_eigenvalues(w: float, g: float, params: RingParams) -> np.ndarray:
    """Exact eigenvalues of the corresponding SYMMETRIC (bidirectional,
    each neighbor weighted w/2) ring's Jacobian:

        lambda_k = (-1 + w*g*cos(2*pi*k/N)) / tau

    These are ALWAYS real (the coupling matrix is symmetric), unlike the
    directed ring's generally-complex eigenvalues.
    """
    N = params.n_nodes
    k = np.arange(N)
    return (-1.0 + w * g * np.cos(2 * np.pi * k / N)) / params.tau


def critical_coupling(g: float, params: RingParams, mode_k: int) -> float:
    """The coupling strength w at which Re(lambda_k) first crosses zero
    for the DIRECTED ring, given mode_k. Solves
    -1 + w*g*cos(2*pi*mode_k/N) = 0 for w.
    """
    N = params.n_nodes
    theta = 2 * np.pi * mode_k / N
    return 1.0 / (g * np.cos(theta))


def onset_angular_frequency(params: RingParams, mode_k: int) -> float:
    """The oscillation angular frequency at the moment mode_k's real part
    crosses zero -- exactly |tan(2*pi*mode_k/N)| / tau, INDEPENDENT of g
    and of the critical coupling's exact value (the g-dependence cancels;
    see handout Section 3.4 for why). A good independently-checkable fact.
    """
    N = params.n_nodes
    theta = 2 * np.pi * mode_k / N
    return abs(np.tan(theta)) / params.tau


def most_unstable_mode(g: float, params: RingParams) -> int:
    """Which Fourier mode k in {0, ..., N-1} has the LARGEST real part of
    lambda_k for a given sign of g (assumes w will be chosen with sign
    opposite to g if g>0, i.e. this returns the mode that goes unstable
    FIRST as |w| grows, for inhibitory closed-loop coupling). For this
    project's directed ring, computed by simply maximizing -cos(theta_k)
    over k (the relevant case, inhibitory feedback around the loop).

    Modes k and N-k are exact mathematical conjugates (identical real
    part, opposite-signed imaginary part) and so are tied in exact
    arithmetic -- floating-point rounding can otherwise break that tie
    unpredictably in either direction, so ties (within a small tolerance)
    are explicitly broken in favor of the SMALLER k for a deterministic,
    reproducible choice of which of the two conjugate modes to report.
    """
    N = params.n_nodes
    k = np.arange(N)
    theta = 2 * np.pi * k / N
    values = -np.cos(theta)
    best = np.max(values)
    tied = np.where(values >= best - 1e-9)[0]
    return int(tied.min())


def simulate_directed_ring(w: float, params: RingParams, T: float, dt: float,
                            sigma: float = 0.0, rng: np.random.Generator = None,
                            x0: np.ndarray = None) -> np.ndarray:
    """Euler-Maruyama simulation of the full NONLINEAR directed-ring
    system. Returns an (n_steps, n_nodes) array. sigma=0 gives a
    deterministic (noise-free) simulation.
    """
    N = params.n_nodes
    n_steps = int(round(T / dt))
    x = np.zeros((n_steps, N))
    x[0] = x0 if x0 is not None else np.zeros(N)
    sqdt = np.sqrt(dt)
    for t in range(1, n_steps):
        upstream = np.roll(x[t - 1], 1)
        dx = (-x[t - 1] + w * np.tanh(upstream)) / params.tau
        noise = sigma * sqdt * rng.standard_normal(N) if sigma > 0 else 0.0
        x[t] = x[t - 1] + dx * dt + noise
    return x


def simulate_symmetric_ring(w: float, params: RingParams, T: float, dt: float,
                             sigma: float = 0.0, rng: np.random.Generator = None,
                             x0: np.ndarray = None) -> np.ndarray:
    """Euler-Maruyama simulation of the full NONLINEAR symmetric-ring
    (bidirectional, each neighbor weighted w/2) system."""
    N = params.n_nodes
    n_steps = int(round(T / dt))
    x = np.zeros((n_steps, N))
    x[0] = x0 if x0 is not None else np.zeros(N)
    sqdt = np.sqrt(dt)
    for t in range(1, n_steps):
        left = np.roll(x[t - 1], 1)
        right = np.roll(x[t - 1], -1)
        dx = (-x[t - 1] + (w / 2) * np.tanh(left) + (w / 2) * np.tanh(right)) / params.tau
        noise = sigma * sqdt * rng.standard_normal(N) if sigma > 0 else 0.0
        x[t] = x[t - 1] + dx * dt + noise
    return x
