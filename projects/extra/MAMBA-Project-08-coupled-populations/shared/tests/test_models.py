"""Unit tests for models.py: circulant eigenvalue formulas, critical
coupling / onset frequency derivations, and ring simulation."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models import (  # noqa: E402
    RingParams, circulant_ring_eigenvalues, symmetric_ring_eigenvalues,
    critical_coupling, onset_angular_frequency, most_unstable_mode,
    simulate_directed_ring, simulate_symmetric_ring,
)


def test_circulant_eigenvalues_match_direct_matrix_eigendecomposition():
    """Cross-check the closed-form circulant formula against a brute-force
    numpy eigendecomposition of the literal Jacobian matrix, for several
    (N, w, g) combinations."""
    for N in [3, 4, 5, 7]:
        for w in [-0.5, 0.7, -1.5]:
            params = RingParams(n_nodes=N, tau=1.0)
            g = 1.0
            J = np.zeros((N, N))
            for i in range(N):
                J[i, i] = -1.0
                J[i, (i - 1) % N] = w * g
            direct_eigs = np.sort_complex(np.linalg.eigvals(J))
            formula_eigs = np.sort_complex(circulant_ring_eigenvalues(w, g, params))
            np.testing.assert_allclose(direct_eigs, formula_eigs, atol=1e-8)


def test_symmetric_eigenvalues_are_real_and_match_direct_matrix():
    for N in [4, 5, 6]:
        for w in [-1.0, 1.2]:
            params = RingParams(n_nodes=N, tau=1.0)
            g = 1.0
            J = np.zeros((N, N))
            for i in range(N):
                J[i, i] = -1.0
                J[i, (i - 1) % N] = w * g / 2
                J[i, (i + 1) % N] = w * g / 2
            direct_eigs = np.sort(np.linalg.eigvals(J).real)
            formula_eigs = np.sort(symmetric_ring_eigenvalues(w, g, params).real)
            np.testing.assert_allclose(direct_eigs, formula_eigs, atol=1e-8)
            # confirm they are (numerically) exactly real
            direct_eigs_complex = np.linalg.eigvals(J)
            assert np.max(np.abs(direct_eigs_complex.imag)) < 1e-10


def test_critical_coupling_gives_exactly_zero_real_part():
    params = RingParams(n_nodes=5, tau=1.0)
    g = 1.0
    mode_k = most_unstable_mode(g, params)
    w_crit = critical_coupling(g, params, mode_k)
    eigs = circulant_ring_eigenvalues(w_crit, g, params)
    assert eigs[mode_k].real == pytest.approx(0.0, abs=1e-10)


def test_onset_frequency_matches_hand_formula_for_N5():
    """N=5, mode k=2: theta=4*pi/5, onset freq = |tan(4pi/5)| ~ 0.7265."""
    params = RingParams(n_nodes=5, tau=1.0)
    freq = onset_angular_frequency(params, mode_k=2)
    assert freq == pytest.approx(0.72654, abs=1e-4)


def test_onset_frequency_independent_of_gain():
    """The onset frequency should not depend on g at all (only on N, mode_k, tau)."""
    params = RingParams(n_nodes=5, tau=1.0)
    freq = onset_angular_frequency(params, mode_k=2)
    # recompute a different way: at critical coupling for THIS g, check Im/onset matches
    for g in [0.5, 1.0, 2.0, 5.0]:
        w_crit = critical_coupling(g, params, mode_k=2)
        eigs = circulant_ring_eigenvalues(w_crit, g, params)
        assert abs(eigs[2].imag) == pytest.approx(freq, abs=1e-8)


def test_most_unstable_mode_is_deterministic_for_odd_N():
    """For N=5, modes k=2 and k=3 are exact conjugates (tied); the
    function must deterministically return the smaller of the two."""
    params = RingParams(n_nodes=5, tau=1.0)
    mode_k = most_unstable_mode(1.0, params)
    assert mode_k == 2


def test_most_unstable_mode_for_even_N_is_the_real_nyquist_mode():
    """For even N, the most destabilizing mode under inhibitory feedback
    is k=N/2 exactly (a real eigenvalue, the alternating pattern)."""
    params = RingParams(n_nodes=6, tau=1.0)
    mode_k = most_unstable_mode(1.0, params)
    assert mode_k == 3  # N/2
    eigs = circulant_ring_eigenvalues(critical_coupling(1.0, params, 3), 1.0, params)
    assert abs(eigs[3].imag) < 1e-10  # real eigenvalue, not oscillatory


def test_simulate_directed_ring_shape_and_zero_fixed_point():
    params = RingParams(n_nodes=5, tau=1.0)
    x = simulate_directed_ring(w=-0.5, params=params, T=10.0, dt=0.01, sigma=0.0,
                                x0=0.5 * np.ones(5))
    assert x.shape == (1000, 5)
    # subthreshold, should decay toward the x*=0 fixed point
    assert np.abs(x[-1]).max() < 0.05


def test_simulate_directed_ring_subthreshold_decays_to_zero():
    rng = np.random.default_rng(0)
    params = RingParams(n_nodes=5, tau=1.0)
    x = simulate_directed_ring(w=-0.8, params=params, T=50.0, dt=0.01, sigma=0.0,
                                x0=rng.normal(size=5))
    assert np.abs(x[-1]).max() < 1e-6


def test_simulate_directed_ring_superthreshold_sustains_oscillation():
    params = RingParams(n_nodes=5, tau=1.0)
    x = simulate_directed_ring(w=-2.0, params=params, T=300.0, dt=0.01, sigma=0.0,
                                x0=np.array([0.1, -0.05, 0.02, 0.0, -0.03]))
    tail = x[int(len(x) * 0.7):]
    amplitude = tail.max() - tail.min()
    assert amplitude > 1.0  # genuine sustained oscillation, not decayed to zero
    assert np.all(np.isfinite(tail))  # bounded, not diverging


def test_simulate_symmetric_ring_superthreshold_reaches_static_pattern():
    params = RingParams(n_nodes=5, tau=1.0)
    x = simulate_symmetric_ring(w=-1.5, params=params, T=200.0, dt=0.01, sigma=0.0,
                                 x0=np.array([0.1, -0.05, 0.02, 0.0, -0.03]))
    tail = x[int(len(x) * 0.8):]
    temporal_std = tail.std(axis=0)
    assert np.all(temporal_std < 1e-3)  # converged to a STATIC pattern, not oscillating
    spatial_spread = tail[-1].max() - tail[-1].min()
    assert spatial_spread > 0.1  # but a genuinely non-uniform spatial pattern
