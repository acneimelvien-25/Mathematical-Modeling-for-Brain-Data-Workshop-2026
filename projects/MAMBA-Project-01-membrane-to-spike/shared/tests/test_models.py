"""Unit tests for models.py: the LIF ODE, its analytic firing rate, and
internal consistency of the HH reference model used only for data
generation."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models import HodgkinHuxleyNeuron, LeakyIntegrateFire, LIFParams  # noqa: E402


def test_lif_subthreshold_relaxation_matches_analytic_solution():
    """With a constant sub-rheobase current, the LIF membrane potential
    should relax to E_L + R*I with the correct analytic time course,
    V(t) = Vss - (Vss - E_L) exp(-t/tau_m). This checks the Euler
    integrator against the closed-form solution for the linear ODE."""
    p = LIFParams(tau_m=10.0, R=5.0, E_L=-65.0, V_th=-50.0, V_reset=-65.0, t_ref=2.0)
    lif = LeakyIntegrateFire(p)
    I_dc = 1.0  # subthreshold: R*I=5 < V_th-E_L=15
    dt = 0.001  # small dt so Euler error is negligible for the tolerance used
    t, V, spikes = lif.simulate(lambda tt: I_dc, t_end=80.0, dt=dt)
    assert len(spikes) == 0
    Vss = p.E_L + p.R * I_dc
    analytic = Vss - (Vss - p.E_L) * np.exp(-t / p.tau_m)
    assert np.max(np.abs(V - analytic)) < 0.05


def test_lif_spikes_above_rheobase_and_not_below():
    p = LIFParams(tau_m=10.0, R=5.0, E_L=-65.0, V_th=-50.0, V_reset=-65.0, t_ref=2.0)
    lif = LeakyIntegrateFire(p)
    rheobase = (p.V_th - p.E_L) / p.R  # = 3.0
    t, V, spikes_below = lif.simulate(lambda tt: 0.9 * rheobase, t_end=200.0, dt=0.01)
    t, V, spikes_above = lif.simulate(lambda tt: 1.5 * rheobase, t_end=200.0, dt=0.01)
    assert len(spikes_below) == 0
    assert len(spikes_above) > 0


def test_lif_refractory_period_is_respected():
    """No two recorded spikes should be closer together than t_ref."""
    p = LIFParams(tau_m=8.0, R=6.0, E_L=-65.0, V_th=-50.0, V_reset=-70.0, t_ref=3.0)
    lif = LeakyIntegrateFire(p)
    t, V, spikes = lif.simulate(lambda tt: 10.0, t_end=300.0, dt=0.01)
    assert len(spikes) > 3
    isis = np.diff(spikes)
    assert np.all(isis >= p.t_ref - 1e-9)


def test_analytic_firing_rate_matches_brute_force_simulation():
    """The nontrivial, independently-checkable calculation: the closed-form
    steady-state rate should match a brute-force numeric simulation to
    within a couple of percent for a comfortably-suprathreshold current."""
    p = LIFParams(tau_m=10.0, R=5.0, E_L=-65.0, V_th=-50.0, V_reset=-65.0, t_ref=2.0)
    lif = LeakyIntegrateFire(p)
    I_dc = 6.0  # rheobase is 3.0, so this is comfortably suprathreshold
    t, V, spikes = lif.simulate(lambda tt: I_dc, t_end=2000.0, dt=0.005)
    empirical_rate_hz = 1000.0 * len(spikes) / t[-1]
    analytic_rate_hz = LeakyIntegrateFire.analytic_firing_rate(
        np.array([I_dc]), p.tau_m, p.R, p.E_L, p.V_th, p.V_reset, p.t_ref)[0]
    assert empirical_rate_hz == pytest.approx(analytic_rate_hz, rel=0.03)


def test_analytic_firing_rate_is_zero_below_rheobase():
    p = LIFParams(tau_m=10.0, R=5.0, E_L=-65.0, V_th=-50.0, V_reset=-65.0, t_ref=2.0)
    rheobase = (p.V_th - p.E_L) / p.R
    rate = LeakyIntegrateFire.analytic_firing_rate(
        np.array([0.5 * rheobase]), p.tau_m, p.R, p.E_L, p.V_th, p.V_reset, p.t_ref)
    assert rate[0] == 0.0


def test_analytic_firing_rate_approaches_1_over_tref_at_high_current():
    """As I -> infinity, isi -> t_ref (the drive term dominates both the
    numerator and denominator of the log ratio, which -> ln(1) = 0)."""
    p = LIFParams(tau_m=10.0, R=5.0, E_L=-65.0, V_th=-50.0, V_reset=-65.0, t_ref=2.0)
    rate = LeakyIntegrateFire.analytic_firing_rate(
        np.array([1e7]), p.tau_m, p.R, p.E_L, p.V_th, p.V_reset, p.t_ref)
    assert rate[0] == pytest.approx(1000.0 / p.t_ref, rel=1e-3)


def test_hh_resting_state_is_a_true_fixed_point():
    """The resting state returned by HodgkinHuxleyNeuron.resting_state()
    should have (approximately) zero derivative -- i.e. it should actually
    be an equilibrium of the full 4-variable system at I=0."""
    hh = HodgkinHuxleyNeuron()
    V0, m0, h0, n0 = hh.resting_state()
    dV, dm, dh, dn = hh._derivatives(V0, m0, h0, n0, I=0.0)
    assert abs(dV) < 1e-3
    assert abs(dm) < 1e-5
    assert abs(dh) < 1e-5
    assert abs(dn) < 1e-5


def test_hh_voltage_stays_in_physiological_range():
    """A sanity check on the integrator: even for a suprathreshold current,
    the HH membrane potential should stay within a physiologically
    reasonable envelope (no blow-up from an unstable integration step)."""
    hh = HodgkinHuxleyNeuron()
    rest = hh.resting_state()
    t, V = hh.simulate(lambda tt: 10.0, t_end=50.0, dt=0.01, record_every=10, init_state=rest)
    assert V.min() > -100.0
    assert V.max() < 60.0
