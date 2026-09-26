"""
models.py
=========
Neuron model implementations for MAMBA Project 01: Membrane to Spike.

This module contains two model classes:

1. ``HodgkinHuxleyNeuron`` — the biophysical *reference* model used only to
   generate the synthetic observational data (``generate_data.py``). Students
   do not need to reimplement this class; it is documented here for
   instructor transparency and for the unit tests that check internal
   consistency of the data-generation pipeline. It is intentionally NOT
   imported by any of the student-facing analysis scaffolding.

2. ``LeakyIntegrateFire`` — the candidate model students derive, implement,
   and fit against the observed data. This is the model family named in the
   student handout.

All voltages are in millivolts (mV), all times in milliseconds (ms), all
currents in microamps per square centimetre (uA/cm^2), all conductances in
millisiemens per square centimetre (mS/cm^2), and all capacitances in
microfarads per square centimetre (uF/cm^2). Using the classic Hodgkin-Huxley
(1952) unit convention throughout keeps every quantity dimensionally
consistent (see ``student/exercise_handout.md``, Section 3, for a full
dimensional-analysis walkthrough).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# 1. Hodgkin-Huxley reference model (data-generation only)
# ---------------------------------------------------------------------------

@dataclass
class HodgkinHuxleyParams:
    """Classic squid giant axon parameters (Hodgkin & Huxley, 1952)."""

    C_m: float = 1.0        # uF/cm^2
    g_Na: float = 120.0     # mS/cm^2
    g_K: float = 36.0       # mS/cm^2
    g_L: float = 0.3        # mS/cm^2
    E_Na: float = 50.0      # mV
    E_K: float = -77.0      # mV
    E_L: float = -54.387    # mV


def _safe_exp_ratio(numerator_coeff: float, v_shift: np.ndarray, denom: float) -> np.ndarray:
    """Evaluate x / (1 - exp(-x/denom)) with the removable singularity at x=0
    handled by a first-order Taylor expansion, needed because several HH rate
    functions have this form and V can pass exactly through the singular
    point during numerical integration."""
    x = v_shift
    small = np.abs(x) < 1e-6
    with np.errstate(divide="ignore", invalid="ignore"):
        out = numerator_coeff * x / (1.0 - np.exp(-x / denom))
    # Taylor expansion of x/(1-exp(-x/denom)) around x=0 is `denom` (L'Hopital)
    out = np.where(small, numerator_coeff * denom, out)
    return out


class HodgkinHuxleyNeuron:
    """Biophysical reference neuron used to synthesize the workshop data.

    Gating kinetics use the standard alpha/beta rate functions in the
    absolute-voltage convention (Dayan & Abbott, 2001, Ch. 5):

        alpha_n(V) = 0.01 (V+55) / (1 - exp(-(V+55)/10))
        beta_n(V)  = 0.125 exp(-(V+65)/80)
        alpha_m(V) = 0.1 (V+40) / (1 - exp(-(V+40)/10))
        beta_m(V)  = 4 exp(-(V+65)/18)
        alpha_h(V) = 0.07 exp(-(V+65)/20)
        beta_h(V)  = 1 / (1 + exp(-(V+35)/10))

    and the membrane equation is

        C_m dV/dt = I(t) - g_Na m^3 h (V - E_Na) - g_K n^4 (V - E_K) - g_L (V - E_L)
        dm/dt = alpha_m(V)(1-m) - beta_m(V) m      (similarly for h, n)
    """

    def __init__(self, params: Optional[HodgkinHuxleyParams] = None):
        self.p = params or HodgkinHuxleyParams()

    # -- gating rate functions -------------------------------------------------
    @staticmethod
    def alpha_n(V):
        return _safe_exp_ratio(0.01, V + 55.0, 10.0)

    @staticmethod
    def beta_n(V):
        return 0.125 * np.exp(-(V + 65.0) / 80.0)

    @staticmethod
    def alpha_m(V):
        return _safe_exp_ratio(0.1, V + 40.0, 10.0)

    @staticmethod
    def beta_m(V):
        return 4.0 * np.exp(-(V + 65.0) / 18.0)

    @staticmethod
    def alpha_h(V):
        return 0.07 * np.exp(-(V + 65.0) / 20.0)

    @staticmethod
    def beta_h(V):
        return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

    def resting_state(self) -> tuple[float, float, float, float]:
        """Find the resting equilibrium (V0, m0, h0, n0) at I=0 by integrating
        from a plausible starting point until the state stops changing. Used
        both to initialise simulations and, in ``analysis.py``, to compute
        the ground-truth effective (tau_m, R_input) used for grading.
        """
        V, m, h, n = -65.0, 0.05, 0.6, 0.32
        dt = 0.01
        for _ in range(200_000):  # 2000 ms settling time
            dV, dm, dh, dn = self._derivatives(V, m, h, n, I=0.0)
            V += dt * dV
            m += dt * dm
            h += dt * dh
            n += dt * dn
        return V, m, h, n

    def _derivatives(self, V, m, h, n, I):
        p = self.p
        I_Na = p.g_Na * (m ** 3) * h * (V - p.E_Na)
        I_K = p.g_K * (n ** 4) * (V - p.E_K)
        I_L = p.g_L * (V - p.E_L)
        dV = (I - I_Na - I_K - I_L) / p.C_m
        dm = self.alpha_m(V) * (1 - m) - self.beta_m(V) * m
        dh = self.alpha_h(V) * (1 - h) - self.beta_h(V) * h
        dn = self.alpha_n(V) * (1 - n) - self.beta_n(V) * n
        return dV, dm, dh, dn

    def simulate(
        self,
        I_of_t: Callable[[float], float],
        t_end: float,
        dt: float = 0.01,
        record_every: int = 10,
        init_state: Optional[tuple[float, float, float, float]] = None,
    ):
        """Integrate the HH system with classical RK4 at internal step ``dt``,
        recording every ``record_every`` steps (so the recorded time step is
        ``dt * record_every``).

        Returns
        -------
        t_rec : (N,) array, ms
        V_rec : (N,) array, mV
        """
        if init_state is None:
            V, m, h, n = self.resting_state()
        else:
            V, m, h, n = init_state

        n_steps = int(round(t_end / dt))
        t_rec, V_rec = [], []

        def deriv_vec(state, t):
            Vv, mv, hv, nv = state
            I = I_of_t(t)
            dV, dm, dh, dn = self._derivatives(Vv, mv, hv, nv, I)
            return np.array([dV, dm, dh, dn])

        state = np.array([V, m, h, n])
        for step in range(n_steps + 1):
            t = step * dt
            if step % record_every == 0:
                t_rec.append(t)
                V_rec.append(state[0])
            if step == n_steps:
                break
            k1 = deriv_vec(state, t)
            k2 = deriv_vec(state + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = deriv_vec(state + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = deriv_vec(state + dt * k3, t + dt)
            state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        return np.asarray(t_rec), np.asarray(V_rec)


# ---------------------------------------------------------------------------
# 2. Leaky integrate-and-fire model (the model family students fit)
# ---------------------------------------------------------------------------

@dataclass
class LIFParams:
    """Parameters of the leaky integrate-and-fire model.

    tau_m       : membrane time constant, ms
    R           : input resistance, MOhm-equivalent in these units
                  (mV per uA/cm^2, i.e. mV*cm^2/uA)
    E_L         : resting/leak potential, mV
    V_th        : spike threshold, mV
    V_reset     : post-spike reset potential, mV
    t_ref       : absolute refractory period, ms
    """

    tau_m: float
    R: float
    E_L: float
    V_th: float
    V_reset: float
    t_ref: float


class LeakyIntegrateFire:
    """Leaky integrate-and-fire neuron.

    Governing ODE (see handout Section 3 for the RC-circuit derivation from
    Kirchhoff's current law):

        tau_m dV/dt = -(V - E_L) + R * I(t),   for V < V_th

    with an instantaneous reset V -> V_reset and an absolute refractory
    period t_ref whenever V crosses V_th from below.
    """

    def __init__(self, params: LIFParams):
        self.p = params

    def simulate(self, I_of_t: Callable[[float], float], t_end: float, dt: float,
                 V0: Optional[float] = None):
        """Euler-integrate the LIF model. Returns (t, V, spike_times)."""
        p = self.p
        V = p.E_L if V0 is None else V0
        n_steps = int(round(t_end / dt))
        t = np.arange(n_steps + 1) * dt
        V_trace = np.empty(n_steps + 1)
        spikes = []
        refractory_until = -np.inf

        for i, ti in enumerate(t):
            V_trace[i] = V
            if ti < refractory_until:
                V = p.V_reset
                continue
            I = I_of_t(ti)
            dV = (-(V - p.E_L) + p.R * I) / p.tau_m
            V = V + dt * dV
            if V >= p.V_th:
                spikes.append(ti)
                V = p.V_reset
                refractory_until = ti + p.t_ref

        return t, V_trace, np.array(spikes)

    @staticmethod
    def analytic_firing_rate(I: np.ndarray, tau_m: float, R: float, E_L: float,
                              V_th: float, V_reset: float, t_ref: float) -> np.ndarray:
        """Closed-form steady-state firing rate of the LIF model for constant
        input current(s) ``I`` (array or scalar), derived by integrating the
        subthreshold ODE from V_reset to V_th (handout Section 4 walks
        through this derivation in full):

            f(I) = 0                                            if R*I <= (V_th - E_L)
            f(I) = [ t_ref + tau_m * ln( (R*I - (V_reset-E_L)) /
                                          (R*I - (V_th-E_L)) ) ]^-1   otherwise

        This is the "nontrivial calculation that can be checked independently
        of the final software output" required by the project design
        constraints: it can be evaluated with a calculator given fitted
        parameters and compared directly against the empirical spike counts.
        """
        I = np.atleast_1d(np.asarray(I, dtype=float))
        rate = np.zeros_like(I)
        rheobase_drive = V_th - E_L
        drive = R * I
        can_spike = drive > rheobase_drive
        num = drive[can_spike] - (V_reset - E_L)
        den = drive[can_spike] - (V_th - E_L)
        isi = t_ref + tau_m * np.log(num / den)
        rate[can_spike] = 1000.0 / isi  # ms -> spikes/second
        return rate
