"""
models.py
=========
Model implementation for MAMBA Project 02: Population Rhythms.

A single class, ``WilsonCowanPopulation``, implements a two-population
(excitatory E / inhibitory I) rate model of the Wilson-Cowan family:

    tau_E dE/dt = -E + S(w_EE E - w_EI I + I_E; a_E, theta_E)
    tau_I dI/dt = -I + S(w_IE E - w_II I + I_I; a_I, theta_I)

    S(x; a, theta) = 1 / (1 + exp(-a (x - theta)))

E and I are dimensionless population activity variables (interpretable as
the fraction of each population active per unit time, following Wilson &
Cowan 1972); tau_E, tau_I are population time constants (ms); w_EE, w_EI,
w_IE, w_II >= 0 are coupling weights; a_* are sigmoid gains and theta_* are
sigmoid thresholds (in the same units as the net input); I_E, I_I are
external inputs to each population.

This same class is used both to generate the workshop's synthetic dataset
(with hidden ground-truth parameters, see ``generate_data.py``) and as the
model family participants are asked to derive, linearize, and fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.optimize import fsolve


def sigmoid(x, a: float, theta: float):
    """Logistic response function S(x; a, theta) = 1/(1+exp(-a(x-theta)))."""
    return 1.0 / (1.0 + np.exp(-a * (x - theta)))


def sigmoid_derivative_from_value(s, a: float):
    """dS/dx expressed in terms of the ALREADY-EVALUATED value s=S(x;a,theta):
    S' = a * s * (1-s). Useful because the Jacobian only ever needs S' at a
    point where S has already been evaluated."""
    return a * s * (1.0 - s)


@dataclass
class WilsonCowanParams:
    tau_E: float
    tau_I: float
    w_EE: float
    w_EI: float
    w_IE: float
    w_II: float
    a_E: float
    theta_E: float
    a_I: float
    theta_I: float
    I_E: float  # fixed background drive to the E population


class WilsonCowanPopulation:
    """Two-population Wilson-Cowan-style rate model.

    The external drive to the inhibitory population, ``I_I``, is treated
    as the experiment's control variable (analogous to injected current in
    a single-neuron current-clamp experiment) and is passed to each method
    separately rather than being stored on ``self`` -- every other
    parameter in ``WilsonCowanParams`` is fixed for a given "preparation."
    """

    def __init__(self, params: WilsonCowanParams):
        self.p = params

    # -- right-hand side -----------------------------------------------------
    def _derivatives(self, E, I, I_I):
        p = self.p
        u_E = p.w_EE * E - p.w_EI * I + p.I_E
        u_I = p.w_IE * E - p.w_II * I + I_I
        S_E = sigmoid(u_E, p.a_E, p.theta_E)
        S_I = sigmoid(u_I, p.a_I, p.theta_I)
        dE = (-E + S_E) / p.tau_E
        dI = (-I + S_I) / p.tau_I
        return dE, dI

    # -- fixed point -----------------------------------------------------------
    def fixed_point(self, I_I: float, init=(0.3, 0.3)):
        """Solve for the equilibrium (E*, I*) at a given I_I via Newton's
        method (``scipy.optimize.fsolve``). The fixed-point condition is

            E* = S(w_EE E* - w_EI I* + I_E; a_E, theta_E)
            I* = S(w_IE E* - w_II I* + I_I; a_I, theta_I)

        which is generally transcendental (no closed form) because S is a
        sigmoid -- this is why a numerical root-finder, not algebra, is the
        right tool here.
        """
        p = self.p

        def eqs(x):
            E, I = x
            S_E = sigmoid(p.w_EE * E - p.w_EI * I + p.I_E, p.a_E, p.theta_E)
            S_I = sigmoid(p.w_IE * E - p.w_II * I + I_I, p.a_I, p.theta_I)
            return [S_E - E, S_I - I]

        sol = fsolve(eqs, init, full_output=False)
        return float(sol[0]), float(sol[1])

    # -- Jacobian --------------------------------------------------------------
    def jacobian(self, E: float, I: float, I_I: float) -> np.ndarray:
        """Analytic Jacobian of the deterministic system at the point (E,I),
        for the given I_I. Derived by differentiating the right-hand side:

            dE/dt = f_E(E,I) = (-E + S(u_E))/tau_E,   u_E = w_EE E - w_EI I + I_E
            dI/dt = f_I(E,I) = (-I + S(u_I))/tau_I,   u_I = w_IE E - w_II I + I_I

            J = [[df_E/dE, df_E/dI], [df_I/dE, df_I/dI]]
              = [[(-1 + w_EE S_E')/tau_E,  -w_EI S_E'/tau_E],
                 [ w_IE S_I'/tau_I,        (-1 - w_II S_I')/tau_I]]

        where S_E' and S_I' are the sigmoid derivatives evaluated at (E,I).
        """
        p = self.p
        u_E = p.w_EE * E - p.w_EI * I + p.I_E
        u_I = p.w_IE * E - p.w_II * I + I_I
        S_E = sigmoid(u_E, p.a_E, p.theta_E)
        S_I = sigmoid(u_I, p.a_I, p.theta_I)
        S_E_prime = sigmoid_derivative_from_value(S_E, p.a_E)
        S_I_prime = sigmoid_derivative_from_value(S_I, p.a_I)
        J = np.array([
            [(-1.0 + p.w_EE * S_E_prime) / p.tau_E, (-p.w_EI * S_E_prime) / p.tau_E],
            [(p.w_IE * S_I_prime) / p.tau_I, (-1.0 - p.w_II * S_I_prime) / p.tau_I],
        ])
        return J

    # -- deterministic simulation (RK4) ----------------------------------------
    def simulate_deterministic(self, I_I: float, t_end: float, dt: float,
                                init=(0.05, 0.1)):
        """RK4 integration of the noise-free ODE. Returns (t, E, I)."""
        n_steps = int(round(t_end / dt))
        t = np.arange(n_steps + 1) * dt
        E = np.empty(n_steps + 1)
        I = np.empty(n_steps + 1)
        E[0], I[0] = init

        def deriv(Ev, Iv):
            return self._derivatives(Ev, Iv, I_I)

        for k in range(n_steps):
            k1E, k1I = deriv(E[k], I[k])
            k2E, k2I = deriv(E[k] + 0.5 * dt * k1E, I[k] + 0.5 * dt * k1I)
            k3E, k3I = deriv(E[k] + 0.5 * dt * k2E, I[k] + 0.5 * dt * k2I)
            k4E, k4I = deriv(E[k] + dt * k3E, I[k] + dt * k3I)
            E[k + 1] = E[k] + (dt / 6.0) * (k1E + 2 * k2E + 2 * k3E + k4E)
            I[k + 1] = I[k] + (dt / 6.0) * (k1I + 2 * k2I + 2 * k3I + k4I)
        return t, E, I

    # -- stochastic simulation (Euler-Maruyama) --------------------------------
    def simulate_stochastic(self, I_I: float, t_end: float, dt: float,
                             sigma: float, rng: np.random.Generator,
                             init=(0.05, 0.1)):
        """Euler-Maruyama integration of the stochastic system

            dE = f_E(E,I) dt + sigma dW_E
            dI = f_I(E,I) dt + sigma dW_I

        i.e. independent additive white noise of amplitude ``sigma`` on
        each population, representing intrinsic (finite-size /
        channel-type) population-activity fluctuations rather than
        measurement noise (measurement noise is added separately -- see
        ``generate_data.py``). This is the standard Euler-Maruyama update:
        each step adds a deterministic RK-style drift plus a Gaussian
        increment scaled by sqrt(dt) (NOT dt) as required for a Wiener
        process. Returns (t, E, I).
        """
        n_steps = int(round(t_end / dt))
        t = np.arange(n_steps + 1) * dt
        E = np.empty(n_steps + 1)
        I = np.empty(n_steps + 1)
        E[0], I[0] = init
        sqrt_dt = np.sqrt(dt)

        for k in range(n_steps):
            dE, dI = self._derivatives(E[k], I[k], I_I)
            E[k + 1] = E[k] + dt * dE + sigma * sqrt_dt * rng.normal()
            I[k + 1] = I[k] + dt * dI + sigma * sqrt_dt * rng.normal()
        return t, E, I
