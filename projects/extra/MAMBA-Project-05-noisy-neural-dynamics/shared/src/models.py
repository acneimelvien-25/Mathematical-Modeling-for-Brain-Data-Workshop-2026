"""
models.py
=========
Model implementation for MAMBA Project 04: Noisy Neural Dynamics.

The generative model is a single passive (non-spiking) membrane compartment
whose voltage follows a leaky (Ornstein-Uhlenbeck) process:

    tau dV/dt = -(V - V_ss) + xi(t)

where ``xi(t)`` is white noise representing fluctuating synaptic/background
input, so that in Euler-Maruyama form:

    dV = -(V - V_ss)/tau dt + sigma_p dW

``tau`` is the membrane time constant (ms), ``V_ss`` is the mean/steady-state
voltage set by the (unknown) mean input drive, and ``sigma_p`` is the process
noise amplitude (mV per sqrt(ms)) set by the (unknown) input fluctuation
strength. This is the same passive membrane equation used in Project 01,
but WITHOUT any threshold/spiking mechanism (V never resets -- there is no
"integrate-and-FIRE" here, only "integrate"), and observed at IRREGULAR
time intervals with additional measurement noise on top.

``OUProcess`` provides the EXACT discrete-time solution for an arbitrary
time step (not just a fixed grid), which is the correct way to simulate or
reason about this process at irregular sample times -- there is no need to
resort to small-step Euler-Maruyama integration when the exact transition
distribution is known in closed form.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OUParams:
    tau: float       # membrane time constant, ms
    V_ss: float      # steady-state (mean) voltage, mV
    sigma_p: float   # process noise amplitude, mV / sqrt(ms)


class OUProcess:
    """A scalar Ornstein-Uhlenbeck process, used both as the (hidden)
    generative model for the workshop's data and as the model family
    participants are asked to fit.

    Stationary distribution: Normal(V_ss, sigma_p^2 * tau / 2).

    Exact transition over an elapsed time ``dt`` (arbitrary, not
    necessarily small or constant): given V(t), the distribution of
    V(t+dt) is

        Normal( V_ss + (V(t)-V_ss) * exp(-dt/tau),
                (sigma_p^2 * tau / 2) * (1 - exp(-2 dt / tau)) )

    This is the unique EXACT solution of the linear SDE above (not an
    approximation), and is what makes irregular sampling tractable: each
    consecutive pair of observations, however far apart, has a known,
    exact conditional distribution given its own specific elapsed time.
    """

    def __init__(self, params: OUParams):
        self.p = params

    @property
    def stationary_variance(self) -> float:
        return self.p.sigma_p ** 2 * self.p.tau / 2.0

    def transition_mean(self, V_prev: np.ndarray, dt: np.ndarray) -> np.ndarray:
        p = self.p
        return p.V_ss + (V_prev - p.V_ss) * np.exp(-dt / p.tau)

    def transition_variance(self, dt: np.ndarray) -> np.ndarray:
        p = self.p
        return self.stationary_variance * (1.0 - np.exp(-2.0 * dt / p.tau))

    def simulate_at_times(self, times: np.ndarray, rng: np.random.Generator,
                            init: float = None) -> np.ndarray:
        """Simulate the EXACT process at an arbitrary (e.g. irregularly
        spaced) sequence of times, using the closed-form transition above
        at each successive gap. The first sample is drawn from the
        stationary distribution unless ``init`` is given."""
        n = len(times)
        V = np.empty(n)
        V[0] = init if init is not None else rng.normal(self.p.V_ss, np.sqrt(self.stationary_variance))
        for i in range(1, n):
            dt = times[i] - times[i - 1]
            mean = self.transition_mean(V[i - 1], dt)
            var = self.transition_variance(dt)
            V[i] = rng.normal(mean, np.sqrt(max(var, 0.0)))
        return V


# ---------------------------------------------------------------------------
# The variogram (structure function): the key analysis primitive
# ---------------------------------------------------------------------------

def variogram_formula(dt: np.ndarray, tau: float, sigma_p2_tau: float, obs_2sigma2: float) -> np.ndarray:
    """Predicted Var[V_obs(t+dt) - V_obs(t)] for the OU process plus
    independent observation noise, as a function of the elapsed time dt.

    Parametrized by ``sigma_p2_tau = sigma_p**2 * tau`` (the natural
    combination that appears in the formula) and
    ``obs_2sigma2 = 2 * sigma_obs**2``, rather than by sigma_p and
    sigma_obs directly, because these two combinations are what the
    variogram's shape actually constrains -- fitting in these units keeps
    the optimization well-conditioned. Recover sigma_p and sigma_obs from
    the fitted values with sigma_p = sqrt(sigma_p2_tau / tau) and
    sigma_obs = sqrt(obs_2sigma2 / 2) once tau itself is known.

    Derive this formula yourself (handout Section 3.3) before using it --
    it follows from the transition mean/variance above, plus the fact that
    observation noise adds independently at both endpoints of the
    difference.
    """
    return obs_2sigma2 + sigma_p2_tau * (1.0 - np.exp(-dt / tau))
