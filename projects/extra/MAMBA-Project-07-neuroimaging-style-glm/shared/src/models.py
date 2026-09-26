"""
models.py
=========
Model implementation for MAMBA Project 07: Neuroimaging-Style GLM.

Scenario: a simulated multi-channel recording (64 channels, arranged on
an 8x8 grid -- think a simplified ECoG/EEG array or a small fMRI voxel
patch) during a block-randomized two-condition task. EVERY channel is
observed on the SAME set of trials with the SAME design matrix X
(intercept + condition dummy) -- this is "mass-univariate" analysis,
exactly how real fMRI/EEG/ECoG studies analyze multi-channel data: fit
the identical GLM independently to every channel's data column.

Trials are grouped into consecutive BLOCKS of fixed size, each block
assigned entirely to one condition (a block design, common in real
neuroimaging because many designs need each condition sustained for
multiple consecutive samples). Each channel's noise is an independent
AR(1) (first-order autoregressive) process across trials -- a standard,
realistic model of slow physiological/instrumental noise correlation in
neural time series. Only a small minority of channels carry a genuine
condition effect; the rest are exactly null.

Because the noise is autocorrelated AND the design is blocked (so
condition is correlated with local trial position), the ORDINARY
least-squares standard error formula -- which assumes independent,
identically distributed errors -- is not valid here. See the student
handout Section 3 for the exact "sandwich" variance formula this
implies, and shared/src/analysis.py for a permutation-based approach
that sidesteps needing to get the noise model exactly right at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


N_CHANNELS = 64
GRID_SHAPE = (8, 8)  # 64 = 8x8, for spatial visualization only


@dataclass
class BlockDesignParams:
    n_blocks: int
    block_size: int

    @property
    def n_trials(self) -> int:
        return self.n_blocks * self.block_size


@dataclass
class NoiseParams:
    phi: float     # AR(1) autocorrelation coefficient, same for every channel
    sigma: float   # marginal (stationary) noise SD, same for every channel


def make_block_labels(n_blocks: int, rng: np.random.Generator) -> np.ndarray:
    """A balanced random assignment of n_blocks blocks to condition 0 or 1
    (exactly n_blocks/2 of each), in random order. n_blocks must be even.
    """
    if n_blocks % 2 != 0:
        raise ValueError("n_blocks must be even for a balanced design")
    labels = np.array([0] * (n_blocks // 2) + [1] * (n_blocks // 2))
    rng.shuffle(labels)
    return labels


def expand_block_labels(block_labels: np.ndarray, block_size: int) -> np.ndarray:
    """Repeat each block label block_size times to get a per-trial condition array."""
    return np.repeat(block_labels, block_size)


def ar1_noise(n_trials: int, noise_params: NoiseParams, rng: np.random.Generator) -> np.ndarray:
    """Generate one length-n_trials AR(1) noise trajectory:

        e[0] ~ Normal(0, sigma^2)               (stationary initial draw)
        e[t] = phi*e[t-1] + sqrt(1-phi^2)*sigma*z[t],   z[t] ~ Normal(0,1)

    This parameterization keeps Var(e[t]) = sigma^2 for every t regardless
    of phi (a "stationary" AR(1) process), so sigma controls noise
    magnitude and phi controls autocorrelation independently.
    """
    z = rng.normal(0, 1, size=n_trials)
    e = np.zeros(n_trials)
    e[0] = z[0] * noise_params.sigma
    innovation_scale = np.sqrt(1 - noise_params.phi ** 2) * noise_params.sigma
    for t in range(1, n_trials):
        e[t] = noise_params.phi * e[t - 1] + innovation_scale * z[t]
    return e


def ar1_covariance_matrix(n: int, phi: float, sigma: float) -> np.ndarray:
    """The exact n x n stationary AR(1) covariance matrix:
    Sigma[i, j] = sigma^2 * phi^|i-j|.
    """
    idx = np.arange(n)
    exponent = np.abs(idx[:, None] - idx[None, :])
    return sigma ** 2 * phi ** exponent


def simulate_multichannel_session(design: BlockDesignParams, noise: NoiseParams,
                                   signal_channels: np.ndarray, true_effect: float,
                                   rng: np.random.Generator) -> dict:
    """Simulate one full multi-channel session. Returns a dict with:
    block_labels (n_blocks,), condition (n_trials,),
    Y (n_trials x N_CHANNELS), true_effect_by_channel (N_CHANNELS,).
    """
    block_labels = make_block_labels(design.n_blocks, rng)
    condition = expand_block_labels(block_labels, design.block_size)

    true_effect_by_channel = np.zeros(N_CHANNELS)
    true_effect_by_channel[signal_channels] = true_effect

    Y = np.zeros((design.n_trials, N_CHANNELS))
    for c in range(N_CHANNELS):
        Y[:, c] = true_effect_by_channel[c] * condition + ar1_noise(design.n_trials, noise, rng)

    return {
        "block_labels": block_labels,
        "condition": condition,
        "Y": Y,
        "true_effect_by_channel": true_effect_by_channel,
    }


def build_design_matrix(condition: np.ndarray) -> np.ndarray:
    """[intercept, condition] design matrix -- IDENTICAL across every
    channel (mass-univariate analysis: same X, different Y column)."""
    condition = np.asarray(condition, dtype=float)
    return np.column_stack([np.ones(len(condition)), condition])
