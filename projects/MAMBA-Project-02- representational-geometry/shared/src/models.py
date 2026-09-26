"""
models.py
=========
Model implementation for MAMBA Project 03: Representational Geometry.

Two things live here:

1. ``GaussianBumpPopulation`` -- the biophysically-motivated (but hidden
   from students) generative model used only to synthesize the workshop's
   dataset (see ``generate_data.py``). Each simulated neuron has a
   Gaussian ("bump") tuning curve over a 2-D stimulus feature space,
   centered at a random preferred location -- a standard, well-established
   model of feature-tuned cortical neurons (used for orientation tuning,
   place/grid-cell-like spatial tuning, and many other feature-tuned
   population models).

2. A small set of plain linear-algebra PRIMITIVES (``fit_pca``,
   ``pca_project``, ``pca_reconstruct``) that participants are expected to
   understand and could reimplement themselves from the derivation in the
   handout -- they are provided here mainly so the test suite and the
   instructor reference pipeline have one unambiguous, checked
   implementation to compare against, not because participants are meant
   to treat PCA as a black box.

All stimulus feature coordinates are dimensionless, in [-1, 1] (per
feature axis). Neural responses are modeled as trial-level spike COUNTS
over a fixed integration window (see ``generate_data.py``), so they are
non-negative integers with a Poisson-like mean-variance relationship --
consistent with the workshop's probability curriculum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# 1. Hidden generative model (data-generation only)
# ---------------------------------------------------------------------------

@dataclass
class GaussianBumpPopulationParams:
    preferred_xy: np.ndarray   # (M, 2) each neuron's preferred stimulus location
    sigma: np.ndarray          # (M,) each neuron's tuning width
    gain: np.ndarray           # (M,) each neuron's peak response amplitude (Hz)
    baseline: np.ndarray       # (M,) each neuron's baseline response (Hz)


class GaussianBumpPopulation:
    """A population of M neurons, each with a 2-D Gaussian tuning curve:

        rate_m(x, y) = baseline_m + gain_m * exp(-d_m^2 / (2 sigma_m^2))
        d_m^2 = (x - x_m)^2 + (y - y_m)^2

    where (x_m, y_m) is neuron m's preferred stimulus location. This is a
    standard "bump" tuning-curve model. Because each neuron's response is
    a NONLINEAR (Gaussian) function of the 2 stimulus features, the
    resulting population response manifold is not, in general, capturable
    by a 2-dimensional LINEAR subspace even though the stimulus itself has
    only 2 true degrees of freedom -- this is the source of the
    "intrinsic vs. linear dimensionality" gap the project is built around.
    """

    def __init__(self, params: GaussianBumpPopulationParams):
        self.p = params

    @property
    def n_neurons(self) -> int:
        return self.p.preferred_xy.shape[0]

    def mean_rate(self, xy: np.ndarray) -> np.ndarray:
        """Mean firing rate (Hz) of every neuron for a batch of stimuli.

        Parameters
        ----------
        xy : (N, 2) array of stimulus feature coordinates

        Returns
        -------
        (N, M) array of mean rates
        """
        p = self.p
        d2 = ((xy[:, None, :] - p.preferred_xy[None, :, :]) ** 2).sum(axis=2)
        return p.baseline[None, :] + p.gain[None, :] * np.exp(-d2 / (2 * p.sigma[None, :] ** 2))

    def sample_counts(self, xy: np.ndarray, n_trials: int, integration_time_s: float,
                       rng: np.random.Generator) -> np.ndarray:
        """Sample trial-level Poisson spike counts.

        Returns
        -------
        (N, n_trials, M) array of non-negative integer spike counts.
        """
        rates = self.mean_rate(xy)  # (N, M)
        lam = rates[:, None, :] * integration_time_s  # (N, 1, M) broadcast
        lam = np.broadcast_to(lam, (rates.shape[0], n_trials, rates.shape[1]))
        return rng.poisson(lam)


# ---------------------------------------------------------------------------
# 2. Linear-algebra primitives (PCA)
# ---------------------------------------------------------------------------

@dataclass
class PCABasis:
    mean: np.ndarray        # (M,) the mean response vector subtracted before projecting
    components: np.ndarray  # (M, M) columns are principal axes, sorted by decreasing variance
    variance: np.ndarray    # (M,) variance explained by each component (eigenvalues of the covariance)


def fit_pca(X: np.ndarray) -> PCABasis:
    """Fit a PCA basis to a (n_samples, n_features) data matrix via the
    eigendecomposition of its sample covariance matrix (mean-subtracted
    first). Returns components sorted by decreasing explained variance.

    This is deliberately implemented via ``np.linalg.eigh`` on the
    covariance matrix (rather than via SVD of the data matrix directly)
    because that is the derivation given in the handout -- the two are
    mathematically equivalent for this purpose, and
    ``shared/tests/test_models.py`` checks that this implementation agrees
    with an independent SVD-based computation to numerical precision.
    """
    mean = X.mean(axis=0)
    Xc = X - mean
    n = Xc.shape[0]
    cov = (Xc.T @ Xc) / (n - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending order
    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order], 0.0, None)
    eigvecs = eigvecs[:, order]
    return PCABasis(mean=mean, components=eigvecs, variance=eigvals)


def variance_explained_ratio(basis: PCABasis) -> np.ndarray:
    total = basis.variance.sum()
    return basis.variance / total if total > 0 else np.zeros_like(basis.variance)


def pca_project(basis: PCABasis, X: np.ndarray, k: int) -> np.ndarray:
    """Project (n_samples, n_features) data onto the top-k components."""
    return (X - basis.mean) @ basis.components[:, :k]


def pca_reconstruct(basis: PCABasis, scores: np.ndarray, k: int) -> np.ndarray:
    """Invert a top-k projection back into the original feature space."""
    return scores @ basis.components[:, :k].T + basis.mean
