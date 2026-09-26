"""
analysis.py
===========
Reference analysis pipeline for MAMBA Project 03: Representational
Geometry.

Implements one complete, defensible path from the raw observations in
``shared/data/`` to: (1) a PCA basis fit on FIT-stimulus trial-averaged
responses and its variance-explained curve, (2) a representational
geometry comparison (neural distances vs. true stimulus-feature
distances), (3) a cross-validated choice of how many principal components
to decode from, (4) a linear decoder trained on FIT-stimulus single-trial
data, and (5) a validation report on the HELD-OUT stimuli, which are never
touched by any of the above.

Participants are NOT required to follow this exact path -- see the
handout's "what is intentionally left for the group to decide" section.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr

from models import PCABasis, fit_pca, variance_explained_ratio, pca_project


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(data_dir: str):
    npz = np.load(os.path.join(data_dir, "population_responses.npz"))
    # NOTE: np.genfromtxt's automatic string-width inference is not always
    # reliable (it can truncate a column to the width of an early value,
    # e.g. "holdout" -> "hol") -- use csv.DictReader instead for the one
    # column ('role') that holds variable-length strings.
    import csv as _csv
    with open(os.path.join(data_dir, "stimulus_grid.csv"), newline="") as f:
        rows = list(_csv.DictReader(f))
    grid = [
        {"stimulus_id": int(r["stimulus_id"]), "feature_x": float(r["feature_x"]),
         "feature_y": float(r["feature_y"]), "n_trials": int(r["n_trials"]),
         "integration_time_s": float(r["integration_time_s"]), "role": r["role"]}
        for r in rows
    ]
    return {
        "spike_counts": npz["spike_counts"], "stimulus_id": npz["stimulus_id"],
        "trial": npz["trial"], "grid": grid,
    }


def stimulus_features(data) -> dict:
    """Map stimulus_id -> (feature_x, feature_y)."""
    return {int(row["stimulus_id"]): (float(row["feature_x"]), float(row["feature_y"]))
            for row in data["grid"]}


def stimulus_roles(data) -> dict:
    return {int(row["stimulus_id"]): str(row["role"]) for row in data["grid"]}


def fit_stimulus_ids(data) -> np.ndarray:
    roles = stimulus_roles(data)
    return np.array(sorted(cid for cid, r in roles.items() if r == "fit"))


def holdout_stimulus_ids(data) -> np.ndarray:
    roles = stimulus_roles(data)
    return np.array(sorted(cid for cid, r in roles.items() if r == "holdout"))


def trials_for_stimulus(data, stimulus_id: int, trial_subset=None):
    """Return the (n_trials, M) spike-count sub-matrix for one stimulus."""
    mask = data["stimulus_id"] == stimulus_id
    trials_here = data["trial"][mask]
    counts_here = data["spike_counts"][mask]
    if trial_subset is not None:
        trial_to_row = {int(t): i for i, t in enumerate(trials_here)}
        order = [trial_to_row[int(t)] for t in trial_subset]
        return counts_here[order], np.asarray(trial_subset)
    return counts_here, trials_here


def trial_averaged_responses(data, stimulus_ids: np.ndarray, trial_subset=None) -> np.ndarray:
    """(len(stimulus_ids), M) matrix of trial-averaged spike counts."""
    rows = []
    for sid in stimulus_ids:
        counts, _ = trials_for_stimulus(data, sid, trial_subset)
        rows.append(counts.mean(axis=0))
    return np.stack(rows, axis=0)


# ---------------------------------------------------------------------------
# Dimensionality analysis
# ---------------------------------------------------------------------------

@dataclass
class DimensionalityReport:
    basis: PCABasis
    variance_ratio: np.ndarray
    cumulative_variance: np.ndarray
    n_components_for_90pct: int
    n_components_for_95pct: int


def analyze_dimensionality(data, fit_ids: np.ndarray) -> DimensionalityReport:
    R_fit = trial_averaged_responses(data, fit_ids)
    basis = fit_pca(R_fit)
    ratio = variance_explained_ratio(basis)
    cumvar = np.cumsum(ratio)
    n90 = int(np.searchsorted(cumvar, 0.90) + 1)
    n95 = int(np.searchsorted(cumvar, 0.95) + 1)
    return DimensionalityReport(basis=basis, variance_ratio=ratio, cumulative_variance=cumvar,
                                 n_components_for_90pct=n90, n_components_for_95pct=n95)


# ---------------------------------------------------------------------------
# Representational geometry (RSA-style)
# ---------------------------------------------------------------------------

@dataclass
class GeometryReport:
    neural_distances: np.ndarray
    stimulus_distances: np.ndarray
    pearson_r: float
    spearman_r: float


def analyze_geometry(data, stimulus_ids: np.ndarray) -> GeometryReport:
    """Compare pairwise Euclidean distances in NEURAL response space
    (trial-averaged) against pairwise Euclidean distances in the TRUE
    stimulus feature space, across every pair of the given stimuli. A
    high correlation means the population preserves the relative
    geometry of the stimulus space (up to a monotonic transform, for the
    Spearman version); it does not require the neural code to be an
    isometry (equal absolute distances)."""
    R = trial_averaged_responses(data, stimulus_ids)
    feats = stimulus_features(data)
    xy = np.array([feats[sid] for sid in stimulus_ids])

    neural_d = pdist(R, metric="euclidean")
    stim_d = pdist(xy, metric="euclidean")
    r_p, _ = pearsonr(neural_d, stim_d)
    r_s, _ = spearmanr(neural_d, stim_d)
    return GeometryReport(neural_distances=neural_d, stimulus_distances=stim_d,
                           pearson_r=float(r_p), spearman_r=float(r_s))


# ---------------------------------------------------------------------------
# Decoder: cross-validated choice of k, then train + evaluate
# ---------------------------------------------------------------------------

@dataclass
class Decoder:
    basis: PCABasis
    k: int
    weights: np.ndarray  # (k+1, 2): row 0 is intercept


def _design_matrix(scores: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(scores.shape[0]), scores])


def train_decoder(data, fit_ids: np.ndarray, basis: PCABasis, k: int) -> Decoder:
    """Train a linear (ordinary least squares) decoder from the top-k PCA
    scores of SINGLE-TRIAL responses to the true 2-D stimulus features,
    using every trial of every FIT stimulus."""
    feats = stimulus_features(data)
    X_rows, Y_rows = [], []
    for sid in fit_ids:
        counts, trial_ids = trials_for_stimulus(data, sid)
        scores = pca_project(basis, counts, k)
        X_rows.append(scores)
        Y_rows.append(np.tile(feats[sid], (counts.shape[0], 1)))
    X = np.concatenate(X_rows, axis=0)
    Y = np.concatenate(Y_rows, axis=0)
    Xd = _design_matrix(X)
    W, *_ = np.linalg.lstsq(Xd, Y, rcond=None)
    return Decoder(basis=basis, k=k, weights=W)


def decode(decoder: Decoder, counts: np.ndarray) -> np.ndarray:
    """Predict (x, y) for a batch of single-trial count vectors, shape (n, M)."""
    scores = pca_project(decoder.basis, counts, decoder.k)
    Xd = _design_matrix(scores)
    return Xd @ decoder.weights


def mean_decoding_error(data, decoder: Decoder, stimulus_ids: np.ndarray) -> float:
    feats = stimulus_features(data)
    errors = []
    for sid in stimulus_ids:
        counts, _ = trials_for_stimulus(data, sid)
        preds = decode(decoder, counts)
        true_xy = np.array(feats[sid])
        errors.extend(np.linalg.norm(preds - true_xy[None, :], axis=1).tolist())
    return float(np.mean(errors))


def cross_validate_k(data, fit_ids: np.ndarray, k_values, n_folds: int = 5, seed: int = 0):
    """Choose the number of principal components to decode from via
    trial-level K-fold cross-validation WITHIN the fit stimuli (never
    touching holdout stimuli). For each fold, the PCA basis itself is
    refit on that fold's training trials only (trial-averaged), so the
    cross-validation honestly reflects the whole pipeline, not just the
    final regression step.

    Returns a dict k -> mean cross-validated decoding error.
    """
    rng = np.random.default_rng(seed)
    n_trials = int(data["grid"][0]["n_trials"])
    fold_assignment = rng.permutation(n_trials) % n_folds

    results = {k: [] for k in k_values}
    for fold in range(n_folds):
        train_trials = np.where(fold_assignment != fold)[0]
        val_trials = np.where(fold_assignment == fold)[0]

        R_train = trial_averaged_responses(data, fit_ids, trial_subset=train_trials)
        basis = fit_pca(R_train)

        for k in k_values:
            # train decoder on this fold's training trials
            feats = stimulus_features(data)
            X_rows, Y_rows = [], []
            for sid in fit_ids:
                counts, _ = trials_for_stimulus(data, sid, trial_subset=train_trials)
                X_rows.append(pca_project(basis, counts, k))
                Y_rows.append(np.tile(feats[sid], (counts.shape[0], 1)))
            X = np.concatenate(X_rows, axis=0)
            Y = np.concatenate(Y_rows, axis=0)
            W, *_ = np.linalg.lstsq(_design_matrix(X), Y, rcond=None)

            # evaluate on this fold's held-out trials (still FIT stimuli --
            # this is a within-fit-set cross-validation, distinct from the
            # project's held-out STIMULI validation)
            errs = []
            for sid in fit_ids:
                counts, _ = trials_for_stimulus(data, sid, trial_subset=val_trials)
                scores = pca_project(basis, counts, k)
                preds = _design_matrix(scores) @ W
                true_xy = np.array(feats[sid])
                errs.extend(np.linalg.norm(preds - true_xy[None, :], axis=1).tolist())
            results[k].append(np.mean(errs))

    return {k: float(np.mean(v)) for k, v in results.items()}


# ---------------------------------------------------------------------------
# Full validation report
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    chosen_k: int
    cv_curve: dict
    fit_error: float
    holdout_error: float
    fit_geometry: GeometryReport
    holdout_geometry: GeometryReport


def full_pipeline(data, k_values=range(1, 41), n_folds: int = 5, seed: int = 0) -> tuple:
    """Run the complete reference pipeline: dimensionality analysis,
    cross-validated k selection, decoder training, and holdout
    validation. Returns (DimensionalityReport, ValidationReport, Decoder)."""
    fit_ids = fit_stimulus_ids(data)
    holdout_ids = holdout_stimulus_ids(data)

    dim_report = analyze_dimensionality(data, fit_ids)

    cv_curve = cross_validate_k(data, fit_ids, k_values, n_folds=n_folds, seed=seed)
    chosen_k = min(cv_curve, key=cv_curve.get)

    decoder = train_decoder(data, fit_ids, dim_report.basis, chosen_k)
    fit_err = mean_decoding_error(data, decoder, fit_ids)
    holdout_err = mean_decoding_error(data, decoder, holdout_ids)

    fit_geom = analyze_geometry(data, fit_ids)
    holdout_geom = analyze_geometry(data, np.concatenate([fit_ids, holdout_ids]))

    report = ValidationReport(chosen_k=chosen_k, cv_curve=cv_curve, fit_error=fit_err,
                               holdout_error=holdout_err, fit_geometry=fit_geom,
                               holdout_geometry=holdout_geom)
    return dim_report, report, decoder


# ---------------------------------------------------------------------------
# Bootstrap over trials
# ---------------------------------------------------------------------------

def bootstrap_holdout_error(data, decoder: Decoder, holdout_ids: np.ndarray,
                              n_boot: int = 500, seed: int = 0, ci: float = 0.90):
    rng = np.random.default_rng(seed)
    n_trials = int(data["grid"][0]["n_trials"])
    feats = stimulus_features(data)

    draws = []
    for b in range(n_boot):
        errs = []
        for sid in holdout_ids:
            counts, _ = trials_for_stimulus(data, sid)
            boot_idx = rng.integers(0, n_trials, size=n_trials)
            preds = decode(decoder, counts[boot_idx])
            true_xy = np.array(feats[sid])
            errs.extend(np.linalg.norm(preds - true_xy[None, :], axis=1).tolist())
        draws.append(np.mean(errs))
    draws = np.asarray(draws)
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    return (float(np.quantile(draws, lo_q)), float(np.median(draws)),
            float(np.quantile(draws, hi_q))), draws
