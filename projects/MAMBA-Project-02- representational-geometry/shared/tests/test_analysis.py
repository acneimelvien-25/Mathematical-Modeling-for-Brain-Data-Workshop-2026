"""Unit tests for analysis.py: dimensionality analysis, representational
geometry, cross-validated decoder selection, and the full pipeline,
checked against synthetic data built directly from KNOWN structure (not
the Gaussian-bump generator -- that is exercised by the end-to-end smoke
test instead)."""

import csv
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import analysis  # noqa: E402
from models import GaussianBumpPopulation, GaussianBumpPopulationParams  # noqa: E402


def _make_synthetic_dataset(n_neurons=20, n_stimuli_per_side=5, n_trials=8,
                              n_holdout=4, seed=0):
    """Build a dataset dict shaped like analysis.load_dataset()'s output,
    from a KNOWN Gaussian-bump population -- so downstream pipeline
    outputs (dimensionality, decoding accuracy) can be sanity-checked
    against a controllable ground truth."""
    rng = np.random.default_rng(seed)
    vals = np.linspace(-1, 1, n_stimuli_per_side)
    xx, yy = np.meshgrid(vals, vals)
    grid_xy = np.stack([xx.ravel(), yy.ravel()], axis=1)
    n_stim = grid_xy.shape[0]

    params = GaussianBumpPopulationParams(
        preferred_xy=rng.uniform(-1.2, 1.2, size=(n_neurons, 2)),
        sigma=rng.uniform(0.4, 0.9, size=n_neurons),
        gain=rng.uniform(5, 15, size=n_neurons),
        baseline=rng.uniform(1, 3, size=n_neurons),
    )
    pop = GaussianBumpPopulation(params)
    counts = pop.sample_counts(grid_xy, n_trials, integration_time_s=1.0, rng=rng)  # (n_stim, n_trials, M)

    holdout_ids = set(rng.choice(n_stim, size=n_holdout, replace=False).tolist())
    grid = [
        {"stimulus_id": i, "feature_x": float(grid_xy[i, 0]), "feature_y": float(grid_xy[i, 1]),
         "n_trials": n_trials, "integration_time_s": 1.0,
         "role": "holdout" if i in holdout_ids else "fit"}
        for i in range(n_stim)
    ]

    stimulus_id_col = np.repeat(np.arange(n_stim), n_trials)
    trial_col = np.tile(np.arange(n_trials), n_stim)
    counts_flat = counts.reshape(n_stim * n_trials, n_neurons)

    return {"spike_counts": counts_flat, "stimulus_id": stimulus_id_col,
            "trial": trial_col, "grid": grid}


def test_load_dataset_role_column_is_not_truncated(tmp_path=None):
    """Regression test for a real bug hit during development: writing
    stimulus_grid.csv's role column via a fixed-width numpy string array
    silently truncated 'holdout' to 'hol'. Build a tiny CSV the same way
    generate_data.py does and check the loader reads it back correctly."""
    with tempfile.TemporaryDirectory() as d:
        npz_path = os.path.join(d, "population_responses.npz")
        np.savez_compressed(npz_path, spike_counts=np.zeros((2, 3), dtype=np.int16),
                             stimulus_id=np.array([0, 1], dtype=np.int16),
                             trial=np.array([0, 0], dtype=np.int16))
        with open(os.path.join(d, "stimulus_grid.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["stimulus_id", "feature_x", "feature_y", "n_trials",
                        "integration_time_s", "role"])
            w.writerow([0, 0.0, 0.0, 1, 1.0, "fit"])
            w.writerow([1, 0.5, 0.5, 1, 1.0, "holdout"])
        data = analysis.load_dataset(d)
        roles = analysis.stimulus_roles(data)
        assert roles[1] == "holdout"
        assert len(analysis.holdout_stimulus_ids(data)) == 1


def test_dimensionality_report_is_internally_consistent():
    data = _make_synthetic_dataset(seed=1)
    fit_ids = analysis.fit_stimulus_ids(data)
    report = analysis.analyze_dimensionality(data, fit_ids)
    assert report.cumulative_variance[-1] == pytest.approx(1.0, abs=1e-6)
    assert report.n_components_for_90pct <= report.n_components_for_95pct
    assert np.all(np.diff(report.cumulative_variance) >= -1e-9)


def test_geometry_report_bounds():
    data = _make_synthetic_dataset(seed=2)
    fit_ids = analysis.fit_stimulus_ids(data)
    report = analysis.analyze_geometry(data, fit_ids)
    assert -1.0 <= report.pearson_r <= 1.0
    assert -1.0 <= report.spearman_r <= 1.0
    # a genuinely tuned population should show a clearly positive
    # relationship between stimulus distance and neural distance
    assert report.pearson_r > 0.3


def test_decoder_beats_naive_baseline():
    """A trained decoder should do substantially better than predicting
    the mean stimulus location for every trial."""
    data = _make_synthetic_dataset(seed=3, n_stimuli_per_side=6, n_holdout=6)
    fit_ids = analysis.fit_stimulus_ids(data)
    holdout_ids = analysis.holdout_stimulus_ids(data)

    dim_report = analysis.analyze_dimensionality(data, fit_ids)
    decoder = analysis.train_decoder(data, fit_ids, dim_report.basis, k=5)
    decoder_err = analysis.mean_decoding_error(data, decoder, holdout_ids)

    feats = analysis.stimulus_features(data)
    fit_mean_xy = np.mean([feats[sid] for sid in fit_ids], axis=0)
    naive_errs = [np.linalg.norm(np.array(feats[sid]) - fit_mean_xy) for sid in holdout_ids]
    naive_err = np.mean(naive_errs)

    assert decoder_err < naive_err


def test_cross_validate_k_returns_all_requested_k_and_finite_errors():
    data = _make_synthetic_dataset(seed=4, n_stimuli_per_side=6)
    fit_ids = analysis.fit_stimulus_ids(data)
    k_values = [1, 2, 4, 8]
    cv_curve = analysis.cross_validate_k(data, fit_ids, k_values, n_folds=3, seed=0)
    assert set(cv_curve.keys()) == set(k_values)
    for k, err in cv_curve.items():
        assert np.isfinite(err)
        assert err >= 0


def test_full_pipeline_runs_and_holdout_error_is_reasonable():
    data = _make_synthetic_dataset(seed=5, n_stimuli_per_side=6, n_trials=10, n_holdout=6)
    dim_report, report, decoder = analysis.full_pipeline(data, k_values=range(1, 15), n_folds=4)
    assert report.chosen_k in range(1, 15)
    assert np.isfinite(report.fit_error)
    assert np.isfinite(report.holdout_error)
    # feature space spans [-1,1]^2 (diagonal ~2.83); a working decoder
    # should do much better than a large fraction of that span
    assert report.holdout_error < 1.0


def test_bootstrap_ci_bounds_are_ordered():
    data = _make_synthetic_dataset(seed=6, n_stimuli_per_side=6, n_holdout=6)
    fit_ids = analysis.fit_stimulus_ids(data)
    holdout_ids = analysis.holdout_stimulus_ids(data)
    dim_report = analysis.analyze_dimensionality(data, fit_ids)
    decoder = analysis.train_decoder(data, fit_ids, dim_report.basis, k=5)
    ci, draws = analysis.bootstrap_holdout_error(data, decoder, holdout_ids, n_boot=30, seed=0)
    lo, mid, hi = ci
    assert lo <= mid <= hi
    assert len(draws) == 30
