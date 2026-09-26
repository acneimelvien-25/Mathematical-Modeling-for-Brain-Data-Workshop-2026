"""End-to-end smoke test: regenerate the workshop dataset in a temporary
directory and confirm it exactly matches the committed data (determinism),
then run the full analysis pipeline against the COMMITTED data and check
that every output is finite and in a sane range, including the headline
findings (PCA dimensionality exceeding the true 2-D stimulus structure,
and a genuine bias-variance trade-off in the decoder's choice of k) this
project is built around."""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
INSTRUCTOR_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "instructor", "reference_outputs")

sys.path.insert(0, SRC_DIR)
import analysis  # noqa: E402


def test_data_regeneration_is_deterministic():
    """Rerunning generate_data.py from scratch must reproduce the committed
    shared/data/ files exactly: byte for byte for the CSV file, and
    numerically for the compressed spike-count array (npz compression
    headers can differ by run, so this array is compared by value, not by
    byte)."""
    with tempfile.TemporaryDirectory() as tmp_data, tempfile.TemporaryDirectory() as tmp_instr:
        result = subprocess.run(
            [sys.executable, os.path.join(SRC_DIR, "generate_data.py"),
             "--outdir", tmp_data, "--instructor-outdir", tmp_instr],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr

        for fname in ["stimulus_grid.csv", "data_dictionary.csv"]:
            with open(os.path.join(tmp_data, fname)) as f_new, \
                 open(os.path.join(DATA_DIR, fname)) as f_committed:
                assert f_new.read() == f_committed.read(), f"{fname} differs after regeneration"

        new_npz = np.load(os.path.join(tmp_data, "population_responses.npz"))
        committed_npz = np.load(os.path.join(DATA_DIR, "population_responses.npz"))
        for key in ["spike_counts", "stimulus_id", "trial"]:
            np.testing.assert_array_equal(new_npz[key], committed_npz[key])

        with open(os.path.join(tmp_instr, "ground_truth.json")) as f_new, \
             open(os.path.join(INSTRUCTOR_DIR, "ground_truth.json")) as f_committed:
            assert json.load(f_new) == json.load(f_committed)


def test_stimulus_grid_role_column_has_both_fit_and_holdout():
    """Regression test for a real bug caught during development: a
    fixed-width numpy string array silently truncated 'holdout' to
    'hol' when the CSV was written, making every stimulus look like
    'fit'. Confirm the committed data doesn't have this problem."""
    data = analysis.load_dataset(DATA_DIR)
    roles = set(analysis.stimulus_roles(data).values())
    assert roles == {"fit", "holdout"}
    assert len(analysis.holdout_stimulus_ids(data)) == 12
    assert len(analysis.fit_stimulus_ids(data)) == 52


def test_end_to_end_pipeline_runs_on_committed_data_and_outputs_are_sane():
    data = analysis.load_dataset(DATA_DIR)
    dim_report, report, decoder = analysis.full_pipeline(data, k_values=range(1, 41), n_folds=5)

    assert np.isfinite(dim_report.cumulative_variance).all()
    assert dim_report.cumulative_variance[-1] == pytest.approx(1.0, abs=1e-5)

    # headline finding #1: linear (PCA) dimensionality needed for 90%
    # variance should EXCEED the true 2-D stimulus dimensionality -- the
    # whole point of this project's design (nonlinear tuning curves)
    assert dim_report.n_components_for_90pct > 2
    assert dim_report.n_components_for_90pct < 20  # but not absurdly high either

    # headline finding #2: representational geometry should be
    # substantially (not perfectly) preserved
    assert 0.5 < report.fit_geometry.pearson_r < 0.999

    # headline finding #3: a genuine bias-variance trade-off should be
    # visible in the cross-validation curve -- error at k=1 should be
    # clearly worse than at the chosen k (underfitting), and the curve
    # should not be monotonically decreasing all the way to the largest
    # tested k (i.e. some overfitting at very high k)
    assert report.cv_curve[1] > report.cv_curve[report.chosen_k] * 1.5
    max_k = max(report.cv_curve)
    assert report.cv_curve[max_k] > report.cv_curve[report.chosen_k]

    assert np.isfinite(report.fit_error) and report.fit_error > 0
    assert np.isfinite(report.holdout_error) and report.holdout_error > 0
    # feature space is [-1,1]^2 (diagonal ~2.83); a working decoder should
    # do much better than a large fraction of that
    assert report.holdout_error < 1.0


def test_bootstrap_runs_without_error_on_committed_data():
    data = analysis.load_dataset(DATA_DIR)
    fit_ids = analysis.fit_stimulus_ids(data)
    holdout_ids = analysis.holdout_stimulus_ids(data)
    dim_report = analysis.analyze_dimensionality(data, fit_ids)
    decoder = analysis.train_decoder(data, fit_ids, dim_report.basis, k=20)
    ci, draws = analysis.bootstrap_holdout_error(data, decoder, holdout_ids, n_boot=50, seed=0)
    lo, mid, hi = ci
    assert np.isfinite(lo) and np.isfinite(mid) and np.isfinite(hi)
    assert lo <= mid <= hi
