"""End-to-end smoke test: regenerate the workshop dataset in a temporary
directory and confirm it exactly matches the committed data (determinism),
then run the full analysis pipeline against the COMMITTED data and check
that every output is finite and in a sane range, including the headline
finding (sparse-session identifiability failure vs. pooled recovery) this
project is built around."""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
INSTRUCTOR_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "instructor", "reference_outputs")

sys.path.insert(0, SRC_DIR)
import analysis  # noqa: E402

TRUE_TAU = 25.0
FIT_IDS = [0, 1, 2, 3]
HOLDOUT_IDS = [4, 5]
SPARSEST_FIT_SESSION = 3  # mean_dt=40ms, the identifiability-failure case


def test_data_regeneration_is_deterministic():
    """Rerunning generate_data.py from scratch must reproduce the committed
    shared/data/ files exactly: byte for byte for the CSV file, and
    numerically for the compressed sample arrays (npz compression headers
    can differ by run, so these are compared by value, not by byte)."""
    with tempfile.TemporaryDirectory() as tmp_data, tempfile.TemporaryDirectory() as tmp_instr:
        result = subprocess.run(
            [sys.executable, os.path.join(SRC_DIR, "generate_data.py"),
             "--outdir", tmp_data, "--instructor-outdir", tmp_instr],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr

        for fname in ["session_protocol.csv", "data_dictionary.csv"]:
            with open(os.path.join(tmp_data, fname)) as f_new, \
                 open(os.path.join(DATA_DIR, fname)) as f_committed:
                assert f_new.read() == f_committed.read(), f"{fname} differs after regeneration"

        new_npz = np.load(os.path.join(tmp_data, "voltage_sessions.npz"))
        committed_npz = np.load(os.path.join(DATA_DIR, "voltage_sessions.npz"))
        for key in ["session_id", "time_ms", "V_obs_mV"]:
            np.testing.assert_array_equal(new_npz[key], committed_npz[key])

        with open(os.path.join(tmp_instr, "ground_truth.json")) as f_new, \
             open(os.path.join(INSTRUCTOR_DIR, "ground_truth.json")) as f_committed:
            assert json.load(f_new) == json.load(f_committed)


def test_sessions_have_irregular_not_fixed_spacing():
    """Regression-style sanity check: confirm the committed data really is
    irregularly sampled (a bug that accidentally generated a fixed grid
    would silently invalidate the whole exercise)."""
    data = analysis.load_dataset(DATA_DIR)
    for sid in FIT_IDS + HOLDOUT_IDS:
        times, _ = analysis.session_data(data, sid)
        gaps = np.diff(times)
        # a fixed-grid bug would give gaps with ~zero variance
        assert gaps.std() / gaps.mean() > 0.1


def test_end_to_end_pipeline_runs_on_committed_data_and_outputs_are_sane():
    data = analysis.load_dataset(DATA_DIR)

    # per-session V_ss estimates should be within a few mV of a
    # plausible physiological range
    for sid in FIT_IDS + HOLDOUT_IDS:
        v_hat = analysis.estimate_V_ss(data, sid)
        assert -80 < v_hat < -40

    pooled = analysis.fit_pooled(data, FIT_IDS, max_lag=200.0)
    assert pooled.success
    assert np.isfinite(pooled.tau) and np.isfinite(pooled.sigma_p) and np.isfinite(pooled.sigma_obs)
    # pooled estimate should be in the right ballpark of the true tau
    assert TRUE_TAU * 0.5 < pooled.tau < TRUE_TAU * 2.0

    # headline finding: the sparsest FIT session's INDEPENDENT fit should
    # be far worse (as a tau estimate) than the POOLED fit that includes it
    per_session = analysis.fit_per_session(data, FIT_IDS, max_lag=200.0)
    sparse_tau = per_session[SPARSEST_FIT_SESSION].tau
    pooled_error = abs(pooled.tau - TRUE_TAU)
    sparse_error = abs(sparse_tau - TRUE_TAU) if np.isfinite(sparse_tau) else np.inf
    assert sparse_error > pooled_error

    report = analysis.validate(data, pooled, HOLDOUT_IDS, max_lag=400.0)
    for sid in HOLDOUT_IDS:
        assert sid in report.per_session
        assert np.isfinite(report.per_session[sid]["mean_relative_error"])
        # the pooled model should predict the holdout sessions' variograms
        # reasonably well, not perfectly (see instructor_guide.md)
        assert report.per_session[sid]["mean_relative_error"] < 1.0


def test_bootstrap_runs_without_error_on_committed_data():
    data = analysis.load_dataset(DATA_DIR)
    summary, draws = analysis.bootstrap_pooled_tau(data, FIT_IDS, n_boot=25, seed=0, max_lag=200.0)
    lo, mid, hi = summary["tau"]
    assert np.isfinite(lo) and np.isfinite(mid) and np.isfinite(hi)
    assert lo <= hi
