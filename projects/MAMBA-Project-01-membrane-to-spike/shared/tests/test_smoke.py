"""End-to-end smoke test: regenerate the workshop dataset in a temporary
directory and confirm it exactly matches the committed data (determinism),
then run the full analysis pipeline against the COMMITTED data and check
that every output is finite and in a sane range. This is the test to run
first when something seems broken."""

import filecmp
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


def test_data_regeneration_is_deterministic():
    """Rerunning generate_data.py from scratch must reproduce the committed
    shared/data/ files exactly, byte for byte, for the CSV files, and
    numerically for the compressed voltage array (npz compression headers
    can differ by run, so this array is compared by value, not by byte)."""
    with tempfile.TemporaryDirectory() as tmp_data, tempfile.TemporaryDirectory() as tmp_instr:
        result = subprocess.run(
            [sys.executable, os.path.join(SRC_DIR, "generate_data.py"),
             "--outdir", tmp_data, "--instructor-outdir", tmp_instr],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr

        for fname in ["spike_times.csv", "stimulus_protocol.csv", "data_dictionary.csv"]:
            with open(os.path.join(tmp_data, fname)) as f_new, \
                 open(os.path.join(DATA_DIR, fname)) as f_committed:
                assert f_new.read() == f_committed.read(), f"{fname} differs after regeneration"

        new_npz = np.load(os.path.join(tmp_data, "voltage_traces.npz"))
        committed_npz = np.load(os.path.join(DATA_DIR, "voltage_traces.npz"))
        for key in ["t_ms", "V_obs_mV", "condition_id", "trial"]:
            np.testing.assert_array_equal(new_npz[key], committed_npz[key])

        with open(os.path.join(tmp_instr, "ground_truth.json")) as f_new, \
             open(os.path.join(INSTRUCTOR_DIR, "ground_truth.json")) as f_committed:
            assert json.load(f_new) == json.load(f_committed)


def test_end_to_end_pipeline_runs_on_committed_data_and_outputs_are_sane():
    data = analysis.load_dataset(DATA_DIR)
    SUB_IDS, SPK_IDS = [0, 1, 2], [5, 6]

    sub = analysis.fit_subthreshold(data, SUB_IDS)
    assert np.isfinite(sub.E_L) and np.isfinite(sub.R) and np.isfinite(sub.tau_m)
    assert sub.tau_m > 0
    assert -80 < sub.E_L < -40  # sane cortical/squid-axon range in mV

    spk = analysis.fit_spiking(data, SPK_IDS, sub)
    assert np.isfinite(spk.V_th) and np.isfinite(spk.V_reset) and np.isfinite(spk.t_ref)
    assert spk.V_reset < spk.V_th  # reset must be more hyperpolarized than threshold
    assert spk.t_ref > 0

    params = analysis.full_fit(data, SUB_IDS, SPK_IDS)
    report = analysis.validate(data, params)
    for cid, err in report.relative_error.items():
        assert np.isfinite(err)

    # the two spiking FIT conditions must be reproduced almost exactly
    # (fit_spiking solves for V_th, t_ref using exactly these two rates)
    assert report.relative_error[5] < 0.05
    assert report.relative_error[6] < 0.05

    # the model-limitation finding this project is built around: the
    # near-rheobase HOLDOUT conditions should show LARGE relative error
    # (the whole point of Project 01's design -- see instructor_guide.md)
    assert report.relative_error[3] > 0.5
    assert report.relative_error[4] > 0.3


def test_bootstrap_runs_without_error_on_committed_data():
    data = analysis.load_dataset(DATA_DIR)
    summary, draws = analysis.bootstrap_ci(data, [0, 1, 2], [5, 6], n_boot=25, seed=0)
    assert set(summary.keys()) == {"tau_m", "R", "E_L", "V_th", "V_reset", "t_ref"}
    for k, (lo, mid, hi) in summary.items():
        assert np.isfinite(lo) and np.isfinite(mid) and np.isfinite(hi)
        assert lo <= hi
