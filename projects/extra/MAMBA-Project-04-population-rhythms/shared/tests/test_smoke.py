"""End-to-end smoke test: regenerate the workshop dataset in a temporary
directory and confirm it exactly matches the committed data (determinism),
then run the full analysis pipeline against the COMMITTED data and check
that every output is finite and in a sane range, including the headline
finding (a genuine, quantifiable bifurcation-location and stability
prediction) this project is built around."""

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

SUBTHRESHOLD_FIT_IDS = [0, 1, 2, 3]  # I_I = 5.0, 4.5, 4.0, 3.5 (stable spiral regime)
HOLDOUT_IDS = [4, 5]                 # I_I = 3.0, 2.0 (oscillatory / limit-cycle regime)


def test_data_regeneration_is_deterministic():
    """Rerunning generate_data.py from scratch must reproduce the committed
    shared/data/ files exactly: byte for byte for the CSV/JSON files, and
    numerically for the compressed activity array (npz compression headers
    can differ by run, so this array is compared by value, not by byte)."""
    with tempfile.TemporaryDirectory() as tmp_data, tempfile.TemporaryDirectory() as tmp_instr:
        result = subprocess.run(
            [sys.executable, os.path.join(SRC_DIR, "generate_data.py"),
             "--outdir", tmp_data, "--instructor-outdir", tmp_instr],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr

        for fname in ["stimulus_protocol.csv", "data_dictionary.csv", "noise_calibration.json"]:
            with open(os.path.join(tmp_data, fname)) as f_new, \
                 open(os.path.join(DATA_DIR, fname)) as f_committed:
                assert f_new.read() == f_committed.read(), f"{fname} differs after regeneration"

        new_npz = np.load(os.path.join(tmp_data, "population_traces.npz"))
        committed_npz = np.load(os.path.join(DATA_DIR, "population_traces.npz"))
        for key in ["t_ms", "E_obs", "I_obs", "condition_id", "trial"]:
            np.testing.assert_array_equal(new_npz[key], committed_npz[key])

        with open(os.path.join(tmp_instr, "ground_truth.json")) as f_new, \
             open(os.path.join(INSTRUCTOR_DIR, "ground_truth.json")) as f_committed:
            assert json.load(f_new) == json.load(f_committed)


def test_end_to_end_pipeline_runs_on_committed_data_and_outputs_are_sane():
    data = analysis.load_dataset(DATA_DIR)

    fit_results = {cid: analysis.fit_condition(data, cid, data["sigma_obs"])
                   for cid in SUBTHRESHOLD_FIT_IDS}
    for cid, fr in fit_results.items():
        assert np.all(np.isfinite(fr.eig))
        # every fit condition is in the (genuinely) stable regime -- the
        # estimate should agree with that, at least in sign, for a sound
        # pipeline (this is the internal-consistency check a group should
        # also perform -- see instructor_guide.md)
        assert analysis.dominant_eigenvalue(fr.eig).real < 0.05

    bp = analysis.predict_bifurcation(fit_results)
    assert np.isfinite(bp.I_I_crit)
    # the true bifurcation sits at approximately I_I=3.10 (see
    # instructor_guide.md for the exact derivation); the reference pipeline
    # should recover this to within a wide but meaningful tolerance
    assert 2.5 < bp.I_I_crit < 3.7

    report = analysis.validate(data, bp)
    for cid, d in report.per_condition.items():
        assert np.isfinite(d["predicted_freq_hz"])
        assert np.isfinite(d["observed_freq_hz"])

    # headline finding: every FIT condition should be predicted stable,
    # and every HOLDOUT condition should be predicted oscillatory (i.e.
    # the extrapolated crossing correctly separates the two groups the
    # dataset was deliberately designed around)
    for cid in SUBTHRESHOLD_FIT_IDS:
        assert report.per_condition[cid]["predicted_stability"] == "stable"
    for cid in HOLDOUT_IDS:
        assert report.per_condition[cid]["predicted_stability"] == "oscillatory"

    # model-free amplitude indicator (std_E) must increase monotonically
    # as I_I decreases across ALL conditions -- the underlying phenomenon
    # the whole exercise is built around, independent of any fitting
    stds = [report.per_condition[cid]["std_E"]
            for cid in sorted(report.per_condition, key=lambda c: -report.per_condition[c]["I_I"])]
    assert all(a <= b + 1e-9 for a, b in zip(stds, stds[1:]))


def test_bootstrap_runs_without_error_on_committed_data():
    data = analysis.load_dataset(DATA_DIR)
    summary, draws = analysis.bootstrap_ci(data, SUBTHRESHOLD_FIT_IDS, n_boot=25, seed=0)
    assert set(summary.keys()) == {"I_I_crit", "slope_re"}
    for k, (lo, mid, hi) in summary.items():
        assert np.isfinite(lo) and np.isfinite(mid) and np.isfinite(hi)
        assert lo <= hi
