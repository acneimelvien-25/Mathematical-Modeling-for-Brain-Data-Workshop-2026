"""End-to-end smoke test: regenerate the workshop dataset in a temporary
directory and confirm it exactly matches the committed data (determinism),
then run the full reference analysis pipeline against the COMMITTED data
and check that every output is finite, sane, and reproduces this
project's headline findings (the pooled gain estimate is close to the
true value; the predicted critical coupling and onset frequency closely
match direct simulation on holdout data; the holdout traveling-wave phase
pattern matches the predicted mode-2, 144-degree lag)."""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
INSTRUCTOR_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "instructor", "reference_outputs")

sys.path.insert(0, SRC_DIR)
import analysis  # noqa: E402
from models import RingParams, simulate_directed_ring  # noqa: E402

with open(os.path.join(INSTRUCTOR_DIR, "ground_truth.json")) as _f:
    GT = json.load(_f)

N_NODES = GT["n_nodes"]
TAU = GT["tau"]
FIT_W = GT["fit_w"]
HOLDOUT_W = GT["holdout_w"]
MODE_K = GT["most_unstable_mode_k"]
CRITICAL_W_TRUE = GT["critical_w_true"]
ONSET_FREQ_TRUE = GT["onset_angular_frequency_true"]
PHASE_LAG_TRUE = GT["phase_lag_deg_true"]


def test_data_regeneration_is_deterministic():
    with tempfile.TemporaryDirectory() as tmp_data, tempfile.TemporaryDirectory() as tmp_instr:
        result = subprocess.run(
            [sys.executable, os.path.join(SRC_DIR, "generate_data.py"),
             "--outdir", tmp_data, "--instructor-outdir", tmp_instr],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        for fname in ["fit_timeseries.csv", "holdout_timeseries.csv", "data_dictionary.csv", "README.md"]:
            with open(os.path.join(tmp_data, fname)) as f_new, \
                 open(os.path.join(DATA_DIR, fname)) as f_committed:
                assert f_new.read() == f_committed.read(), f"{fname} differs after regeneration"
        with open(os.path.join(tmp_instr, "ground_truth.json")) as f_new, \
             open(os.path.join(INSTRUCTOR_DIR, "ground_truth.json")) as f_committed:
            assert json.load(f_new) == json.load(f_committed)


def _load_fit_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "fit_timeseries.csv"))
    node_cols = [c for c in df.columns if c.startswith("x")]
    by_w = {}
    for w in FIT_W:
        sub = df[df.w == w].sort_values("time")
        by_w[w] = sub[node_cols].to_numpy()
    return by_w


def _load_holdout_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "holdout_timeseries.csv"))
    node_cols = [c for c in df.columns if c.startswith("x")]
    by_w = {}
    for w in HOLDOUT_W:
        sub = df[df.w == w].sort_values("time")
        by_w[w] = sub[node_cols].to_numpy()
    return by_w


def test_gain_estimate_close_to_true_value_on_committed_data():
    fit_data = _load_fit_data()
    dt_eff = 0.05
    coupling_coefs = []
    for w in FIT_W:
        fit = analysis.fit_local_linear_model(fit_data[w], dt_eff)
        coupling_coefs.append(fit.coupling_coef)
    g_hat = analysis.pooled_gain_estimate(FIT_W, coupling_coefs)
    assert g_hat == pytest.approx(GT["g_true"], abs=0.1)


def test_predicted_critical_coupling_close_to_true_value():
    fit_data = _load_fit_data()
    dt_eff = 0.05
    coupling_coefs = [analysis.fit_local_linear_model(fit_data[w], dt_eff).coupling_coef for w in FIT_W]
    g_hat = analysis.pooled_gain_estimate(FIT_W, coupling_coefs)
    summary = analysis.predicted_bifurcation_summary(g_hat, N_NODES, TAU, MODE_K)
    assert summary["critical_w"] == pytest.approx(CRITICAL_W_TRUE, rel=0.1)
    # onset frequency should match closely regardless of g_hat (it's g-independent)
    assert summary["onset_angular_frequency"] == pytest.approx(ONSET_FREQ_TRUE, abs=1e-6)
    assert summary["phase_lag_deg"] == pytest.approx(PHASE_LAG_TRUE, abs=1e-6)


def test_holdout_data_shows_growing_oscillation_amplitude_with_w():
    holdout_data = _load_holdout_data()
    dt_eff = 0.05
    amplitudes = []
    for w in HOLDOUT_W:
        x = holdout_data[w]
        tail = x[int(len(x) * 0.7):]
        amplitudes.append(analysis.oscillation_amplitude(tail[:, 0]))
    # all holdout w are past threshold -> amplitude should be clearly nonzero and
    # (not strictly monotonic near onset, but) larger for the most negative w
    assert all(a > 0.3 for a in amplitudes)
    assert amplitudes[-1] > amplitudes[0]  # w=-2.5 has bigger amplitude than w=-1.3


def test_holdout_oscillation_frequency_near_onset_matches_prediction():
    holdout_data = _load_holdout_data()
    dt_eff = 0.05
    w_near_onset = min(HOLDOUT_W, key=lambda w: abs(w - CRITICAL_W_TRUE))
    x = holdout_data[w_near_onset]
    tail = x[int(len(x) * 0.5):]
    freq = analysis.dominant_angular_frequency(tail[:, 0], dt_eff)
    assert freq == pytest.approx(ONSET_FREQ_TRUE, rel=0.15)


def test_holdout_phase_lag_matches_predicted_traveling_wave_pattern():
    holdout_data = _load_holdout_data()
    w = HOLDOUT_W[0]  # closest to threshold, cleanest sinusoidal pattern
    x = holdout_data[w]
    tail = x[int(len(x) * 0.5):]
    lags = analysis.mean_phase_lag(tail)
    predicted_rad = np.deg2rad(PHASE_LAG_TRUE)
    for lag in lags:
        agreement = min(abs(lag - predicted_rad), abs(lag + predicted_rad),
                         abs(abs(lag) - predicted_rad))
        assert agreement < 0.2  # radians, ~11 degrees tolerance


def test_symmetric_ring_gives_no_sustained_oscillation_past_threshold():
    from models import simulate_symmetric_ring
    params = RingParams(n_nodes=N_NODES, tau=TAU)
    rng = np.random.default_rng(0)
    x = simulate_symmetric_ring(w=-2.0, params=params, T=200.0, dt=0.01, sigma=0.0,
                                 x0=0.1 * rng.standard_normal(N_NODES))
    tail = x[int(len(x) * 0.8):]
    assert np.all(tail.std(axis=0) < 1e-2)  # converged to a static pattern
