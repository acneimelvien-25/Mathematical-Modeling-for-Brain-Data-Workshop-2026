"""End-to-end smoke test: regenerate the workshop dataset in a temporary
directory and confirm it exactly matches the committed data (determinism),
then run the full reference analysis pipeline against the COMMITTED data
and check that every output is finite, sane, and reproduces this
project's headline findings (H1 wins model comparison; the shared gain
variance is recovered in the right ballpark; overdispersion is NOT
reliably detectable at the lowest contrast even at the longest window)."""

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

with open(os.path.join(INSTRUCTOR_DIR, "ground_truth.json")) as _f:
    GT = json.load(_f)

DT_MS = GT["dt_ms"]
FIT_CONTRASTS = GT["fit_contrasts_pct"]
HOLDOUT_CONTRASTS = GT["holdout_contrasts_pct"]
ALL_CONTRASTS = GT["contrasts_pct"]
FIT_WINDOWS = GT["fit_windows_ms"]
HOLDOUT_WINDOW = GT["holdout_window_ms"]
ALL_WINDOWS = GT["windows_ms"]
SIGMA_G_TRUE = GT["sigma_g_true"]
LOWEST_CONTRAST = min(ALL_CONTRASTS)


def test_data_regeneration_is_deterministic():
    """Rerunning generate_data.py from scratch must reproduce the
    committed shared/data/ files exactly (byte for byte for every file,
    since everything here is plain CSV/markdown/JSON -- no compressed
    binary formats in this project's data)."""
    with tempfile.TemporaryDirectory() as tmp_data, tempfile.TemporaryDirectory() as tmp_instr:
        result = subprocess.run(
            [sys.executable, os.path.join(SRC_DIR, "generate_data.py"),
             "--outdir", tmp_data, "--instructor-outdir", tmp_instr],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr

        for fname in ["spike_times.csv", "trial_protocol.csv", "data_dictionary.csv", "README.md"]:
            with open(os.path.join(tmp_data, fname)) as f_new, \
                 open(os.path.join(DATA_DIR, fname)) as f_committed:
                assert f_new.read() == f_committed.read(), f"{fname} differs after regeneration"

        with open(os.path.join(tmp_instr, "ground_truth.json")) as f_new, \
             open(os.path.join(INSTRUCTOR_DIR, "ground_truth.json")) as f_committed:
            assert json.load(f_new) == json.load(f_committed)


def _load_condition_counts():
    protocol = pd.read_csv(os.path.join(DATA_DIR, "trial_protocol.csv"))
    spikes = pd.read_csv(os.path.join(DATA_DIR, "spike_times.csv"))
    counts_by_condition = {}
    trial_ids_by_condition = {}
    for c in ALL_CONTRASTS:
        trial_ids = protocol.loc[protocol.condition_pct == c, "trial_id"].to_numpy()
        sub = spikes[spikes.condition_pct == c]
        counts = analysis.counts_from_spike_times(
            trial_ids, sub.trial_id.to_numpy(), sub.spike_time_ms.to_numpy(), ALL_WINDOWS)
        counts_by_condition[c] = counts
        trial_ids_by_condition[c] = trial_ids
    return counts_by_condition, protocol


def test_trial_counts_and_roles_match_ground_truth_protocol():
    counts_by_condition, protocol = _load_condition_counts()
    for c in ALL_CONTRASTS:
        assert counts_by_condition[c].shape == (GT["n_trials_per_condition"], len(ALL_WINDOWS))
    expected_role = {c: ("fit" if c in FIT_CONTRASTS else "holdout") for c in ALL_CONTRASTS}
    for c in ALL_CONTRASTS:
        roles = protocol.loc[protocol.condition_pct == c, "condition_role"].unique()
        assert list(roles) == [expected_role[c]]


def test_end_to_end_pipeline_recovers_sigma_g_in_right_ballpark():
    counts_by_condition, _ = _load_condition_counts()
    win_idx = {w: i for i, w in enumerate(ALL_WINDOWS)}
    fit_idx = [win_idx[w] for w in FIT_WINDOWS]

    lam_hat = {c: analysis.estimate_lambda_hz(counts_by_condition[c][:, win_idx[ALL_WINDOWS[0]]],
                                               ALL_WINDOWS[0]) for c in ALL_CONTRASTS}

    slopes = {}
    for c in FIT_CONTRASTS:
        ff = analysis.empirical_fano_factor(counts_by_condition[c][:, fit_idx])
        slope, _ = analysis.fit_ff_vs_window_slope(FIT_WINDOWS, ff)
        slopes[c] = slope

    sg2_hat = analysis.pooled_sigma_g2({c: lam_hat[c] for c in FIT_CONTRASTS}, slopes)
    sigma_g_hat = np.sqrt(max(sg2_hat, 0.0))

    assert np.isfinite(sigma_g_hat)
    # should be within a factor of ~1.5 of the true value -- a loose bound
    # appropriate for a single, realistically-sized (n=250/condition)
    # committed dataset, not a large-sample asymptotic check
    assert 0.5 * SIGMA_G_TRUE < sigma_g_hat < 1.5 * SIGMA_G_TRUE


def test_model_comparison_prefers_doubly_stochastic_model_on_committed_data():
    counts_by_condition, _ = _load_condition_counts()
    win_idx = {w: i for i, w in enumerate(ALL_WINDOWS)}
    fit_idx = [win_idx[w] for w in FIT_WINDOWS]

    lam_hat = {c: analysis.estimate_lambda_hz(counts_by_condition[c][:, win_idx[ALL_WINDOWS[0]]],
                                               ALL_WINDOWS[0]) for c in ALL_CONTRASTS}
    slopes = {}
    cells = {}
    for c in FIT_CONTRASTS:
        ff = analysis.empirical_fano_factor(counts_by_condition[c][:, fit_idx])
        slope, _ = analysis.fit_ff_vs_window_slope(FIT_WINDOWS, ff)
        slopes[c] = slope
        for j, w in zip(fit_idx, FIT_WINDOWS):
            cells[(c, w)] = counts_by_condition[c][:, j]

    sg2_hat = analysis.pooled_sigma_g2({c: lam_hat[c] for c in FIT_CONTRASTS}, slopes)
    phi_hat = analysis.fit_phi_constant_overdispersion(cells)
    result = analysis.compare_models(cells, {c: lam_hat[c] for c in FIT_CONTRASTS}, DT_MS, sg2_hat, phi_hat)

    assert result.best_model() == "H1_doubly_stochastic"
    assert result.aic["H1_doubly_stochastic"] < result.aic["H0_poisson"]
    assert result.aic["H1_doubly_stochastic"] < result.aic["H2_constant_overdispersion"]


def test_holdout_condition_predictions_are_in_reasonable_agreement():
    counts_by_condition, _ = _load_condition_counts()
    win_idx = {w: i for i, w in enumerate(ALL_WINDOWS)}
    fit_idx = [win_idx[w] for w in FIT_WINDOWS]

    lam_hat = {c: analysis.estimate_lambda_hz(counts_by_condition[c][:, win_idx[ALL_WINDOWS[0]]],
                                               ALL_WINDOWS[0]) for c in ALL_CONTRASTS}
    slopes = {}
    for c in FIT_CONTRASTS:
        ff = analysis.empirical_fano_factor(counts_by_condition[c][:, fit_idx])
        slope, _ = analysis.fit_ff_vs_window_slope(FIT_WINDOWS, ff)
        slopes[c] = slope
    sg2_hat = analysis.pooled_sigma_g2({c: lam_hat[c] for c in FIT_CONTRASTS}, slopes)

    for c in HOLDOUT_CONTRASTS:
        ff_obs = analysis.empirical_fano_factor(counts_by_condition[c])
        ff_pred = analysis.predict_holdout_ff(lam_hat[c], ALL_WINDOWS, DT_MS, sg2_hat)
        rmse = analysis.holdout_rmse(ff_obs, ff_pred)
        assert rmse < 0.35  # loose bound; see instructor materials for the exact committed values


def test_lowest_contrast_overdispersion_is_not_reliably_detectable():
    """The project's core identifiability finding: at the lowest tested
    contrast, the SAME gain-fluctuation mechanism operates, but the
    resulting overdispersion is too small, relative to counting noise and
    the available number of trials, to be reliably distinguished from a
    strict Poisson process -- even at the longest available window."""
    counts_by_condition, _ = _load_condition_counts()
    win_idx = {w: i for i, w in enumerate(ALL_WINDOWS)}
    lam_hat_low = analysis.estimate_lambda_hz(
        counts_by_condition[LOWEST_CONTRAST][:, win_idx[ALL_WINDOWS[0]]], ALL_WINDOWS[0])

    rng = np.random.default_rng(0)
    n_significant = 0
    for T in ALL_WINDOWS:
        counts = counts_by_condition[LOWEST_CONTRAST][:, win_idx[T]]
        _, pval = analysis.poisson_null_detectability_pvalue(counts, lam_hat_low, T, n_boot=2000, rng=rng)
        if pval < 0.01:
            n_significant += 1
    # allow at most one false positive across the 6 tested windows at
    # this stringent threshold (multiple-comparisons realism)
    assert n_significant <= 1


def test_bootstrap_sigma_g_ci_runs_without_error_on_committed_data():
    counts_by_condition, _ = _load_condition_counts()
    win_idx = {w: i for i, w in enumerate(ALL_WINDOWS)}
    fit_idx = [win_idx[w] for w in FIT_WINDOWS]
    lam_hat = {c: analysis.estimate_lambda_hz(counts_by_condition[c][:, win_idx[ALL_WINDOWS[0]]],
                                               ALL_WINDOWS[0]) for c in FIT_CONTRASTS}
    rng = np.random.default_rng(1)
    boot = analysis.bootstrap_sigma_g_ci(
        counts_by_condition, FIT_CONTRASTS, fit_idx, FIT_WINDOWS, lam_hat, n_boot=50, rng=rng)
    assert len(boot) == 50
    assert np.all(np.isfinite(boot))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    assert lo <= hi
