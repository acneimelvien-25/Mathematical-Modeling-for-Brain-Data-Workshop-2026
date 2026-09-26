"""End-to-end smoke test: regenerate the workshop dataset in a temporary
directory and confirm it exactly matches the committed data (determinism),
then run the full reference analysis pipeline against the COMMITTED data
and check that every output is finite, sane, and reproduces this
project's headline findings (the naive model's contrast coefficient is
biased substantially more than its orientation coefficients; the naive
model's residuals are strongly autocorrelated while the full model's are
not; the full model predicts holdout data noticeably better)."""

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
from models import build_design_matrix, drift_basis  # noqa: E402

with open(os.path.join(INSTRUCTOR_DIR, "ground_truth.json")) as _f:
    GT = json.load(_f)

TRUE_EFFECTS = GT["true_effects"]
TRUE_DRIFT = GT["true_drift_params"]
N_TRIALS = GT["n_trials"]
N_HOLDOUT = GT["n_holdout"]


def test_data_regeneration_is_deterministic():
    with tempfile.TemporaryDirectory() as tmp_data, tempfile.TemporaryDirectory() as tmp_instr:
        result = subprocess.run(
            [sys.executable, os.path.join(SRC_DIR, "generate_data.py"),
             "--outdir", tmp_data, "--instructor-outdir", tmp_instr],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        for fname in ["trials.csv", "data_dictionary.csv", "README.md"]:
            with open(os.path.join(tmp_data, fname)) as f_new, \
                 open(os.path.join(DATA_DIR, fname)) as f_committed:
                assert f_new.read() == f_committed.read(), f"{fname} differs after regeneration"
        with open(os.path.join(tmp_instr, "ground_truth.json")) as f_new, \
             open(os.path.join(INSTRUCTOR_DIR, "ground_truth.json")) as f_committed:
            assert json.load(f_new) == json.load(f_committed)


def _load_trials():
    return pd.read_csv(os.path.join(DATA_DIR, "trials.csv"))


def test_trial_roles_match_ground_truth_split():
    df = _load_trials()
    assert len(df) == N_TRIALS
    assert (df.trial_role == "holdout").sum() == N_HOLDOUT
    assert (df.trial_role == "fit").sum() == N_TRIALS - N_HOLDOUT
    # holdout must be the LAST trials in session-time order
    holdout_idx = df.loc[df.trial_role == "holdout", "trial_index"].to_numpy()
    fit_idx = df.loc[df.trial_role == "fit", "trial_index"].to_numpy()
    assert holdout_idx.min() > fit_idx.max()


def _fit_both_models(df_fit):
    X_naive = build_design_matrix(df_fit.orientation_deg.to_numpy(), df_fit.contrast_pct.to_numpy())
    t_frac = df_fit.trial_index.to_numpy() / N_TRIALS
    X_full = build_design_matrix(df_fit.orientation_deg.to_numpy(), df_fit.contrast_pct.to_numpy(),
                                  include_drift_basis=True, t_frac=t_frac)
    fit_naive = analysis.fit_ols(X_naive, df_fit.response.to_numpy())
    fit_full = analysis.fit_ols(X_full, df_fit.response.to_numpy())
    return X_naive, X_full, fit_naive, fit_full, t_frac


def test_naive_model_contrast_coefficient_is_biased_more_than_orientation():
    df = _load_trials()
    df_fit = df[df.trial_role == "fit"]
    X_naive, X_full, fit_naive, fit_full, t_frac = _fit_both_models(df_fit)

    true_beta = np.array([TRUE_EFFECTS["beta0"], TRUE_EFFECTS["beta_45"], TRUE_EFFECTS["beta_90"],
                           TRUE_EFFECTS["beta_135"], TRUE_EFFECTS["beta_contrast"]])
    beta2_true = np.array([TRUE_DRIFT["d_lin"], TRUE_DRIFT["d_quad"], TRUE_DRIFT["d_sin"], TRUE_DRIFT["d_cos"]])
    X_drift = drift_basis(t_frac)
    predicted_bias = analysis.omitted_variable_bias(X_naive, X_drift, beta2_true)

    # relative bias (as a fraction of the true effect size) should be
    # clearly largest for the contrast regressor (index 4)
    rel_bias = np.abs(predicted_bias) / np.abs(true_beta)
    assert rel_bias[4] == rel_bias.max()
    assert rel_bias[4] > 3 * np.median(rel_bias[1:4])  # orientation dummies, excluding intercept


def test_naive_model_residuals_are_autocorrelated_full_model_is_not():
    df = _load_trials()
    df_fit = df[df.trial_role == "fit"]
    X_naive, X_full, fit_naive, fit_full, t_frac = _fit_both_models(df_fit)
    ac_naive = analysis.residual_autocorrelation(fit_naive.residuals, lags=[1, 2, 5, 10])
    ac_full = analysis.residual_autocorrelation(fit_full.residuals, lags=[1, 2, 5, 10])
    assert ac_naive[1] > 0.2
    assert abs(ac_full[1]) < 0.1


def test_nested_f_test_strongly_favors_full_model():
    df = _load_trials()
    df_fit = df[df.trial_role == "fit"]
    X_naive, X_full, fit_naive, fit_full, t_frac = _fit_both_models(df_fit)
    result = analysis.nested_f_test(fit_naive, fit_full)
    assert result.pvalue < 1e-10


def test_full_model_predicts_holdout_better_than_naive():
    df = _load_trials()
    df_fit = df[df.trial_role == "fit"]
    df_holdout = df[df.trial_role == "holdout"]
    X_naive, X_full, fit_naive, fit_full, t_frac = _fit_both_models(df_fit)

    X_naive_ho = build_design_matrix(df_holdout.orientation_deg.to_numpy(), df_holdout.contrast_pct.to_numpy())
    t_frac_ho = df_holdout.trial_index.to_numpy() / N_TRIALS
    X_full_ho = build_design_matrix(df_holdout.orientation_deg.to_numpy(), df_holdout.contrast_pct.to_numpy(),
                                     include_drift_basis=True, t_frac=t_frac_ho)

    pred_naive = analysis.predict(X_naive_ho, fit_naive.beta)
    pred_full = analysis.predict(X_full_ho, fit_full.beta)
    m_naive = analysis.holdout_metrics(df_holdout.response.to_numpy(), pred_naive)
    m_full = analysis.holdout_metrics(df_holdout.response.to_numpy(), pred_full)

    assert m_full["rmse"] < m_naive["rmse"]
    assert m_full["r2"] > m_naive["r2"]


def test_contrast_evaluation_on_committed_data_runs_and_is_finite():
    df = _load_trials()
    df_fit = df[df.trial_role == "fit"]
    X_naive, X_full, fit_naive, fit_full, t_frac = _fit_both_models(df_fit)
    c_90_vs_0 = np.array([0, 0, 1, 0, 0.0, 0, 0, 0, 0])
    result = analysis.evaluate_contrast(c_90_vs_0, fit_full)
    assert np.isfinite(result.estimate)
    assert np.isfinite(result.pvalue)
    assert result.pvalue < 0.01  # true effect (3.5) is large relative to noise


def test_bootstrap_ci_runs_on_committed_data():
    df = _load_trials()
    df_fit = df[df.trial_role == "fit"]
    X_naive, X_full, fit_naive, fit_full, t_frac = _fit_both_models(df_fit)
    c_90_vs_0 = np.array([0, 0, 1, 0, 0.0, 0, 0, 0, 0])
    boot = analysis.residual_bootstrap_ci(X_full, fit_full, c_90_vs_0, n_boot=200,
                                           rng=np.random.default_rng(0))
    assert len(boot) == 200
    assert np.all(np.isfinite(boot))
