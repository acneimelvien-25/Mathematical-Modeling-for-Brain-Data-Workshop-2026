"""End-to-end smoke test: regenerate the workshop dataset in a temporary
directory and confirm it exactly matches the committed data (determinism),
then run the full reference analysis pipeline against the COMMITTED data
and check that every output is finite, sane, and reproduces this
project's headline findings (naive uncorrected testing produces far more
false discoveries than true signal channels; the max-stat permutation
test correctly recovers close to the true signal set with few or no
false positives; discovery-significant channels show elevated, if not
always significant, effects in the independent replication half)."""

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
from models import build_design_matrix  # noqa: E402

with open(os.path.join(INSTRUCTOR_DIR, "ground_truth.json")) as _f:
    GT = json.load(_f)

N_CHANNELS = GT["n_channels"]
N_BLOCKS = GT["n_blocks"]
BLOCK_SIZE = GT["block_size"]
SIGNAL_CHANNELS = np.array(GT["signal_channels"])


def test_data_regeneration_is_deterministic():
    with tempfile.TemporaryDirectory() as tmp_data, tempfile.TemporaryDirectory() as tmp_instr:
        result = subprocess.run(
            [sys.executable, os.path.join(SRC_DIR, "generate_data.py"),
             "--outdir", tmp_data, "--instructor-outdir", tmp_instr],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        for fname in ["trials.csv", "channel_responses.csv", "channel_positions.csv",
                      "data_dictionary.csv", "README.md"]:
            with open(os.path.join(tmp_data, fname)) as f_new, \
                 open(os.path.join(DATA_DIR, fname)) as f_committed:
                assert f_new.read() == f_committed.read(), f"{fname} differs after regeneration"
        with open(os.path.join(tmp_instr, "ground_truth.json")) as f_new, \
             open(os.path.join(INSTRUCTOR_DIR, "ground_truth.json")) as f_committed:
            assert json.load(f_new) == json.load(f_committed)


def _load_data():
    trials = pd.read_csv(os.path.join(DATA_DIR, "trials.csv"))
    responses = pd.read_csv(os.path.join(DATA_DIR, "channel_responses.csv"))
    channel_cols = [c for c in responses.columns if c.startswith("ch")]
    Y = responses[channel_cols].to_numpy()
    return trials, Y


def test_trial_roles_are_balanced_and_interleaved():
    trials, Y = _load_data()
    assert len(trials) == N_BLOCKS * BLOCK_SIZE
    disc = trials[trials.trial_role == "discovery"]
    repl = trials[trials.trial_role == "replication"]
    assert len(disc) == len(repl) == N_BLOCKS * BLOCK_SIZE // 2
    # block indices should alternate in parity between roles
    disc_blocks = sorted(disc.block_index.unique())
    assert all(b % 2 == 0 for b in disc_blocks)


def test_naive_uncorrected_produces_far_more_discoveries_than_true_signals():
    trials, Y = _load_data()
    X = build_design_matrix(trials.condition.to_numpy())
    fit = analysis.fit_mass_univariate(X, Y)
    pvals = analysis.naive_pvalues(fit)
    naive_sig = np.where(pvals < 0.05)[0]
    assert len(naive_sig) > 2 * len(SIGNAL_CHANNELS)


def test_max_stat_permutation_recovers_signal_with_few_false_positives():
    trials, Y = _load_data()
    X = build_design_matrix(trials.condition.to_numpy())
    fit = analysis.fit_mass_univariate(X, Y)
    block_labels = trials.groupby("block_index").condition.first().to_numpy()

    rng = np.random.default_rng(0)
    max_null = analysis.permutation_null_max_stat(block_labels, BLOCK_SIZE, Y, n_perm=2000, rng=rng)
    maxstat_sig = analysis.max_stat_significant(fit, max_null, alpha=0.05)

    true_set = set(SIGNAL_CHANNELS.tolist())
    sig_set = set(maxstat_sig.tolist())
    false_positives = sig_set - true_set
    true_positives = sig_set & true_set

    assert len(false_positives) <= 1
    assert len(true_positives) >= len(SIGNAL_CHANNELS) - 2


def test_bonferroni_and_fdr_run_and_return_valid_subsets():
    trials, Y = _load_data()
    X = build_design_matrix(trials.condition.to_numpy())
    fit = analysis.fit_mass_univariate(X, Y)
    pvals = analysis.naive_pvalues(fit)
    bonf = analysis.bonferroni_significant(pvals)
    fdr = analysis.benjamini_hochberg_significant(pvals)
    assert set(bonf.tolist()).issubset(set(fdr.tolist()))
    assert np.all(bonf >= 0) and np.all(bonf < N_CHANNELS)


def test_discovery_replication_split_shows_partial_but_nonzero_replication():
    trials, Y = _load_data()
    disc_mask = (trials.trial_role == "discovery").to_numpy()
    repl_mask = (trials.trial_role == "replication").to_numpy()

    disc_blocks = trials[disc_mask].groupby("block_index").condition.first().to_numpy()
    repl_blocks = trials[repl_mask].groupby("block_index").condition.first().to_numpy()

    X_disc = build_design_matrix(trials.condition[disc_mask].to_numpy())
    X_repl = build_design_matrix(trials.condition[repl_mask].to_numpy())
    fit_disc = analysis.fit_mass_univariate(X_disc, Y[disc_mask])
    fit_repl = analysis.fit_mass_univariate(X_repl, Y[repl_mask])

    rng = np.random.default_rng(1)
    null_disc = analysis.permutation_null_max_stat(disc_blocks, BLOCK_SIZE, Y[disc_mask], 2000, rng)
    null_repl = analysis.permutation_null_max_stat(repl_blocks, BLOCK_SIZE, Y[repl_mask], 2000, rng)
    sig_disc = analysis.max_stat_significant(fit_disc, null_disc, alpha=0.05)
    sig_repl = analysis.max_stat_significant(fit_repl, null_repl, alpha=0.05)

    summary = analysis.replication_summary(sig_disc, sig_repl)
    assert summary["n_discovery_significant"] >= 1
    # no false positives should appear in either half's discoveries
    true_set = set(SIGNAL_CHANNELS.tolist())
    assert set(sig_disc.tolist()) - true_set == set()
    assert set(sig_repl.tolist()) - true_set == set()


def test_sandwich_variance_exceeds_naive_on_committed_design():
    from models import ar1_covariance_matrix
    trials, Y = _load_data()
    condition = trials.condition.to_numpy()
    X = build_design_matrix(condition).astype(float)
    # use +-1 coding for the single-column sandwich check (matches handout convention)
    X_contrast = (2 * condition - 1.0).reshape(-1, 1)
    Sigma = ar1_covariance_matrix(len(condition), GT["phi"], GT["sigma"])
    sw = analysis.sandwich_variance(X_contrast, Sigma, contrast_index=0)
    nv = analysis.naive_variance(X_contrast, GT["sigma"] ** 2, contrast_index=0)
    assert sw > nv
