"""Unit tests for models.py: orientation/contrast schedules, design
matrix construction, and the drift function/basis."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models import (  # noqa: E402
    TrueEffects, DriftParams, ORIENTATIONS_DEG, CONTRASTS_PCT,
    drift_function, drift_basis, generate_orientation_sequence,
    generate_contrast_schedule, orientation_dummies, log_contrast_regressor,
    simulate_session, build_design_matrix, MAIN_EFFECT_NAMES,
)


def test_orientation_sequence_is_block_balanced():
    rng = np.random.default_rng(0)
    seq = generate_orientation_sequence(400, rng)
    # every consecutive block of 4 contains each orientation exactly once
    for start in range(0, 400, 4):
        block = seq[start:start + 4]
        assert set(block.tolist()) == set(ORIENTATIONS_DEG)


def test_orientation_sequence_uncorrelated_with_trial_number():
    rng = np.random.default_rng(1)
    n = 2000
    seq = generate_orientation_sequence(n, rng)
    t_frac = np.arange(n) / n
    for ori in ORIENTATIONS_DEG:
        dummy = (seq == ori).astype(float)
        corr = np.corrcoef(dummy, t_frac)[0, 1]
        assert abs(corr) < 0.02


def test_contrast_schedule_is_correlated_with_trial_number():
    """This IS the intended confound -- contrast should show a real,
    positive, moderate correlation with trial index, unlike orientation."""
    rng = np.random.default_rng(2)
    n = 2000
    seq = generate_contrast_schedule(n, rng)
    t_frac = np.arange(n) / n
    corr = np.corrcoef(np.log2(seq / 12.5), t_frac)[0, 1]
    assert 0.15 < corr < 0.6


def test_contrast_schedule_uses_only_valid_levels():
    rng = np.random.default_rng(3)
    seq = generate_contrast_schedule(300, rng)
    assert set(np.unique(seq).tolist()).issubset(set(CONTRASTS_PCT))


def test_orientation_dummies_are_mutually_exclusive_and_baseline_is_zero():
    ori = np.array([0.0, 45.0, 90.0, 135.0, 0.0])
    D45, D90, D135 = orientation_dummies(ori)
    stacked = np.vstack([D45, D90, D135])
    assert np.all(stacked.sum(axis=0) <= 1)
    assert D45[0] == 0 and D90[0] == 0 and D135[0] == 0  # baseline (0 deg) trial


def test_log_contrast_regressor_zero_at_baseline_and_correct_steps():
    logC = log_contrast_regressor(np.array([12.5, 25.0, 50.0, 100.0]))
    np.testing.assert_allclose(logC, [0.0, 1.0, 2.0, 3.0])


def test_drift_function_is_deterministic_and_matches_hand_formula():
    p = DriftParams(d_lin=-3.0, d_quad=2.0, d_sin=1.5, d_cos=-1.0)
    t = np.array([0.0, 0.25, 0.5, 0.75])
    expected = (p.d_lin * t + p.d_quad * t ** 2
                + p.d_sin * np.sin(2 * np.pi * t) + p.d_cos * np.cos(2 * np.pi * t))
    np.testing.assert_allclose(drift_function(t, p), expected)


def test_drift_basis_exactly_spans_the_true_drift_function():
    """By construction, drift_function is a LINEAR combination of the
    columns of drift_basis -- confirm this holds exactly (up to floating
    point), which is what makes the reference Model B able to recover
    the drift exactly (given enough data and no noise)."""
    p = DriftParams(d_lin=-3.0, d_quad=2.0, d_sin=1.5, d_cos=-1.0)
    t = np.linspace(0, 1, 50, endpoint=False)
    basis = drift_basis(t)
    coeffs = np.array([p.d_lin, p.d_quad, p.d_sin, p.d_cos])
    reconstructed = basis @ coeffs
    np.testing.assert_allclose(reconstructed, drift_function(t, p), atol=1e-10)


def test_simulate_session_shapes_and_keys():
    rng = np.random.default_rng(4)
    effects = TrueEffects(beta0=5, beta_45=1, beta_90=2, beta_135=0.5, beta_contrast=1.5, sigma_noise=1.0)
    drift = DriftParams(d_lin=0, d_quad=0, d_sin=0, d_cos=0)
    session = simulate_session(200, effects, drift, rng)
    for key in ["trial_index", "orientation_deg", "contrast_pct", "t_frac", "response",
                "mean_response", "drift_true"]:
        assert key in session
        assert len(session[key]) == 200


def test_simulate_session_zero_noise_zero_drift_matches_deterministic_formula():
    """With sigma_noise = 0 and a null drift, response should EXACTLY
    equal the deterministic linear combination of true effects (a strong
    sanity check that no term is dropped or double-counted)."""
    rng = np.random.default_rng(5)
    effects = TrueEffects(beta0=5, beta_45=1.2, beta_90=3.5, beta_135=0.5, beta_contrast=2.0, sigma_noise=0.0)
    drift = DriftParams(d_lin=0, d_quad=0, d_sin=0, d_cos=0)
    session = simulate_session(400, effects, drift, rng)
    D45, D90, D135 = orientation_dummies(session["orientation_deg"])
    logC = log_contrast_regressor(session["contrast_pct"])
    expected = effects.beta0 + effects.beta_45 * D45 + effects.beta_90 * D90 + effects.beta_135 * D135 + effects.beta_contrast * logC
    np.testing.assert_allclose(session["response"], expected, atol=1e-10)


def test_build_design_matrix_shape_and_column_order():
    ori = np.array([0.0, 45.0, 90.0, 135.0])
    contrast = np.array([12.5, 25.0, 50.0, 100.0])
    X = build_design_matrix(ori, contrast)
    assert X.shape == (4, len(MAIN_EFFECT_NAMES))
    np.testing.assert_allclose(X[:, 0], 1.0)  # intercept column


def test_build_design_matrix_with_drift_basis_shape():
    ori = np.array([0.0, 45.0, 90.0, 135.0])
    contrast = np.array([12.5, 25.0, 50.0, 100.0])
    t_frac = np.array([0.0, 0.25, 0.5, 0.75])
    X = build_design_matrix(ori, contrast, include_drift_basis=True, t_frac=t_frac)
    assert X.shape == (4, len(MAIN_EFFECT_NAMES) + 4)


def test_build_design_matrix_requires_t_frac_for_drift():
    ori = np.array([0.0, 45.0])
    contrast = np.array([12.5, 25.0])
    with pytest.raises(ValueError):
        build_design_matrix(ori, contrast, include_drift_basis=True)
