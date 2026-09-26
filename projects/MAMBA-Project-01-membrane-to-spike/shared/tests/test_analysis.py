"""Unit tests for analysis.py: fitting recovers known parameters from
synthetic noiseless/low-noise data generated directly with LIFParams (NOT
the HH data -- that is exercised by the end-to-end smoke test instead)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models import LIFParams, LeakyIntegrateFire  # noqa: E402
import analysis  # noqa: E402


def _make_synthetic_lif_dataset(true_params: LIFParams, currents, roles, n_trials=10,
                                  t_end=200.0, dt_rec=0.05, sigma_obs=0.3, seed=0):
    """Build a dataset dict in the same shape as analysis.load_dataset(),
    but generated directly from a KNOWN LeakyIntegrateFire model (rather
    than the HH reference), so the fitting pipeline's recovered parameters
    can be checked against an exact ground truth."""
    rng = np.random.default_rng(seed)
    lif = LeakyIntegrateFire(true_params)
    t_axis = np.arange(0, t_end + dt_rec, dt_rec)
    dt_sim = 0.01

    all_cond, all_trial, all_V, spike_rows, protocol_rows = [], [], [], [], []
    for cid, (I_dc, role) in enumerate(zip(currents, roles)):
        for trial in range(n_trials):
            t_sim, V_sim, spikes = lif.simulate(lambda tt: I_dc, t_end, dt_sim)
            V_rec = np.interp(t_axis, t_sim, V_sim) + rng.normal(0, sigma_obs, size=t_axis.shape)
            all_cond.append(cid)
            all_trial.append(trial)
            all_V.append(V_rec.astype(np.float32))
            for s in spikes:
                spike_rows.append((cid, trial, float(s)))
        protocol_rows.append((cid, I_dc, n_trials, t_end, dt_rec, role))

    spikes_arr = np.array(spike_rows, dtype=[("condition_id", "i8"), ("trial", "i8"),
                                              ("spike_time_ms", "f8")])
    protocol_arr = np.array(protocol_rows, dtype=[
        ("condition_id", "i8"), ("I_dc_uA_per_cm2", "f8"), ("n_trials", "i8"),
        ("duration_ms", "f8"), ("dt_recorded_ms", "f8"), ("role", "U10")])

    return {
        "t_ms": t_axis,
        "V_obs_mV": np.stack(all_V, axis=0),
        "condition_id": np.asarray(all_cond),
        "trial": np.asarray(all_trial),
        "spikes": spikes_arr,
        "protocol": protocol_arr,
    }


TRUE_PARAMS = LIFParams(tau_m=10.0, R=5.0, E_L=-65.0, V_th=-50.0, V_reset=-65.0, t_ref=2.0)
# rheobase = (V_th-E_L)/R = 3.0
CURRENTS = [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0]
ROLES = ["fit", "fit", "fit", "holdout", "holdout", "fit", "fit"]
SUB_IDS = [0, 1, 2]
SPK_IDS = [5, 6]


@pytest.fixture(scope="module")
def synthetic_data():
    return _make_synthetic_lif_dataset(TRUE_PARAMS, CURRENTS, ROLES, sigma_obs=0.3, seed=42)


def test_subthreshold_fit_recovers_known_E_L_and_R(synthetic_data):
    sub = analysis.fit_subthreshold(synthetic_data, SUB_IDS)
    assert sub.E_L == pytest.approx(TRUE_PARAMS.E_L, abs=0.5)
    assert sub.R == pytest.approx(TRUE_PARAMS.R, rel=0.15)


def test_subthreshold_fit_recovers_known_tau_m(synthetic_data):
    sub = analysis.fit_subthreshold(synthetic_data, SUB_IDS)
    assert sub.tau_m == pytest.approx(TRUE_PARAMS.tau_m, rel=0.25)


def test_full_fit_reproduces_fit_condition_rates_by_construction(synthetic_data):
    params = analysis.full_fit(synthetic_data, SUB_IDS, SPK_IDS)
    report = analysis.validate(synthetic_data, params)
    for cid in SPK_IDS:
        assert report.relative_error[cid] < 0.05


def test_full_fit_predicts_holdout_reasonably_when_model_family_matches(synthetic_data):
    """Unlike the HH-generated workshop data, this synthetic dataset IS
    drawn from an LIF model -- so a correct fitting pipeline should
    extrapolate to the held-out conditions reasonably well too. This is the
    regression test that would fail if fit_subthreshold/fit_spiking had a
    sign error, a unit error, or an off-by-one in condition bookkeeping."""
    params = analysis.full_fit(synthetic_data, SUB_IDS, SPK_IDS)
    report = analysis.validate(synthetic_data, params)
    for cid in [3, 4]:  # holdout conditions, I=4 and I=6 (both above rheobase=3)
        assert report.relative_error[cid] < 0.35


def test_clean_trial_mask_excludes_contaminated_trials():
    """A trial with a spike in an otherwise-subthreshold condition must be
    excluded from the clean mask."""
    data = _make_synthetic_lif_dataset(TRUE_PARAMS, [1.0], ["fit"], n_trials=5, seed=1)
    # manually inject a fake spike into trial 2 of condition 0
    extra = np.array([(0, 2, 50.0)], dtype=data["spikes"].dtype)
    data["spikes"] = np.concatenate([data["spikes"], extra])
    _, trial_ids = analysis.trials_for_condition(data, 0)
    mask = analysis._clean_trial_mask(data, 0, trial_ids)
    assert mask[list(trial_ids).index(2)] == False  # noqa: E712
    assert mask.sum() == len(trial_ids) - 1


def test_bootstrap_ci_bounds_are_ordered_and_bracket_median(synthetic_data):
    summary, draws = analysis.bootstrap_ci(synthetic_data, SUB_IDS, SPK_IDS, n_boot=40, seed=0)
    for k, (lo, mid, hi) in summary.items():
        assert lo <= mid <= hi
        assert len(draws[k]) > 0
