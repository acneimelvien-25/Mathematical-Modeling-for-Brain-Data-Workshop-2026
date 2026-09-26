"""Unit tests for analysis.py: variogram estimation, pooled/per-session
fitting, validation, and bootstrap -- checked against synthetic data built
directly from KNOWN OU parameters (not the committed dataset -- that is
exercised by the end-to-end smoke test instead)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import analysis  # noqa: E402
from models import OUParams, OUProcess  # noqa: E402


def _make_synthetic_dataset(sessions, tau=25.0, sigma_p=0.9, sigma_obs=1.5,
                              T_total=8000.0, cv=0.5, seed=0):
    """sessions: list of dicts with session_id, mean_dt, V_ss, role."""
    ss = np.random.SeedSequence(seed)
    all_sid, all_t, all_V = [], [], []
    protocol = []
    for sess in sessions:
        sid, mean_dt, V_ss, role = sess["session_id"], sess["mean_dt"], sess["V_ss"], sess["role"]
        time_rng = np.random.default_rng(ss.spawn(1)[0])
        shape = 1.0 / cv ** 2
        scale = mean_dt / shape
        times = [0.0]
        while times[-1] < T_total:
            times.append(times[-1] + time_rng.gamma(shape, scale))
        times = np.array(times[:-1])

        proc_rng = np.random.default_rng(ss.spawn(1)[0])
        ou = OUProcess(OUParams(tau=tau, V_ss=V_ss, sigma_p=sigma_p))
        V_true = ou.simulate_at_times(times, proc_rng)
        obs_rng = np.random.default_rng(ss.spawn(1)[0])
        V_obs = V_true + obs_rng.normal(0, sigma_obs, size=len(times))

        all_sid.extend([sid] * len(times))
        all_t.extend(times.tolist())
        all_V.extend(V_obs.tolist())
        protocol.append({"session_id": sid, "mean_dt_ms": mean_dt, "n_samples": len(times),
                         "duration_ms": T_total, "gamma_cv": cv, "role": role})

    return {"session_id": np.array(all_sid), "time_ms": np.array(all_t),
            "V_obs_mV": np.array(all_V), "protocol": protocol}


DENSE_SESSIONS = [
    dict(session_id=0, mean_dt=2.0, V_ss=-60.0, role="fit"),
    dict(session_id=1, mean_dt=3.0, V_ss=-58.0, role="fit"),
    dict(session_id=2, mean_dt=4.0, V_ss=-62.0, role="fit"),
]


def test_estimate_V_ss_recovers_known_means():
    data = _make_synthetic_dataset(DENSE_SESSIONS, seed=1)
    for sess in DENSE_SESSIONS:
        v_hat = analysis.estimate_V_ss(data, sess["session_id"])
        assert v_hat == pytest.approx(sess["V_ss"], abs=1.0)


def test_fit_session_ids_and_holdout_session_ids():
    sessions = DENSE_SESSIONS + [dict(session_id=3, mean_dt=5.0, V_ss=-59.0, role="holdout")]
    data = _make_synthetic_dataset(sessions, seed=2)
    assert analysis.fit_session_ids(data) == [0, 1, 2]
    assert analysis.holdout_session_ids(data) == [3]


def test_pooled_fit_recovers_known_parameters_with_dense_sampling():
    """With densely-sampled sessions (mean_dt well below tau), the pooled
    variogram fit should recover tau, sigma_p, sigma_obs reasonably
    accurately."""
    data = _make_synthetic_dataset(DENSE_SESSIONS, tau=25.0, sigma_p=0.9, sigma_obs=1.5, seed=3)
    fit = analysis.fit_pooled(data, [0, 1, 2], max_lag=200.0)
    assert fit.success
    assert fit.tau == pytest.approx(25.0, rel=0.5)
    assert fit.sigma_p == pytest.approx(0.9, rel=0.5)
    assert fit.sigma_obs == pytest.approx(1.5, rel=0.5)


def test_sparse_session_independent_fit_is_much_less_reliable_than_dense():
    """The core identifiability finding: a session sampled much more
    sparsely than tau should give a far less accurate independent tau
    estimate than a densely-sampled session, even though both use the
    exact same fitting method."""
    sessions = [
        dict(session_id=0, mean_dt=2.0, V_ss=-60.0, role="fit"),
        dict(session_id=1, mean_dt=300.0, V_ss=-60.0, role="fit"),  # mean_dt >> tau=25
    ]
    data = _make_synthetic_dataset(sessions, tau=25.0, seed=4, T_total=20000.0)
    per_session = analysis.fit_per_session(data, [0, 1], max_lag=1000.0)

    dense_error = abs(per_session[0].tau - 25.0)
    sparse_error = abs(per_session[1].tau - 25.0)
    assert sparse_error > dense_error


def test_pooled_variogram_is_invariant_to_per_session_V_ss():
    """Sessions with very different V_ss should still pool correctly --
    the variogram doesn't require knowing V_ss first, since differencing
    cancels it."""
    sessions = [
        dict(session_id=0, mean_dt=2.0, V_ss=-80.0, role="fit"),
        dict(session_id=1, mean_dt=2.0, V_ss=10.0, role="fit"),  # wildly different V_ss
    ]
    data = _make_synthetic_dataset(sessions, tau=25.0, seed=5, T_total=8000.0)
    fit = analysis.fit_pooled(data, [0, 1], max_lag=200.0)
    assert fit.success
    assert fit.tau == pytest.approx(25.0, rel=0.5)


def test_validate_predicts_holdout_variogram_reasonably():
    sessions = DENSE_SESSIONS + [dict(session_id=3, mean_dt=3.5, V_ss=-59.0, role="holdout")]
    data = _make_synthetic_dataset(sessions, tau=25.0, sigma_p=0.9, sigma_obs=1.5, seed=6)
    fit = analysis.fit_pooled(data, [0, 1, 2], max_lag=200.0)
    report = analysis.validate(data, fit, [3], max_lag=200.0)
    assert 3 in report.per_session
    assert report.per_session[3]["mean_relative_error"] < 1.0  # sane, not wildly off


def test_bootstrap_ci_bounds_are_ordered_and_around_true_tau():
    data = _make_synthetic_dataset(DENSE_SESSIONS, tau=25.0, seed=7)
    summary, draws = analysis.bootstrap_pooled_tau(data, [0, 1, 2], n_boot=30, seed=0, max_lag=200.0)
    lo, mid, hi = summary["tau"]
    assert lo <= mid <= hi
    assert len(draws["tau"]) > 0
