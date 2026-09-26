#!/usr/bin/env python3
"""
generate_data.py
=================
Deterministically synthesizes the "recorded passive membrane" dataset for
MAMBA Project 04: Noisy Neural Dynamics.

A single Ornstein-Uhlenbeck voltage process (``models.OUProcess``) with a
HIDDEN membrane time constant, process noise amplitude, and observation
noise level is recorded across several sessions. Each session has its own
(also hidden) mean voltage ``V_ss`` -- representing session-to-session
differences in ambient synaptic drive -- and its own characteristic
sampling density (mean inter-sample interval), drawn from irregular
(Gamma-distributed) inter-sample intervals rather than a fixed clock. The
script writes:

  shared/data/voltage_sessions.npz  -- irregular (time, voltage) samples, all sessions
  shared/data/session_protocol.csv  -- per-session design (mean dt, n samples, role)
  shared/data/data_dictionary.csv   -- column-by-column data dictionary
  instructor/reference_outputs/ground_truth.json  -- HIDDEN from students:
                                          true tau/sigma_p/sigma_obs/V_ss
                                          and the fit/holdout rationale.

Rerunning this script with MASTER_SEED unchanged reproduces the committed
data files exactly (checked by
``shared/tests/test_smoke.py::test_data_regeneration_is_deterministic``).

Usage
-----
    python3 generate_data.py [--outdir shared/data] [--seed 20260926]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import OUParams, OUProcess  # noqa: E402

MASTER_SEED = 20260926

# Hidden, shared-across-sessions ground truth (the same physical
# compartment is recorded in every session).
TRUE_TAU = 25.0          # ms
TRUE_SIGMA_P = 0.9       # mV / sqrt(ms)
TRUE_SIGMA_OBS = 1.5     # mV

T_TOTAL_MS = 5000.0      # recording duration per session
GAMMA_CV = 0.5           # coefficient of variation of inter-sample intervals

# Each session: (mean_dt_ms, V_ss_mV, role). V_ss differs session to
# session (different ambient input drive); tau/sigma_p/sigma_obs are
# shared and must be recovered by pooling across FIT sessions.
SESSIONS = [
    dict(session_id=0, mean_dt=3.0,   V_ss=-61.0, role="fit"),
    dict(session_id=1, mean_dt=8.0,   V_ss=-57.0, role="fit"),
    dict(session_id=2, mean_dt=15.0,  V_ss=-64.0, role="fit"),
    dict(session_id=3, mean_dt=40.0,  V_ss=-59.0, role="fit"),
    dict(session_id=4, mean_dt=6.0,   V_ss=-62.0, role="holdout"),
    dict(session_id=5, mean_dt=100.0, V_ss=-58.0, role="holdout"),
]


def sample_irregular_times(T_total: float, mean_dt: float, cv: float,
                            rng: np.random.Generator) -> np.ndarray:
    """Gamma-distributed inter-sample intervals with the given mean and
    coefficient of variation (cv=0.5 gives genuinely irregular but not
    degenerate spacing -- not a fixed grid, and not so erratic that
    consecutive samples can be simultaneous)."""
    shape = 1.0 / cv ** 2
    scale = mean_dt / shape
    times = [0.0]
    while times[-1] < T_total:
        times.append(times[-1] + rng.gamma(shape, scale))
    return np.array(times[:-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data"))
    ap.add_argument("--instructor-outdir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "instructor", "reference_outputs"))
    ap.add_argument("--seed", type=int, default=MASTER_SEED)
    args = ap.parse_args()

    outdir = os.path.abspath(args.outdir)
    instructor_outdir = os.path.abspath(args.instructor_outdir)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(instructor_outdir, exist_ok=True)

    ss = np.random.SeedSequence(args.seed)

    all_session_id, all_time, all_V_obs = [], [], []
    session_n_samples = {}

    for sess in SESSIONS:
        sid, mean_dt, V_ss = sess["session_id"], sess["mean_dt"], sess["V_ss"]
        time_rng = np.random.default_rng(ss.spawn(1)[0])
        times = sample_irregular_times(T_TOTAL_MS, mean_dt, GAMMA_CV, time_rng)

        proc_rng = np.random.default_rng(ss.spawn(1)[0])
        ou = OUProcess(OUParams(tau=TRUE_TAU, V_ss=V_ss, sigma_p=TRUE_SIGMA_P))
        V_true = ou.simulate_at_times(times, proc_rng)

        obs_rng = np.random.default_rng(ss.spawn(1)[0])
        V_obs = V_true + obs_rng.normal(0.0, TRUE_SIGMA_OBS, size=len(times))

        all_session_id.extend([sid] * len(times))
        all_time.extend(times.tolist())
        all_V_obs.extend(V_obs.tolist())
        session_n_samples[sid] = len(times)

    np.savez_compressed(
        os.path.join(outdir, "voltage_sessions.npz"),
        session_id=np.array(all_session_id, dtype=np.int16),
        time_ms=np.array(all_time, dtype=np.float64),
        V_obs_mV=np.array(all_V_obs, dtype=np.float32),
    )

    with open(os.path.join(outdir, "session_protocol.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session_id", "mean_dt_ms", "n_samples", "duration_ms",
                     "gamma_cv", "role"])
        for sess in SESSIONS:
            sid = sess["session_id"]
            w.writerow([sid, sess["mean_dt"], session_n_samples[sid], T_TOTAL_MS,
                        GAMMA_CV, sess["role"]])

    dict_rows = [
        ("voltage_sessions.npz : session_id", "int16 array, shape (N_rows,)", "-",
         "Session index for each row. Join with session_protocol.csv."),
        ("voltage_sessions.npz : time_ms", "float64 array, shape (N_rows,)", "ms",
         "Sample time within its session (t=0 is the start of that session's recording). "
         "NOT evenly spaced -- consecutive samples within a session have variable gaps."),
        ("voltage_sessions.npz : V_obs_mV", "float32 array, shape (N_rows,)", "mV",
         "Observed (noisy) membrane voltage at the corresponding time_ms."),
        ("session_protocol.csv : session_id", "int", "-", "Session index, 0-5."),
        ("session_protocol.csv : mean_dt_ms", "float", "ms",
         "The COMMANDED mean inter-sample interval for this session's acquisition. The "
         "ACTUAL sample-to-sample gaps vary around this mean -- see time_ms directly for "
         "the exact, irregular spacing realized in this session."),
        ("session_protocol.csv : n_samples", "int", "-", "Number of samples recorded in this session."),
        ("session_protocol.csv : duration_ms", "float", "ms", "Total recording duration for this session."),
        ("session_protocol.csv : gamma_cv", "float", "-",
         "Coefficient of variation of the inter-sample interval distribution (same for every session)."),
        ("session_protocol.csv : role", "string", "-",
         "'fit' = intended for parameter estimation. 'holdout' = reserved for validation; "
         "do not use these sessions when fitting model parameters."),
    ]
    with open(os.path.join(outdir, "data_dictionary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["field", "type", "units", "description"])
        for row in dict_rows:
            w.writerow(row)

    n_fit = sum(1 for s in SESSIONS if s["role"] == "fit")
    n_holdout = len(SESSIONS) - n_fit
    readme = f"""# Dataset: Noisy Neural Dynamics (Project 04)

Recordings of a single passive (non-spiking) membrane compartment across
{len(SESSIONS)} sessions ({n_fit} fit, {n_holdout} holdout), each
{T_TOTAL_MS/1000:.0f} s long. Sessions differ in their TARGET sampling
density (see `session_protocol.csv`'s `mean_dt_ms`) and in their own mean
voltage level (reflecting session-to-session differences in ambient
synaptic drive) -- but the compartment's own intrinsic dynamics (its
membrane time constant and the amplitude of its input-driven fluctuations)
are the SAME physical property in every session, and recovering them
precisely requires appropriately combining information across sessions.

**Samples within a session are NOT evenly spaced in time.** Consecutive
gaps vary around each session's target mean interval. Do not assume a
fixed sample clock anywhere in your analysis -- use the actual recorded
`time_ms` values.

Files:
- `voltage_sessions.npz` -- irregular (time, voltage) samples, all sessions
- `session_protocol.csv` -- the experiment design, including which sessions
  are recommended for **fitting** vs. reserved for **validation**
- `data_dictionary.csv` -- column-by-column description of every field

Noise model (disclosed; parameters are NOT): the true voltage undergoes
its own intrinsic random fluctuations (present in the underlying dynamics,
continuously, not just at the sampled instants), and the recorded signal
carries additional independent observation noise on top of that at each
sampled instant. No ground-truth parameter values, or the identity of the
underlying process, are included in this folder. That is intentional: your
task is to infer them from the observations alone.
"""
    with open(os.path.join(outdir, "README.md"), "w") as f:
        f.write(readme)

    ground_truth = {
        "reference_model": "Ornstein-Uhlenbeck (leaky, noise-driven) membrane voltage, no spiking",
        "true_tau_ms": TRUE_TAU,
        "true_sigma_p": TRUE_SIGMA_P,
        "true_sigma_obs": TRUE_SIGMA_OBS,
        "true_stationary_std_mV": float(np.sqrt(TRUE_SIGMA_P ** 2 * TRUE_TAU / 2)),
        "sessions": SESSIONS,
        "session_n_samples": session_n_samples,
        "gamma_cv": GAMMA_CV,
        "duration_ms": T_TOTAL_MS,
        "seed": args.seed,
    }
    with open(os.path.join(instructor_outdir, "ground_truth.json"), "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Wrote data files to {outdir}")
    print(f"Wrote instructor ground truth to {instructor_outdir}")
    for sess in SESSIONS:
        sid = sess["session_id"]
        print(f"  session {sid} (mean_dt={sess['mean_dt']}, {sess['role']}): "
              f"n_samples={session_n_samples[sid]}")


if __name__ == "__main__":
    main()
