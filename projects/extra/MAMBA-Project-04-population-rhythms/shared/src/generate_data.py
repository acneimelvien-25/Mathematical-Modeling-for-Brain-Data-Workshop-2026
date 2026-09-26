#!/usr/bin/env python3
"""
generate_data.py
=================
Deterministically synthesizes the "recorded population activity" dataset
for MAMBA Project 02: Population Rhythms.

A Wilson-Cowan-style two-population rate model (``models.WilsonCowanPopulation``)
with HIDDEN ground-truth parameters is simulated under several levels of a
control variable, I_I (external drive to the inhibitory population) --
analogous to a pharmacological or optogenetic manipulation of inhibitory
tone in a real E/I microcircuit. The script writes:

  shared/data/population_traces.npz  -- noisy recorded (E, I) activity, all trials
  shared/data/stimulus_protocol.csv  -- per-condition experiment design
  shared/data/data_dictionary.csv    -- column-by-column data dictionary
  instructor/reference_outputs/ground_truth.json  -- HIDDEN from students:
                                          true Wilson-Cowan parameters, the
                                          true fixed point and Jacobian
                                          eigenvalues at every condition,
                                          and the fit/holdout rationale.

Rerunning this script with MASTER_SEED unchanged reproduces the committed
data files exactly (checked by
``shared/tests/test_smoke.py::test_data_regeneration_is_deterministic``).

Usage
-----
    python3 generate_data.py [--outdir shared/data] [--seed 20260924]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import WilsonCowanParams, WilsonCowanPopulation  # noqa: E402

MASTER_SEED = 20260924

# Ground-truth parameters (never disclosed to students). Found by a
# randomized search for a parameter set with a clean Hopf bifurcation as
# I_I varies over a plausible range -- see instructor_guide.md Section 2
# for the search procedure and why this particular regime was chosen.
TRUE_PARAMS = WilsonCowanParams(
    tau_E=9.383192764607815, tau_I=16.181748555258448,
    w_EE=10.93652298079397, w_EI=11.698217414642752,
    w_IE=13.600568975933758, w_II=0.7523397388964193,
    a_E=1.9423928328489133, theta_E=4.751426573298685,
    a_I=1.677857847008384, theta_I=5.708772875142998,
    I_E=3.4390675155534436,
)

# Experimental design: I_I in the model's native input units. "fit"
# conditions sit safely inside the stable-spiral regime (complex Jacobian
# eigenvalues with negative real part -- damped "quasi-cycle" fluctuations,
# no sustained oscillation). "holdout" conditions sit on the far side of
# the Hopf bifurcation (positive real part -- sustained, nonlinear limit-
# cycle oscillation). See instructor_guide.md for the exact bifurcation
# location and why 3.0/2.0 were chosen.
CONDITIONS = [
    dict(condition_id=0, I_I=5.0, role="fit"),
    dict(condition_id=1, I_I=4.5, role="fit"),
    dict(condition_id=2, I_I=4.0, role="fit"),
    dict(condition_id=3, I_I=3.5, role="fit"),
    dict(condition_id=4, I_I=3.0, role="holdout"),
    dict(condition_id=5, I_I=2.0, role="holdout"),
]

N_TRIALS_PER_CONDITION = 8
BURN_IN_MS = 500.0        # discarded before recording, lets transients settle
T_RECORD_MS = 10000.0     # 10 s recorded per trial
DT_SIM_MS = 0.25
RECORD_EVERY = 2          # recorded dt = 0.5 ms
DT_REC_MS = DT_SIM_MS * RECORD_EVERY

SIGMA_PROC = 0.015        # intrinsic (process) noise amplitude, per sqrt(ms)
SIGMA_OBS = 0.010         # additive observation/measurement noise, same units as E, I


def simulate_recorded_trial(wc: WilsonCowanPopulation, I_I: float, seed_seq,
                             init):
    """Burn in from `init`, then record T_RECORD_MS of stochastic activity,
    downsampled to the recording interval, with observation noise added."""
    rng = np.random.default_rng(seed_seq)
    # burn-in (stochastic, so the population relaxes onto its natural
    # fluctuation distribution around the fixed point, not a noise-free point)
    t_b, E_b, I_b = wc.simulate_stochastic(I_I, BURN_IN_MS, DT_SIM_MS, SIGMA_PROC,
                                            rng, init=init)
    init_after_burn = (E_b[-1], I_b[-1])
    t, E, I = wc.simulate_stochastic(I_I, T_RECORD_MS, DT_SIM_MS, SIGMA_PROC, rng,
                                      init=init_after_burn)
    # downsample to the recording interval
    E_rec = E[::RECORD_EVERY]
    I_rec = I[::RECORD_EVERY]
    t_rec = t[::RECORD_EVERY]
    # independent observation noise, added after downsampling (mimics a
    # separate measurement/readout noise source on top of the true,
    # already-noisy population activity)
    obs_rng = np.random.default_rng(seed_seq.spawn(1)[0])
    E_obs = E_rec + obs_rng.normal(0.0, SIGMA_OBS, size=E_rec.shape)
    I_obs = I_rec + obs_rng.normal(0.0, SIGMA_OBS, size=I_rec.shape)
    return t_rec, E_obs.astype(np.float32), I_obs.astype(np.float32)


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

    wc = WilsonCowanPopulation(TRUE_PARAMS)
    ss = np.random.SeedSequence(args.seed)

    fixed_points = {}
    jacobians = {}
    eigenvalues = {}
    init = (0.3, 0.3)
    for cond in CONDITIONS:
        cid, I_I = cond["condition_id"], cond["I_I"]
        E_star, I_star = wc.fixed_point(I_I, init=init)
        init = (E_star, I_star)  # warm-start the next (nearby) condition
        J = wc.jacobian(E_star, I_star, I_I)
        eig = np.linalg.eigvals(J)
        fixed_points[cid] = (E_star, I_star)
        jacobians[cid] = J
        eigenvalues[cid] = eig

    keys_sorted = []
    all_cond, all_trial, all_E, all_I = [], [], [], []
    t_axis = None
    for cond in CONDITIONS:
        cid, I_I = cond["condition_id"], cond["I_I"]
        E_star, I_star = fixed_points[cid]
        for trial in range(N_TRIALS_PER_CONDITION):
            child_seed = ss.spawn(1)[0]
            t_rec, E_obs, I_obs = simulate_recorded_trial(
                wc, I_I, child_seed, init=(E_star, I_star))
            if t_axis is None:
                t_axis = t_rec.astype(np.float32)
            all_cond.append(cid)
            all_trial.append(trial)
            all_E.append(E_obs)
            all_I.append(I_obs)
            keys_sorted.append((cid, trial))

    E_matrix = np.stack(all_E, axis=0)
    I_matrix = np.stack(all_I, axis=0)
    cond_col = np.array(all_cond, dtype=np.int16)
    trial_col = np.array(all_trial, dtype=np.int16)

    np.savez_compressed(
        os.path.join(outdir, "population_traces.npz"),
        t_ms=t_axis, E_obs=E_matrix, I_obs=I_matrix,
        condition_id=cond_col, trial=trial_col,
    )

    with open(os.path.join(outdir, "stimulus_protocol.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition_id", "I_I", "n_trials", "burn_in_ms",
                     "duration_ms", "dt_recorded_ms", "role"])
        for cond in CONDITIONS:
            w.writerow([cond["condition_id"], cond["I_I"], N_TRIALS_PER_CONDITION,
                        BURN_IN_MS, T_RECORD_MS, DT_REC_MS, cond["role"]])

    dict_rows = [
        ("population_traces.npz : t_ms", "float32 array, shape (T,)", "ms",
         "Recording time axis (t=0 is the start of the RECORDED segment, i.e. "
         "after the burn-in period has already elapsed). Shared by every trial."),
        ("population_traces.npz : E_obs", "float32 array, shape (N_trials, T)", "-",
         "Observed (noisy) excitatory population activity for each trial."),
        ("population_traces.npz : I_obs", "float32 array, shape (N_trials, T)", "-",
         "Observed (noisy) inhibitory population activity for each trial."),
        ("population_traces.npz : condition_id", "int16 array, shape (N_trials,)", "-",
         "Stimulus condition index for each row of E_obs/I_obs. Join with stimulus_protocol.csv."),
        ("population_traces.npz : trial", "int16 array, shape (N_trials,)", "-",
         "Trial number within its condition (0-indexed)."),
        ("stimulus_protocol.csv : condition_id", "int", "-", "Stimulus condition index, 0-5."),
        ("stimulus_protocol.csv : I_I", "float", "model input units",
         "Commanded external drive to the inhibitory population for this condition "
         "(the experiment's control variable -- analogous to a pharmacological or "
         "optogenetic manipulation of inhibitory tone). This value IS given; the "
         "model's other parameters are not."),
        ("stimulus_protocol.csv : n_trials", "int", "-", "Number of repeated trials recorded at this condition."),
        ("stimulus_protocol.csv : burn_in_ms", "float", "ms",
         "Duration simulated (stochastically) and DISCARDED before recording began, "
         "to let the population settle onto its stationary fluctuation distribution."),
        ("stimulus_protocol.csv : duration_ms", "float", "ms", "Recorded duration per trial (after burn-in)."),
        ("stimulus_protocol.csv : dt_recorded_ms", "float", "ms", "Sample interval of E_obs / I_obs."),
        ("stimulus_protocol.csv : role", "string", "-",
         "'fit' = intended for parameter estimation. 'holdout' = reserved for validation; "
         "do not use these conditions when fitting model parameters."),
        ("noise_calibration.json : sigma_obs", "float", "same as E_obs/I_obs",
         "KNOWN (disclosed) measurement-noise standard deviation from an independent "
         "instrument calibration. See noise_calibration.json for details."),
    ]
    with open(os.path.join(outdir, "data_dictionary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["field", "type", "units", "description"])
        for row in dict_rows:
            w.writerow(row)

    # Disclosed instrument calibration: the recording system's measurement
    # noise standard deviation is treated as a KNOWN, separately-calibrated
    # constant (unlike the hidden model parameters) -- see
    # shared/data/README.md and instructor_guide.md for why this is needed.
    with open(os.path.join(outdir, "noise_calibration.json"), "w") as f:
        json.dump({
            "sigma_obs": SIGMA_OBS,
            "units": "same units as E_obs / I_obs (dimensionless population activity)",
            "description": (
                "Standard deviation of the recording system's additive "
                "measurement noise, from an independent instrument calibration "
                "(e.g. a blank/no-signal recording). This value is KNOWN and "
                "may be used directly in your analysis; it is not something "
                "you need to estimate from the population activity data "
                "itself. It does NOT include the population's own intrinsic "
                "activity fluctuations, which are not disclosed."
            ),
        }, f, indent=2)

    readme = f"""# Dataset: Population Rhythms (Project 02)

Recordings of two-population (excitatory E, inhibitory I) activity from a
single "preparation," under {len(CONDITIONS)} levels of a control variable
(external drive to the inhibitory population, `I_I`) -- think of this as a
pharmacological or optogenetic manipulation of inhibitory tone. For each
condition, {N_TRIALS_PER_CONDITION} trials of {T_RECORD_MS/1000:.0f} s were
recorded at a sample interval of {DT_REC_MS:.3g} ms, after a
{BURN_IN_MS:.0f} ms burn-in (not included in the recorded data) to let the
population settle onto its natural fluctuation regime.

Files:
- `population_traces.npz` -- noisy recorded (E, I) activity (see data_dictionary.csv)
- `stimulus_protocol.csv` -- the experiment design, including which conditions
  are recommended for **fitting** vs. reserved for **validation**
- `data_dictionary.csv` -- column-by-column description of every field
- `noise_calibration.json` -- the recording system's KNOWN measurement-noise
  standard deviation (from an independent instrument calibration -- see
  below for why this one noise parameter is disclosed when nothing else is)

Noise model: the population activity itself evolves with intrinsic
stochastic fluctuations (present in the underlying dynamics, not just the
recording) -- this part of the noise model is NOT disclosed. On top of
that, the recorded signal carries additional independent observation
(measurement) noise, and THIS noise's standard deviation is disclosed in
`noise_calibration.json`, as it would be from a real instrument's
calibration record. Think about why you might need to know this value
specifically, given what kind of quantity you are trying to estimate from
short, closely-spaced samples of a noisy signal.

No ground-truth model parameters, fixed points, or stability properties are
included in this folder. That is intentional: your task is to infer them
from the observations alone.
"""
    with open(os.path.join(outdir, "README.md"), "w") as f:
        f.write(readme)

    ground_truth = {
        "reference_model": "Two-population Wilson-Cowan-style rate model (logistic S)",        "true_params": {
            "tau_E_ms": TRUE_PARAMS.tau_E, "tau_I_ms": TRUE_PARAMS.tau_I,
            "w_EE": TRUE_PARAMS.w_EE, "w_EI": TRUE_PARAMS.w_EI,
            "w_IE": TRUE_PARAMS.w_IE, "w_II": TRUE_PARAMS.w_II,
            "a_E": TRUE_PARAMS.a_E, "theta_E": TRUE_PARAMS.theta_E,
            "a_I": TRUE_PARAMS.a_I, "theta_I": TRUE_PARAMS.theta_I,
            "I_E": TRUE_PARAMS.I_E,
        },
        "per_condition": {
            str(cond["condition_id"]): {
                "I_I": cond["I_I"], "role": cond["role"],
                "fixed_point_E_I": list(fixed_points[cond["condition_id"]]),
                "jacobian": jacobians[cond["condition_id"]].tolist(),
                "eigenvalues_real": [float(e.real) for e in eigenvalues[cond["condition_id"]]],
                "eigenvalues_imag": [float(e.imag) for e in eigenvalues[cond["condition_id"]]],
            }
            for cond in CONDITIONS
        },
        "noise_model": {"sigma_process": SIGMA_PROC, "sigma_obs": SIGMA_OBS},
        "seed": args.seed,
    }
    with open(os.path.join(instructor_outdir, "ground_truth.json"), "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Wrote data files to {outdir}")
    print(f"Wrote instructor ground truth to {instructor_outdir}")
    for cond in CONDITIONS:
        cid = cond["condition_id"]
        eig = eigenvalues[cid]
        print(f"  condition {cid} (I_I={cond['I_I']}, {cond['role']}): "
              f"fixed point={fixed_points[cid]}, eig={eig}")


if __name__ == "__main__":
    main()
