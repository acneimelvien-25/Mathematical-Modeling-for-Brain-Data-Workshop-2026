#!/usr/bin/env python3
"""
generate_data.py
=================
Deterministically synthesizes the "recorded neuron" dataset for MAMBA
Project 01: Membrane to Spike.

A Hodgkin-Huxley-style biophysical neuron (``models.HodgkinHuxleyNeuron``) is
driven by a set of constant-current-plus-noise "current-clamp" protocols,
exactly as in a real intracellular electrophysiology experiment. The script
writes:

  shared/data/voltage_traces.npz     -- noisy recorded voltage, all trials
  shared/data/spike_times.csv        -- detected spike times, all trials
  shared/data/stimulus_protocol.csv  -- per-condition experiment design
  shared/data/data_dictionary.csv    -- column-by-column data dictionary
  instructor/reference_outputs/ground_truth.json  -- HIDDEN from students:
                                          true HH parameters, effective
                                          (tau_m, R, E_L) at rest, and the
                                          fit/holdout split rationale.

Everything here is deterministic given MASTER_SEED: rerunning this script
reproduces the committed data files exactly (checked by
``shared/tests/test_smoke.py::test_data_regeneration_is_deterministic``).

Usage
-----
    python3 generate_data.py [--outdir shared/data] [--seed 20260923]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import HodgkinHuxleyNeuron  # noqa: E402

MASTER_SEED = 20260923

# Experimental design -------------------------------------------------------
# I_dc in uA/cm^2. "fit" conditions are the ones students are told to use for
# parameter estimation; "holdout" conditions are reserved for validation and
# deliberately straddle the class-2 excitability jump of the HH reference
# model (see instructor_guide.md, Section "Why 4 and 6 uA/cm^2").
CONDITIONS = [
    dict(condition_id=0, I_dc=0.0, role="fit"),
    dict(condition_id=1, I_dc=1.0, role="fit"),
    dict(condition_id=2, I_dc=1.7, role="fit"),
    dict(condition_id=3, I_dc=4.0, role="holdout"),
    dict(condition_id=4, I_dc=6.0, role="holdout"),
    dict(condition_id=5, I_dc=8.0, role="fit"),
    dict(condition_id=6, I_dc=14.0, role="fit"),
]

N_TRIALS_PER_CONDITION = 10
T_END_MS = 200.0
DT_SIM_MS = 0.01
RECORD_EVERY = 5              # recorded dt = 0.05 ms
DT_REC_MS = DT_SIM_MS * RECORD_EVERY

SIGMA_OBS_MV = 1.0            # additive recording/observation noise
SIGMA_I_UA = 0.6              # OU input-current noise std
TAU_OU_MS = 3.0               # OU correlation time
SPIKE_JITTER_MS = 0.05        # detector timing-precision jitter, std


def ou_process(rng: np.random.Generator, n_steps: int, dt: float, tau: float, sigma: float):
    """Ornstein-Uhlenbeck noise, mean 0, stationary std `sigma`, correlation
    time `tau`, integrated with the exact discrete-time update."""
    x = np.empty(n_steps + 1)
    x[0] = rng.normal(0.0, sigma)
    a = np.exp(-dt / tau)
    b = sigma * np.sqrt(1 - a ** 2)
    noise = rng.normal(0.0, 1.0, size=n_steps)
    for i in range(n_steps):
        x[i + 1] = a * x[i] + b * noise[i]
    return x


def detect_spikes(t: np.ndarray, V: np.ndarray, threshold_mv: float = 0.0) -> np.ndarray:
    """Detect upward threshold crossings of the clean simulated voltage
    (mimics a hardware spike detector operating on the unfiltered signal,
    decoupled from the observation noise added to the voltage the students
    receive)."""
    above = V > threshold_mv
    crossings = np.where(above[1:] & ~above[:-1])[0]
    # linear interpolation of crossing time for sub-sample precision
    t_spk = []
    for idx in crossings:
        v0, v1 = V[idx], V[idx + 1]
        t0, t1 = t[idx], t[idx + 1]
        frac = (threshold_mv - v0) / (v1 - v0) if v1 != v0 else 0.0
        t_spk.append(t0 + frac * (t1 - t0))
    return np.array(t_spk)


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

    hh = HodgkinHuxleyNeuron()
    V0, m0, h0, n0 = hh.resting_state()
    rest_state = (V0, m0, h0, n0)

    # Ground-truth effective (linearized-at-rest) membrane parameters, used
    # ONLY for instructor grading -- never written to shared/data/.
    p = hh.p
    g_Na_eff = p.g_Na * (m0 ** 3) * h0
    g_K_eff = p.g_K * (n0 ** 4)
    g_total = g_Na_eff + g_K_eff + p.g_L
    R_input_true = 1.0 / g_total
    tau_m_true = p.C_m * R_input_true

    ss = np.random.SeedSequence(args.seed)

    all_rows_V = []       # will become structured arrays
    all_rows_spikes = []
    trial_index = 0

    voltage_by_trial = {}   # (condition_id, trial) -> (t_rec, V_obs)

    for cond in CONDITIONS:
        cid, I_dc = cond["condition_id"], cond["I_dc"]
        for trial in range(N_TRIALS_PER_CONDITION):
            child_seed = ss.spawn(1)[0]
            rng = np.random.default_rng(child_seed)

            n_steps = int(round(T_END_MS / DT_SIM_MS))
            ou = ou_process(rng, n_steps, DT_SIM_MS, TAU_OU_MS, SIGMA_I_UA)
            I_arr = I_dc + ou
            # I_of_t looks up the nearest simulation index (dt fixed, so this
            # is an exact index map, not interpolation)
            def I_of_t(t, I_arr=I_arr):
                idx = int(round(t / DT_SIM_MS))
                idx = min(idx, len(I_arr) - 1)
                return I_arr[idx]

            t_rec, V_clean = hh.simulate(
                I_of_t, T_END_MS, dt=DT_SIM_MS, record_every=RECORD_EVERY,
                init_state=rest_state,
            )

            spike_rng = np.random.default_rng(ss.spawn(1)[0])
            spikes_clean = detect_spikes(t_rec, V_clean, threshold_mv=0.0)
            jitter = spike_rng.normal(0.0, SPIKE_JITTER_MS, size=spikes_clean.shape)
            spikes_obs = np.clip(spikes_clean + jitter, 0.0, T_END_MS)

            obs_rng = np.random.default_rng(ss.spawn(1)[0])
            V_obs = V_clean + obs_rng.normal(0.0, SIGMA_OBS_MV, size=V_clean.shape)

            voltage_by_trial[(cid, trial)] = (t_rec, V_obs.astype(np.float32))
            for t_spk in spikes_obs:
                all_rows_spikes.append((cid, trial, float(t_spk)))

            trial_index += 1

    # --- write voltage_traces.npz (compact binary; one 2D array + index) ---
    n_time = len(next(iter(voltage_by_trial.values()))[0])
    t_axis = next(iter(voltage_by_trial.values()))[0].astype(np.float32)
    keys_sorted = sorted(voltage_by_trial.keys())
    V_matrix = np.stack([voltage_by_trial[k][1] for k in keys_sorted], axis=0)
    cond_col = np.array([k[0] for k in keys_sorted], dtype=np.int16)
    trial_col = np.array([k[1] for k in keys_sorted], dtype=np.int16)

    np.savez_compressed(
        os.path.join(outdir, "voltage_traces.npz"),
        t_ms=t_axis,
        V_obs_mV=V_matrix,
        condition_id=cond_col,
        trial=trial_col,
    )

    # --- write spike_times.csv ---
    with open(os.path.join(outdir, "spike_times.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition_id", "trial", "spike_time_ms"])
        for row in sorted(all_rows_spikes):
            w.writerow(row)

    # --- write stimulus_protocol.csv ---
    with open(os.path.join(outdir, "stimulus_protocol.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition_id", "I_dc_uA_per_cm2", "n_trials", "duration_ms",
                     "dt_recorded_ms", "role"])
        for cond in CONDITIONS:
            w.writerow([cond["condition_id"], cond["I_dc"], N_TRIALS_PER_CONDITION,
                        T_END_MS, DT_REC_MS, cond["role"]])

    # --- write data_dictionary.csv ---
    dict_rows = [
        ("voltage_traces.npz : t_ms", "float32 array, shape (T,)",
         "ms", "Recording time axis, shared by every trial. Sample interval = %.3g ms." % DT_REC_MS),
        ("voltage_traces.npz : V_obs_mV", "float32 array, shape (N_trials, T)",
         "mV", "Observed (noisy) membrane potential for each trial. Row order matches condition_id/trial arrays."),
        ("voltage_traces.npz : condition_id", "int16 array, shape (N_trials,)",
         "-", "Stimulus condition index for each row of V_obs_mV. Join with stimulus_protocol.csv."),
        ("voltage_traces.npz : trial", "int16 array, shape (N_trials,)",
         "-", "Trial number within its condition (0-indexed)."),
        ("spike_times.csv : condition_id", "int", "-", "Stimulus condition index."),
        ("spike_times.csv : trial", "int", "-", "Trial number within its condition (0-indexed)."),
        ("spike_times.csv : spike_time_ms", "float", "ms",
         "Detected spike time within the trial, from an independent spike detector. "
         "Not derived from V_obs_mV directly -- do not expect exact sample alignment."),
        ("stimulus_protocol.csv : condition_id", "int", "-", "Stimulus condition index, 0-6."),
        ("stimulus_protocol.csv : I_dc_uA_per_cm2", "float", "uA/cm^2",
         "Commanded (DC) injected current for this condition. The true injected current "
         "fluctuates around this value -- see handout Section 2 for the noise model."),
        ("stimulus_protocol.csv : n_trials", "int", "-", "Number of repeated trials recorded at this condition."),
        ("stimulus_protocol.csv : duration_ms", "float", "ms", "Recording duration per trial."),
        ("stimulus_protocol.csv : dt_recorded_ms", "float", "ms", "Sample interval of V_obs_mV."),
        ("stimulus_protocol.csv : role", "string", "-",
         "'fit' = intended for parameter estimation. 'holdout' = reserved for validation; "
         "do not use these conditions when fitting model parameters."),
    ]
    with open(os.path.join(outdir, "data_dictionary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["field", "type", "units", "description"])
        for row in dict_rows:
            w.writerow(row)

    # --- data README (student-facing, no ground truth) ---
    readme = f"""# Dataset: Membrane to Spike (Project 01)

Recordings from a single neuron under repeated current-clamp stimulation.
For each of {len(CONDITIONS)} stimulus conditions, {N_TRIALS_PER_CONDITION} trials of
{T_END_MS:.0f} ms were recorded at a sample interval of {DT_REC_MS:.3g} ms.

Files:
- `voltage_traces.npz` -- noisy recorded membrane potential (see data_dictionary.csv)
- `spike_times.csv` -- spike times from an independent spike detector
- `stimulus_protocol.csv` -- the experiment design, including which conditions
  are recommended for **fitting** vs. reserved for **validation**
- `data_dictionary.csv` -- column-by-column description of every field

Noise model (disclosed; parameters are NOT): the injected current fluctuates
around its commanded DC value (a colored, mean-reverting fluctuation), and the
recorded voltage carries additional independent per-sample measurement noise.
Both are present in every trial and are part of the inference problem -- do not
assume noise-free access to the true membrane potential or the true injected
current at any instant.

No ground-truth model identity or parameter values are included in this
folder. That is intentional: your task is to infer a model from the
observations alone.
"""
    with open(os.path.join(outdir, "README.md"), "w") as f:
        f.write(readme)

    # --- instructor-only ground truth ---
    ground_truth = {
        "reference_model": "Hodgkin-Huxley (1952) classic squid giant axon parameters",
        "hh_params": {
            "C_m_uF_cm2": p.C_m, "g_Na_mS_cm2": p.g_Na, "g_K_mS_cm2": p.g_K,
            "g_L_mS_cm2": p.g_L, "E_Na_mV": p.E_Na, "E_K_mV": p.E_K, "E_L_mV": p.E_L,
        },
        "resting_state": {"V0_mV": V0, "m0": m0, "h0": h0, "n0": n0},
        "effective_linearized_parameters_at_rest": {
            "tau_m_eff_ms": tau_m_true,
            "R_input_eff_mV_cm2_per_uA": R_input_true,
            "note": "Computed by linearizing the three HH conductances at the "
                    "resting gating-variable values: g_total = g_Na*m0^3*h0 + "
                    "g_K*n0^4 + g_L; R_input = 1/g_total; tau_m = C_m * R_input. "
                    "This is the reference value used to grade the group's "
                    "regression-based tau_m/R estimate -- see instructor_guide.md.",
        },
        "noise_model": {
            "sigma_obs_mV": SIGMA_OBS_MV, "sigma_I_uA_cm2": SIGMA_I_UA,
            "tau_OU_ms": TAU_OU_MS, "spike_jitter_std_ms": SPIKE_JITTER_MS,
        },
        "conditions": CONDITIONS,
        "seed": args.seed,
    }
    with open(os.path.join(instructor_outdir, "ground_truth.json"), "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Wrote data files to {outdir}")
    print(f"Wrote instructor ground truth to {instructor_outdir}")
    print(f"Resting state: V0={V0:.3f} mV, effective tau_m={tau_m_true:.4f} ms, "
          f"R_input={R_input_true:.4f} mV*cm^2/uA")


if __name__ == "__main__":
    main()
