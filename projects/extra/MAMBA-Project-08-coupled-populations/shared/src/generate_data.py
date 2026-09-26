"""
generate_data.py
=================
Deterministic synthetic data generator for MAMBA Project 08: Coupled
Populations.

Running this script regenerates, byte-for-byte given the fixed seed
below:
  - shared/data/fit_timeseries.csv       (noisy, subthreshold node activity)
  - shared/data/holdout_timeseries.csv   (noise-free, near/above-threshold activity)
  - shared/data/data_dictionary.csv
  - shared/data/README.md
  - instructor/reference_outputs/ground_truth.json (hidden true parameters)
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from models import (
    RingParams, simulate_directed_ring, critical_coupling,
    onset_angular_frequency, most_unstable_mode,
)

MASTER_SEED = 20260930

N_NODES = 5
TAU = 1.0
G_TRUE = 1.0  # tanh'(0), exact
SIGMA_NOISE = 0.05

FIT_W = [-0.4, -0.6, -0.8, -1.0, -1.1]
HOLDOUT_W = [-1.3, -1.6, -2.0, -2.5]

FIT_T = 1000.0
FIT_DT = 0.01
FIT_SAVE_EVERY = 5  # save every 5th integration step -> effective dt 0.05

HOLDOUT_T = 600.0
HOLDOUT_DT = 0.01
HOLDOUT_SAVE_EVERY = 5

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "data"))
REF_OUT_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "instructor", "reference_outputs"))


def main(data_dir: str = DATA_DIR, ref_out_dir: str = REF_OUT_DIR):
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(ref_out_dir, exist_ok=True)

    params = RingParams(n_nodes=N_NODES, tau=TAU)
    rng = np.random.default_rng(MASTER_SEED)

    # ---- FIT data: noisy, subthreshold ----
    fit_path = os.path.join(data_dir, "fit_timeseries.csv")
    with open(fit_path, "w", newline="") as f:
        w_csv = csv.writer(f)
        w_csv.writerow(["w", "time"] + [f"x{i}" for i in range(N_NODES)])
        for w in FIT_W:
            x = simulate_directed_ring(w, params, FIT_T, FIT_DT, sigma=SIGMA_NOISE, rng=rng)
            x_sub = x[::FIT_SAVE_EVERY]
            times = np.arange(len(x_sub)) * FIT_DT * FIT_SAVE_EVERY
            for t_idx in range(len(x_sub)):
                w_csv.writerow([w, round(float(times[t_idx]), 4)] +
                                [round(float(v), 6) for v in x_sub[t_idx]])

    # ---- HOLDOUT data: noise-free, near/above threshold ----
    holdout_path = os.path.join(data_dir, "holdout_timeseries.csv")
    with open(holdout_path, "w", newline="") as f:
        w_csv = csv.writer(f)
        w_csv.writerow(["w", "time"] + [f"x{i}" for i in range(N_NODES)])
        for w in HOLDOUT_W:
            x0 = 0.05 * rng.standard_normal(N_NODES)
            x = simulate_directed_ring(w, params, HOLDOUT_T, HOLDOUT_DT, sigma=0.0, x0=x0)
            x_sub = x[::HOLDOUT_SAVE_EVERY]
            times = np.arange(len(x_sub)) * HOLDOUT_DT * HOLDOUT_SAVE_EVERY
            for t_idx in range(len(x_sub)):
                w_csv.writerow([w, round(float(times[t_idx]), 4)] +
                                [round(float(v), 6) for v in x_sub[t_idx]])

    # ---- data_dictionary.csv ----
    dict_path = os.path.join(data_dir, "data_dictionary.csv")
    with open(dict_path, "w", newline="") as f:
        w_csv = csv.writer(f)
        w_csv.writerow(["file", "column", "type", "units", "description"])
        rows = [
            ("fit_timeseries.csv", "w", "float", "-", "Coupling strength for this block of rows (subthreshold; see README)"),
            ("fit_timeseries.csv", "time", "float", "time units (tau=1)", "Time within this w-block's simulation"),
            ("fit_timeseries.csv", "x0 .. x4", "float", "arbitrary activity units", "Node i's activity at this time (noisy)"),
            ("holdout_timeseries.csv", "w", "float", "-", "Coupling strength for this block of rows (near/above threshold; see README)"),
            ("holdout_timeseries.csv", "time", "float", "time units (tau=1)", "Time within this w-block's simulation"),
            ("holdout_timeseries.csv", "x0 .. x4", "float", "arbitrary activity units", "Node i's activity at this time (noise-free)"),
        ]
        for row in rows:
            w_csv.writerow(row)

    # ---- README.md ----
    readme_path = os.path.join(data_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(DATA_README_TEMPLATE)

    # ---- ground_truth.json (instructor only) ----
    mode_k = most_unstable_mode(G_TRUE, params)
    w_crit = critical_coupling(G_TRUE, params, mode_k)
    onset_freq = onset_angular_frequency(params, mode_k)
    phase_lag_deg = 360.0 * mode_k / N_NODES

    gt = {
        "master_seed": MASTER_SEED,
        "n_nodes": N_NODES,
        "tau": TAU,
        "g_true": G_TRUE,
        "sigma_noise": SIGMA_NOISE,
        "fit_w": FIT_W,
        "holdout_w": HOLDOUT_W,
        "most_unstable_mode_k": mode_k,
        "critical_w_true": w_crit,
        "onset_angular_frequency_true": onset_freq,
        "phase_lag_deg_true": phase_lag_deg,
    }
    with open(os.path.join(ref_out_dir, "ground_truth.json"), "w") as f:
        json.dump(gt, f, indent=2)

    print(f"Generated FIT data for w in {FIT_W}, HOLDOUT data for w in {HOLDOUT_W}.")
    print(f"Wrote: {fit_path}\n       {holdout_path}\n       {dict_path}\n       {readme_path}")
    print(f"Wrote (instructor-only): {os.path.join(ref_out_dir, 'ground_truth.json')}")


DATA_README_TEMPLATE = """\
# Project 08 data: coupled-population ring activity

## What was recorded

Five simulated neural populations ("nodes"), arranged so that each node
receives input from exactly one other node -- a directed ring. Each
node's activity was recorded at several values of the coupling strength
`w`, split into two files:

- `fit_timeseries.csv`: node activity recorded at SUBTHRESHOLD coupling
  strengths (the network sits near a stable equilibrium, with visible
  noisy fluctuations around it).
- `holdout_timeseries.csv`: node activity recorded at coupling strengths
  at or above where the network is expected to become unstable, with
  no added recording noise (any structure you see reflects the
  network's own dynamics, not measurement noise).

Use `fit_timeseries.csv` ONLY for estimation (Tasks T1-T3); use
`holdout_timeseries.csv` ONLY to validate your predictions afterward
(Task T4 onward) -- never to fit or tune anything.

## Files

- `fit_timeseries.csv` -- columns: `w`, `time`, `x0`, `x1`, `x2`, `x3`, `x4`.
- `holdout_timeseries.csv` -- same columns, different `w` values.
- `data_dictionary.csv` -- column-by-column description.

## What is NOT given

The exact functional form of each node's dynamics, the sign or
topology of the coupling, and the gain of the nonlinearity relating one
node's activity to its influence on the next are all things your group
is expected to investigate and/or estimate from the data -- none of
these are stated directly here. (The handout gives you a starting
functional FORM to test against the data; confirming it fits, and
estimating its free parameter(s), is Task T1-T3's job.)
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=DATA_DIR, help="Output directory for shared/data files")
    parser.add_argument("--instructor-outdir", default=REF_OUT_DIR,
                         help="Output directory for instructor/reference_outputs/ground_truth.json")
    args = parser.parse_args()
    main(data_dir=args.outdir, ref_out_dir=args.instructor_outdir)
