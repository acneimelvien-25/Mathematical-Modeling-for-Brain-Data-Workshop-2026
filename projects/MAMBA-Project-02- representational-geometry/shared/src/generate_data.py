#!/usr/bin/env python3
"""
generate_data.py
=================
Deterministically synthesizes the "recorded neural population" dataset for
MAMBA Project 03: Representational Geometry.

A population of M neurons with hidden Gaussian-bump tuning curves
(``models.GaussianBumpPopulation``) is presented with a grid of 2-D
structured stimuli and produces trial-level Poisson spike counts. The
script writes:

  shared/data/population_responses.npz  -- trial-level spike counts, all stimuli
  shared/data/stimulus_grid.csv         -- per-stimulus feature coordinates + fit/holdout role
  shared/data/data_dictionary.csv       -- column-by-column data dictionary
  instructor/reference_outputs/ground_truth.json  -- HIDDEN from students:
                                          true tuning parameters and the
                                          fit/holdout split rationale.

Rerunning this script with MASTER_SEED unchanged reproduces the committed
data files exactly (checked by
``shared/tests/test_smoke.py::test_data_regeneration_is_deterministic``).

Usage
-----
    python3 generate_data.py [--outdir shared/data] [--seed 20260925]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import GaussianBumpPopulation, GaussianBumpPopulationParams  # noqa: E402

MASTER_SEED = 20260925

M_NEURONS = 80
PREF_RANGE = 1.3           # preferred locations drawn from [-PREF_RANGE, PREF_RANGE]^2
SIGMA_LO, SIGMA_HI = 0.4, 0.9
GAIN_LO, GAIN_HI = 5.0, 15.0     # Hz
BASELINE_LO, BASELINE_HI = 1.0, 3.0  # Hz

GRID_SIDE = 8              # 8x8 = 64 stimuli
N_HOLDOUT = 12             # interior grid points reserved for validation
N_TRIALS = 10
INTEGRATION_TIME_S = 1.0   # spike-count integration window


def build_stimulus_grid(rng: np.random.SeedSequence):
    vals = np.linspace(-1.0, 1.0, GRID_SIDE)
    xx, yy = np.meshgrid(vals, vals)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (64, 2)
    grid_idx = np.stack([np.repeat(np.arange(GRID_SIDE), GRID_SIDE),
                          np.tile(np.arange(GRID_SIDE), GRID_SIDE)], axis=1)

    # Holdout stimuli are drawn from the INTERIOR of the grid only (never
    # the outer ring), so that holdout evaluation is a genuine
    # interpolation test within the convex hull of the fit stimuli, not an
    # extrapolation test (extrapolation is left as the optional
    # extension -- see the handout).
    interior_mask = ((grid_idx[:, 0] >= 1) & (grid_idx[:, 0] <= GRID_SIDE - 2) &
                      (grid_idx[:, 1] >= 1) & (grid_idx[:, 1] <= GRID_SIDE - 2))
    interior_ids = np.where(interior_mask)[0]

    holdout_rng = np.random.default_rng(rng.spawn(1)[0])
    holdout_ids = np.sort(holdout_rng.choice(interior_ids, size=N_HOLDOUT, replace=False))
    # NOTE: dtype must be wide enough for "holdout" (7 chars) -- building
    # this as a plain Python list (rather than a fixed-width numpy string
    # array inferred from "fit") avoids silent truncation on assignment.
    roles = ["fit"] * grid.shape[0]
    for i in holdout_ids:
        roles[i] = "holdout"
    return grid, roles


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
    grid, roles = build_stimulus_grid(ss)

    pop_rng = np.random.default_rng(ss.spawn(1)[0])
    preferred_xy = pop_rng.uniform(-PREF_RANGE, PREF_RANGE, size=(M_NEURONS, 2))
    sigma = pop_rng.uniform(SIGMA_LO, SIGMA_HI, size=M_NEURONS)
    gain = pop_rng.uniform(GAIN_LO, GAIN_HI, size=M_NEURONS)
    baseline = pop_rng.uniform(BASELINE_LO, BASELINE_HI, size=M_NEURONS)
    params = GaussianBumpPopulationParams(preferred_xy=preferred_xy, sigma=sigma,
                                           gain=gain, baseline=baseline)
    pop = GaussianBumpPopulation(params)

    trial_rng = np.random.default_rng(ss.spawn(1)[0])
    counts = pop.sample_counts(grid, N_TRIALS, INTEGRATION_TIME_S, trial_rng)  # (64, n_trials, M)

    n_stim = grid.shape[0]
    stimulus_id_col = np.repeat(np.arange(n_stim), N_TRIALS).astype(np.int16)
    trial_col = np.tile(np.arange(N_TRIALS), n_stim).astype(np.int16)
    counts_flat = counts.reshape(n_stim * N_TRIALS, M_NEURONS).astype(np.int16)

    np.savez_compressed(
        os.path.join(outdir, "population_responses.npz"),
        spike_counts=counts_flat, stimulus_id=stimulus_id_col, trial=trial_col,
    )

    with open(os.path.join(outdir, "stimulus_grid.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stimulus_id", "feature_x", "feature_y", "n_trials",
                     "integration_time_s", "role"])
        for i in range(n_stim):
            w.writerow([i, grid[i, 0], grid[i, 1], N_TRIALS, INTEGRATION_TIME_S, roles[i]])

    dict_rows = [
        ("population_responses.npz : spike_counts", "int16 array, shape (N_rows, M)", "spikes",
         "Trial-level spike count for every neuron, one row per (stimulus_id, trial) pair. "
         f"M={M_NEURONS} neurons (columns)."),
        ("population_responses.npz : stimulus_id", "int16 array, shape (N_rows,)", "-",
         "Stimulus index for each row of spike_counts. Join with stimulus_grid.csv."),
        ("population_responses.npz : trial", "int16 array, shape (N_rows,)", "-",
         "Trial number within its stimulus (0-indexed)."),
        ("stimulus_grid.csv : stimulus_id", "int", "-", "Stimulus index, 0-63 (an 8x8 grid)."),
        ("stimulus_grid.csv : feature_x", "float", "dimensionless, in [-1,1]",
         "First stimulus feature coordinate (e.g. one axis of a 2-D structured stimulus space)."),
        ("stimulus_grid.csv : feature_y", "float", "dimensionless, in [-1,1]",
         "Second stimulus feature coordinate."),
        ("stimulus_grid.csv : n_trials", "int", "-", "Number of repeated trials recorded at this stimulus."),
        ("stimulus_grid.csv : integration_time_s", "float", "s",
         "Spike-count integration window used for every trial."),
        ("stimulus_grid.csv : role", "string", "-",
         "'fit' = intended for PCA-basis and decoder training. 'holdout' = reserved for "
         "validation; do not use these stimuli when fitting anything."),
    ]
    with open(os.path.join(outdir, "data_dictionary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["field", "type", "units", "description"])
        for row in dict_rows:
            w.writerow(row)

    n_fit = sum(1 for r in roles if r == "fit")
    readme = f"""# Dataset: Representational Geometry (Project 03)

Simulated population recordings from {M_NEURONS} neurons in response to a
structured, 2-dimensional grid of {n_stim} stimuli ({GRID_SIDE}x{GRID_SIDE}),
with {N_TRIALS} repeated trials per stimulus. Each stimulus is described by
two dimensionless feature coordinates in [-1, 1] (think of these as two
independent stimulus properties -- e.g. two axes of a parametric shape or
grating space -- the workshop materials do not commit to a single concrete
sensory modality, since the mathematics does not depend on one).

Files:
- `population_responses.npz` -- trial-level spike counts (see data_dictionary.csv)
- `stimulus_grid.csv` -- the stimulus design, including which stimuli are
  recommended for **fitting** (PCA basis + decoder training) vs. reserved
  for **validation** ({n_fit} fit, {N_HOLDOUT} holdout)
- `data_dictionary.csv` -- column-by-column description of every field

Noise model (disclosed; parameters are NOT): spike counts are generated by
a Poisson process with a stimulus- and neuron-dependent mean rate, over a
fixed {INTEGRATION_TIME_S:.0f} s integration window. This means the
variance of each neuron's response to a repeated stimulus is not a free
parameter -- it is tied to that neuron's mean response on that stimulus
(the defining property of a Poisson process). No information about
individual neurons' tuning (preferred stimulus location, tuning width,
gain, or baseline rate) is included in this folder. That is intentional:
your task is to discover the population's representational structure from
the observations alone.
"""
    with open(os.path.join(outdir, "README.md"), "w") as f:
        f.write(readme)

    ground_truth = {
        "reference_model": "Population of Gaussian-bump-tuned neurons over a 2-D feature space",
        "n_neurons": M_NEURONS,
        "preferred_xy": preferred_xy.tolist(),
        "sigma": sigma.tolist(),
        "gain": gain.tolist(),
        "baseline": baseline.tolist(),
        "grid_side": GRID_SIDE,
        "n_holdout": N_HOLDOUT,
        "holdout_stimulus_ids": [i for i, r in enumerate(roles) if r == "holdout"],
        "integration_time_s": INTEGRATION_TIME_S,
        "n_trials": N_TRIALS,
        "seed": args.seed,
    }
    with open(os.path.join(instructor_outdir, "ground_truth.json"), "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Wrote data files to {outdir}")
    print(f"Wrote instructor ground truth to {instructor_outdir}")
    print(f"{n_fit} fit stimuli, {N_HOLDOUT} holdout stimuli, {M_NEURONS} neurons, "
          f"{N_TRIALS} trials/stimulus")


if __name__ == "__main__":
    main()
