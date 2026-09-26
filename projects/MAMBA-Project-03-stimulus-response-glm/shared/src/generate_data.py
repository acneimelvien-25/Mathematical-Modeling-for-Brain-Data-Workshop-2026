"""
generate_data.py
=================
Deterministic synthetic data generator for MAMBA Project 06:
Stimulus-to-Response GLM.

Running this script regenerates, byte-for-byte given the fixed seed
below:
  - shared/data/trials.csv           (one row per trial)
  - shared/data/data_dictionary.csv  (column documentation)
  - shared/data/README.md            (data description, no ground truth)
  - instructor/reference_outputs/ground_truth.json  (hidden true parameters)
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from models import TrueEffects, DriftParams, simulate_session

MASTER_SEED = 20260928

EFFECTS = TrueEffects(beta0=5.0, beta_45=1.2, beta_90=3.5, beta_135=0.5,
                       beta_contrast=2.0, sigma_noise=1.5)
DRIFT = DriftParams(d_lin=-3.0, d_quad=2.0, d_sin=1.5, d_cos=-1.0)

N_TRIALS = 600
HOLDOUT_FRACTION = 0.15  # last 15% of the session, in time order, reserved

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "data"))
REF_OUT_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "instructor", "reference_outputs"))


def main(data_dir: str = DATA_DIR, ref_out_dir: str = REF_OUT_DIR):
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(ref_out_dir, exist_ok=True)

    rng = np.random.default_rng(MASTER_SEED)
    session = simulate_session(N_TRIALS, EFFECTS, DRIFT, rng)

    n_holdout = int(round(HOLDOUT_FRACTION * N_TRIALS))
    n_fit = N_TRIALS - n_holdout
    trial_role = np.array(["fit"] * n_fit + ["holdout"] * n_holdout)

    trials_path = os.path.join(data_dir, "trials.csv")
    with open(trials_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial_index", "orientation_deg", "contrast_pct", "response", "trial_role"])
        for i in range(N_TRIALS):
            w.writerow([session["trial_index"][i], session["orientation_deg"][i],
                        session["contrast_pct"][i], round(float(session["response"][i]), 4),
                        trial_role[i]])

    dict_path = os.path.join(data_dir, "data_dictionary.csv")
    with open(dict_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "column", "type", "units", "description"])
        rows = [
            ("trials.csv", "trial_index", "int", "-", "0-indexed trial order within the session (this IS the session-time variable)"),
            ("trials.csv", "orientation_deg", "float", "degrees", "Grating orientation shown on this trial: one of 0, 45, 90, 135"),
            ("trials.csv", "contrast_pct", "float", "% Michelson contrast", "Grating contrast shown on this trial: one of 12.5, 25, 50, 100"),
            ("trials.csv", "response", "float", "arbitrary units (peak dF/F)", "Trial-evoked response amplitude, this neuron"),
            ("trials.csv", "trial_role", "str", "-", "'fit' or 'holdout' -- see shared/data/README.md for what this means and how it must be used"),
        ]
        for row in rows:
            w.writerow(row)

    readme_path = os.path.join(data_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(DATA_README_TEMPLATE)

    gt = {
        "master_seed": MASTER_SEED,
        "n_trials": N_TRIALS,
        "n_holdout": n_holdout,
        "holdout_fraction": HOLDOUT_FRACTION,
        "true_effects": {
            "beta0": EFFECTS.beta0, "beta_45": EFFECTS.beta_45, "beta_90": EFFECTS.beta_90,
            "beta_135": EFFECTS.beta_135, "beta_contrast": EFFECTS.beta_contrast,
            "sigma_noise": EFFECTS.sigma_noise,
        },
        "true_drift_params": {
            "d_lin": DRIFT.d_lin, "d_quad": DRIFT.d_quad, "d_sin": DRIFT.d_sin, "d_cos": DRIFT.d_cos,
        },
    }
    with open(os.path.join(ref_out_dir, "ground_truth.json"), "w") as f:
        json.dump(gt, f, indent=2)

    print(f"Generated {N_TRIALS} trials ({n_fit} fit, {n_holdout} holdout).")
    print(f"Wrote: {trials_path}\n       {dict_path}\n       {readme_path}")
    print(f"Wrote (instructor-only): {os.path.join(ref_out_dir, 'ground_truth.json')}")


DATA_README_TEMPLATE = """\
# Project 06 data: single-neuron stimulus-evoked responses

## What was recorded

A single 2-photon-imaged neuron in mouse visual cortex, recorded across
one continuous session of 600 trials. On each trial the animal viewed a
drifting grating at one of four orientations (0, 45, 90, 135 degrees)
and one of four contrasts (12.5, 25, 50, 100 percent). `response` is the
trial-evoked response amplitude (peak dF/F, arbitrary units) -- a single
continuous number per trial.

## Files

- `trials.csv` -- one row per trial, in the ORIGINAL SESSION ORDER (do
  not shuffle it): `trial_index`, `orientation_deg`, `contrast_pct`,
  `response`, `trial_role`.
- `data_dictionary.csv` -- column-by-column description.

## The `trial_role` column

The final 15% of trials (in session-time order) are marked `holdout`;
the rest are marked `fit`. Both are equally valid, equally noisy real
recordings. This is an analysis-plan constraint your group must respect:
estimate every parameter using `fit`-role trials only, and use
`holdout`-role trials only to validate predictions afterward -- never to
estimate anything.

## Was the stimulus schedule fully randomized?

`orientation_deg` was randomized in a block design (every 4 consecutive
trials contains one of each orientation, in random order). Whether
`contrast_pct` was drawn independently of session time, or whether its
distribution shifted across the session, is a question your own analysis
of `trials.csv` should be able to answer -- it is not asserted here
either way.

## Noise model (qualitative)

`response` reflects: the stimulus-driven effects of orientation and
contrast; ordinary trial-to-trial measurement noise; and POSSIBLY a
slow, stimulus-independent change in this neuron's baseline
responsiveness over the course of the session (e.g. from adaptation,
imaging-plane drift, or a slow change in behavioral state) -- something
several 2-photon datasets like this one are known to show. Whether it is
present in this particular recording, and if so, how strongly it affects
your conclusions, is exactly what this exercise asks you to determine.
No further detail is given here.
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=DATA_DIR, help="Output directory for shared/data files")
    parser.add_argument("--instructor-outdir", default=REF_OUT_DIR,
                         help="Output directory for instructor/reference_outputs/ground_truth.json")
    args = parser.parse_args()
    main(data_dir=args.outdir, ref_out_dir=args.instructor_outdir)
