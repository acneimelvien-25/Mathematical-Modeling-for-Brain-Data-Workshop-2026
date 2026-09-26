"""
generate_data.py
=================
Deterministic synthetic data generator for MAMBA Project 05: Spike-Count
Uncertainty.

Scenario: a single extracellularly recorded visual-cortex neuron, tested
with drifting-grating stimuli at five contrast levels, ~250 repeated
trials per contrast. The neuron's trial-to-trial spike count reflects two
things: its (contrast-dependent) mean firing rate, and a session-wide,
condition-independent trial-to-trial excitability ("gain") fluctuation --
see models.py for the full generative model and its exact statistics.

Running this script regenerates, byte-for-byte given the fixed seed below:
  - shared/data/spike_times.csv       (long-format spike table)
  - shared/data/trial_protocol.csv    (one row per trial; fit/holdout role)
  - shared/data/data_dictionary.csv   (column documentation)
  - shared/data/README.md             (data description, no ground truth)
  - instructor/reference_outputs/ground_truth.json  (hidden true parameters)

Nothing about sigma_g, the tuning-curve parameters, or the fit/holdout
window split is written to any student-facing file.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from models import (
    TuningCurveParams,
    GainNoiseParams,
    naka_rushton,
    generate_binned_spike_counts,
)

# ---------------------------------------------------------------------------
# Ground truth (hidden from participants)
# ---------------------------------------------------------------------------

MASTER_SEED = 20260927

TUNING = TuningCurveParams(R0=2.0, Rmax=38.0, c50=25.0, n=2.5)
GAIN = GainNoiseParams(sigma_g=0.18)

DT_MS = 1.0                      # bin width, ms
TRIAL_DURATION_MS = 800.0        # every trial is recorded this long
N_TRIALS_PER_CONDITION = 250

CONTRASTS_PCT = [6.0, 12.0, 25.0, 50.0, 100.0]
FIT_CONTRASTS = [6.0, 25.0, 100.0]
HOLDOUT_CONTRASTS = [12.0, 50.0]

WINDOWS_MS = [20.0, 50.0, 100.0, 200.0, 400.0, 800.0]
FIT_WINDOWS_MS = [20.0, 50.0, 100.0, 200.0, 400.0]
HOLDOUT_WINDOW_MS = [800.0]

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "data"))
REF_OUT_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "instructor", "reference_outputs"))


def condition_role(c: float) -> str:
    return "fit" if c in FIT_CONTRASTS else "holdout"


def main(data_dir: str = DATA_DIR, ref_out_dir: str = REF_OUT_DIR):
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(ref_out_dir, exist_ok=True)

    rng = np.random.default_rng(MASTER_SEED)
    n_bins_max = int(round(TRIAL_DURATION_MS / DT_MS))

    lam_true = {c: float(naka_rushton(np.array([c]), TUNING)[0]) for c in CONTRASTS_PCT}

    spike_rows = []       # (trial_id, condition_pct, spike_time_ms)
    protocol_rows = []    # (trial_id, condition_pct, condition_role, trial_duration_ms)

    trial_id = 0
    all_bins = {}  # condition_pct -> (n_trials, n_bins_max) array, kept only for reference-output computations
    # IMPORTANT: iterate conditions in a FIXED, documented order so that the
    # random stream (and hence the committed data) is exactly reproducible.
    for c in CONTRASTS_PCT:
        bins = generate_binned_spike_counts(
            lam_hz=lam_true[c],
            n_trials=N_TRIALS_PER_CONDITION,
            dt_ms=DT_MS,
            n_bins_max=n_bins_max,
            gain_params=GAIN,
            rng=rng,
        )
        all_bins[c] = bins
        role = condition_role(c)
        for local_i in range(N_TRIALS_PER_CONDITION):
            protocol_rows.append((trial_id, c, role, TRIAL_DURATION_MS))
            spike_bin_idx = np.nonzero(bins[local_i])[0]
            # bin k covers [k*dt, (k+1)*dt); record the spike at the bin's
            # left edge plus half a bin, in ms, rounded to 0.1 ms.
            spike_times = np.round((spike_bin_idx + 0.5) * DT_MS, 1)
            for t in spike_times:
                spike_rows.append((trial_id, c, t))
            trial_id += 1

    # ---- write shared/data/trial_protocol.csv ----
    protocol_path = os.path.join(data_dir, "trial_protocol.csv")
    with open(protocol_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial_id", "condition_pct", "condition_role", "trial_duration_ms"])
        for row in protocol_rows:
            w.writerow(row)

    # ---- write shared/data/spike_times.csv ----
    spikes_path = os.path.join(data_dir, "spike_times.csv")
    with open(spikes_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial_id", "condition_pct", "spike_time_ms"])
        for row in spike_rows:
            w.writerow(row)

    # ---- write shared/data/data_dictionary.csv ----
    dict_path = os.path.join(data_dir, "data_dictionary.csv")
    with open(dict_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "column", "type", "units", "description"])
        rows = [
            ("trial_protocol.csv", "trial_id", "int", "-", "Unique trial identifier, 0..1249; joins to spike_times.csv"),
            ("trial_protocol.csv", "condition_pct", "float", "% Michelson contrast", "Stimulus contrast for this trial (one of 6, 12, 25, 50, 100)"),
            ("trial_protocol.csv", "condition_role", "str", "-", "'fit' or 'holdout' -- see shared/data/README.md for what this means and how it must be used"),
            ("trial_protocol.csv", "trial_duration_ms", "float", "ms", "Total recorded duration of this trial from stimulus onset (t=0)"),
            ("spike_times.csv", "trial_id", "int", "-", "Joins to trial_protocol.csv"),
            ("spike_times.csv", "condition_pct", "float", "% Michelson contrast", "Redundant with trial_protocol.csv, included for convenience"),
            ("spike_times.csv", "spike_time_ms", "float", "ms", "Time of a single spike relative to stimulus onset (t=0) in this trial; one row per spike (trials with 0 spikes contribute no rows)"),
        ]
        for row in rows:
            w.writerow(row)

    # ---- write shared/data/README.md ----
    readme_path = os.path.join(data_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(DATA_README_TEMPLATE)

    # ---- write instructor/reference_outputs/ground_truth.json ----
    gt = {
        "master_seed": MASTER_SEED,
        "tuning_curve": {"R0_Hz": TUNING.R0, "Rmax_Hz": TUNING.Rmax, "c50_pct": TUNING.c50, "n": TUNING.n},
        "sigma_g_true": GAIN.sigma_g,
        "sigma_g2_true": GAIN.sigma_g ** 2,
        "dt_ms": DT_MS,
        "trial_duration_ms": TRIAL_DURATION_MS,
        "n_trials_per_condition": N_TRIALS_PER_CONDITION,
        "contrasts_pct": CONTRASTS_PCT,
        "fit_contrasts_pct": FIT_CONTRASTS,
        "holdout_contrasts_pct": HOLDOUT_CONTRASTS,
        "windows_ms": WINDOWS_MS,
        "fit_windows_ms": FIT_WINDOWS_MS,
        "holdout_window_ms": HOLDOUT_WINDOW_MS,
        "lambda_true_hz": lam_true,
    }
    with open(os.path.join(ref_out_dir, "ground_truth.json"), "w") as f:
        json.dump(gt, f, indent=2)

    n_spikes = len(spike_rows)
    n_trials_total = trial_id
    print(f"Generated {n_trials_total} trials across {len(CONTRASTS_PCT)} conditions, {n_spikes} total spikes.")
    print(f"Wrote: {protocol_path}\n       {spikes_path}\n       {dict_path}\n       {readme_path}")
    print(f"Wrote (instructor-only): {os.path.join(ref_out_dir, 'ground_truth.json')}")


DATA_README_TEMPLATE = """\
# Project 05 data: spike counts across contrast conditions

## What was recorded

A single extracellularly recorded neuron (visual cortex, awake animal),
tested with a drifting-grating stimulus at five contrast levels:
6%, 12%, 25%, 50%, and 100% Michelson contrast. Each contrast was
presented for 250 trials, in randomly interleaved order during the
recording session (trial order does not matter for this exercise and is
not preserved). Every trial is 800 ms long, timed from stimulus onset
(t = 0).

## Files

- `spike_times.csv` -- one row per spike: `trial_id`, `condition_pct`,
  `spike_time_ms` (time from stimulus onset). Trials with zero spikes
  contribute no rows -- do not assume every `trial_id` appears here.
- `trial_protocol.csv` -- one row per trial (1250 rows total: 5 conditions
  x 250 trials): `trial_id`, `condition_pct`, `condition_role`,
  `trial_duration_ms`.
- `data_dictionary.csv` -- column-by-column description of the two files
  above.

## The `condition_role` column

Each contrast condition is labeled `fit` or `holdout` in
`trial_protocol.csv`. This is not a data-quality label -- both roles
contain equally valid, equally noisy real recordings. It is an
analysis-plan constraint your group must respect:

- Any parameter that is shared across conditions (i.e. assumed to be a
  property of the recording session as a whole, not of any one stimulus)
  may be ESTIMATED using `fit`-role trials only.
- `holdout`-role trials must be reserved purely for VALIDATING
  predictions made from parameters estimated on `fit` trials -- never
  used to estimate those parameters in the first place.

The exercise handout also asks you to reserve a portion of the
*counting-window range* the same way (see handout Section 2) -- that
split is a matter of how you choose to analyze this same spike-time
data, not a property of the data files themselves, so it is not encoded
as a column here.

## Noise model (qualitative)

Spike generation on each trial reflects two sources of variability:

1. Ordinary point-process ("shot") noise: even at a perfectly constant
   underlying rate, the exact number and timing of spikes in a fixed
   window varies from trial to trial.
2. A trial-to-trial fluctuation in the neuron's overall excitability
   (sometimes attributed to arousal, attention, or other slow network
   state changes), which very likely differs from trial to trial. Its
   statistical size and character are exactly what this exercise asks
   you to characterize -- no value for it is given here.

No further detail about the noise model is provided in student-facing
files. Deriving what its presence or absence would predict about how
spike-count variability should depend on the counting-window length is
the mathematical heart of this exercise (see the handout).
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=DATA_DIR, help="Output directory for shared/data files")
    parser.add_argument("--instructor-outdir", default=REF_OUT_DIR,
                         help="Output directory for instructor/reference_outputs/ground_truth.json")
    args = parser.parse_args()
    main(data_dir=args.outdir, ref_out_dir=args.instructor_outdir)
