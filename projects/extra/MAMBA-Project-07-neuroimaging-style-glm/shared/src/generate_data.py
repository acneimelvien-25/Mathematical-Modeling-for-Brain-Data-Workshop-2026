"""
generate_data.py
=================
Deterministic synthetic data generator for MAMBA Project 07:
Neuroimaging-Style GLM.

Running this script regenerates, byte-for-byte given the fixed seed
below:
  - shared/data/trials.csv           (block_index, trial_index, condition)
  - shared/data/channel_responses.csv (one row per trial, one column per channel)
  - shared/data/channel_positions.csv (channel_id, grid_row, grid_col)
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
    N_CHANNELS, GRID_SHAPE, BlockDesignParams, NoiseParams,
    simulate_multichannel_session,
)

MASTER_SEED = 20260929

DESIGN = BlockDesignParams(n_blocks=40, block_size=5)  # 200 trials
NOISE = NoiseParams(phi=0.6, sigma=1.0)
N_SIGNAL_CHANNELS = 6
TRUE_EFFECT = 1.5

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "data"))
REF_OUT_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "instructor", "reference_outputs"))


def main(data_dir: str = DATA_DIR, ref_out_dir: str = REF_OUT_DIR):
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(ref_out_dir, exist_ok=True)

    rng = np.random.default_rng(MASTER_SEED)
    signal_channels = np.sort(rng.choice(N_CHANNELS, size=N_SIGNAL_CHANNELS, replace=False))
    session = simulate_multichannel_session(DESIGN, NOISE, signal_channels, TRUE_EFFECT, rng)

    block_labels = session["block_labels"]
    condition = session["condition"]
    Y = session["Y"]

    # discovery/replication split: EVEN-indexed blocks = discovery,
    # ODD-indexed blocks = replication (interleaved split-half, not a
    # first-half/second-half split -- avoids conflating with any
    # session-time trend, of which there is deliberately none in this model)
    block_idx = np.arange(DESIGN.n_blocks)
    block_role = np.where(block_idx % 2 == 0, "discovery", "replication")
    trial_block_id = np.repeat(np.arange(DESIGN.n_blocks), DESIGN.block_size)
    trial_role = block_role[trial_block_id]

    # ---- shared/data/trials.csv ----
    trials_path = os.path.join(data_dir, "trials.csv")
    with open(trials_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial_index", "block_index", "condition", "trial_role"])
        for t in range(DESIGN.n_trials):
            w.writerow([t, int(trial_block_id[t]), int(condition[t]), trial_role[t]])

    # ---- shared/data/channel_responses.csv ----
    resp_path = os.path.join(data_dir, "channel_responses.csv")
    with open(resp_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial_index"] + [f"ch{c:02d}" for c in range(N_CHANNELS)])
        for t in range(DESIGN.n_trials):
            w.writerow([t] + [round(float(v), 5) for v in Y[t, :]])

    # ---- shared/data/channel_positions.csv ----
    pos_path = os.path.join(data_dir, "channel_positions.csv")
    with open(pos_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["channel_id", "grid_row", "grid_col"])
        n_rows, n_cols = GRID_SHAPE
        for c in range(N_CHANNELS):
            w.writerow([c, c // n_cols, c % n_cols])

    # ---- data_dictionary.csv ----
    dict_path = os.path.join(data_dir, "data_dictionary.csv")
    with open(dict_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "column", "type", "units", "description"])
        rows = [
            ("trials.csv", "trial_index", "int", "-", "0-indexed trial order (joins to channel_responses.csv)"),
            ("trials.csv", "block_index", "int", "-", "Which consecutive block of trials this trial belongs to (0-39)"),
            ("trials.csv", "condition", "int (0/1)", "-", "Task condition for this trial's block (constant within a block)"),
            ("trials.csv", "trial_role", "str", "-", "'discovery' or 'replication' -- see shared/data/README.md"),
            ("channel_responses.csv", "trial_index", "int", "-", "Joins to trials.csv"),
            ("channel_responses.csv", "ch00 .. ch63", "float", "arbitrary units", "This channel's response on this trial"),
            ("channel_positions.csv", "channel_id", "int", "-", "0-63, matches the chNN column suffix in channel_responses.csv"),
            ("channel_positions.csv", "grid_row", "int", "-", "Row position (0-7) in the 8x8 sensor array, for visualization only"),
            ("channel_positions.csv", "grid_col", "int", "-", "Column position (0-7) in the 8x8 sensor array, for visualization only"),
        ]
        for row in rows:
            w.writerow(row)

    # ---- README.md ----
    readme_path = os.path.join(data_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(DATA_README_TEMPLATE)

    # ---- ground_truth.json (instructor only) ----
    gt = {
        "master_seed": MASTER_SEED,
        "n_channels": N_CHANNELS,
        "grid_shape": list(GRID_SHAPE),
        "n_blocks": DESIGN.n_blocks,
        "block_size": DESIGN.block_size,
        "n_trials": DESIGN.n_trials,
        "phi": NOISE.phi,
        "sigma": NOISE.sigma,
        "n_signal_channels": N_SIGNAL_CHANNELS,
        "true_effect": TRUE_EFFECT,
        "signal_channels": signal_channels.tolist(),
    }
    with open(os.path.join(ref_out_dir, "ground_truth.json"), "w") as f:
        json.dump(gt, f, indent=2)

    print(f"Generated {DESIGN.n_trials} trials x {N_CHANNELS} channels "
          f"({DESIGN.n_blocks} blocks of {DESIGN.block_size}).")
    print(f"Wrote: {trials_path}\n       {resp_path}\n       {pos_path}\n       {dict_path}\n       {readme_path}")
    print(f"Wrote (instructor-only): {os.path.join(ref_out_dir, 'ground_truth.json')}")


DATA_README_TEMPLATE = """\
# Project 07 data: multi-channel task recording

## What was recorded

A simulated 64-channel recording array (think a small ECoG grid, EEG
montage, or fMRI voxel patch -- arranged as an 8x8 grid, see
`channel_positions.csv`), recorded during a two-condition task. Trials
are grouped into 40 consecutive BLOCKS of 5 trials each; every trial in
a block shares the same condition (a block design). Every channel is
observed on the SAME 200 trials -- there is one shared design (one
`condition` sequence) for the whole array.

## Files

- `trials.csv` -- one row per trial: `trial_index`, `block_index`,
  `condition` (0/1), `trial_role`.
- `channel_responses.csv` -- one row per trial, one column per channel
  (`ch00` .. `ch63`): this trial's response on that channel.
- `channel_positions.csv` -- each channel's position in the 8x8 array,
  for visualization only (spatial position plays no other role in this
  exercise).
- `data_dictionary.csv` -- column-by-column description of the files above.

## The `trial_role` column

Blocks are split by PARITY of block index (even-indexed blocks =
`discovery`, odd-indexed blocks = `replication`) -- an interleaved
split-half, not a first-half/second-half split. Use `discovery`-role
trials for your primary analysis (Tasks T1-T5); use `replication`-role
trials ONLY to check how many of your discovery-stage conclusions hold
up independently (Task T6) -- never to select or threshold channels in
the first place.

## Noise model (qualitative)

Every channel's trial-to-trial response reflects: a possible
condition-related effect (present in some channels, absent in others --
which is which is exactly what this exercise asks you to determine);
and channel noise that may or may not be simply independent from trial
to trial -- something your own residual/diagnostic analysis should be
able to establish, not something asserted here. No further detail about
the noise process, and no information about which or how many channels
carry a real effect, is given in this file.
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=DATA_DIR, help="Output directory for shared/data files")
    parser.add_argument("--instructor-outdir", default=REF_OUT_DIR,
                         help="Output directory for instructor/reference_outputs/ground_truth.json")
    args = parser.parse_args()
    main(data_dir=args.outdir, ref_out_dir=args.instructor_outdir)
