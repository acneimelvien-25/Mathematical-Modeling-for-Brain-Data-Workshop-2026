# Project 08 — How Coupling Strength Changes Network Dynamics

A **one-day project** for MAMBA Workshop 2026 participants.
See `student/exercise_handout.docx` (or `.pdf`) for the full brief.

## Scenario

Given the activity of 5 coupled nodes at several coupling strengths w, measure how the amplitude of activity grows as w approaches a critical value.

## Who this is for

Workshop participants who have covered: Basic Python, standard deviation, the idea of a network. No material beyond the workshop's
own curriculum is required.

## What you will learn

- Read a wide-format CSV (one column per node).
- Compute standard deviation as a measure of amplitude.
- Understand how a linear network approaches instability.

## Group size and estimated effort

**1 person (or a pair), about 4–6 hours** — one day.

## Data (in `shared/data/`)

- fit_timeseries.csv — node activity at several subthreshold w.
- holdout_timeseries.csv — node activity at near-threshold w.

## Steps

1. Load fit_timeseries.csv, print the unique w values.
2. Plot the 5 nodes' activity for one w value.
3. For each w, compute mean amplitude (std) across the 5 nodes.
4. Plot amplitude vs w for both fit and holdout data.

## What to hand in

- The filled notebook.
- Time-series plot from Step 2.
- Amplitude vs w plot (both fit and holdout).
- 3-4 sentence paragraph on how the network responds to stronger coupling.

## How to run

```bash
# 1. Install dependencies (once)
pip install -r shared/requirements.txt

# 2. Open the starter notebook
jupyter notebook student/notebooks/00_starter.ipynb
```

## Files

```
README.md                        -- this file
student/
  exercise_handout.docx / .pdf   -- the participant brief
  submission_template.md         -- what to submit
  notebooks/
    00_starter.ipynb             -- start here
shared/
  data/                          -- the dataset
  src/                           -- helper code
  requirements.txt
```
