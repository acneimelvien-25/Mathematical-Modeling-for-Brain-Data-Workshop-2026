# Project 06 — Is Spike Counting Poisson?

A **one-day project** for MAMBA Workshop 2026 participants.
See `student/exercise_handout.docx` (or `.pdf`) for the full brief.

## Scenario

Given 1250 trials of spike times at 5 contrast levels, test whether spike counts follow a Poisson distribution or are overdispersed.

## Who this is for

Workshop participants who have covered: Basic Python, mean and variance, the Poisson distribution. No material beyond the workshop's
own curriculum is required.

## What you will learn

- Group data by trial using pandas groupby.
- Compute mean and variance of a count distribution.
- Interpret the Fano factor (variance ÷ mean).

## Group size and estimated effort

**1 person (or a pair), about 4–6 hours** — one day.

## Data (in `shared/data/`)

- trial_protocol.csv — trial_id, contrast, role.
- spike_times.csv — one row per spike: trial_id, contrast, time.

## Steps

1. Load both files, print trials per contrast.
2. Count spikes per trial (trials with 0 spikes count as 0).
3. For each contrast, compute mean, variance, and Fano factor.
4. Plot Fano factor vs contrast; mark the Poisson baseline (F = 1).

## What to hand in

- The filled notebook.
- Table of contrast, mean count, variance, Fano factor.
- Fano vs contrast plot.
- 3-4 sentence paragraph on whether the counts are Poisson.

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
