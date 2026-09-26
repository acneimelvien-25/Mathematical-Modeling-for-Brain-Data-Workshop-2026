# Project 05 — Estimate the Mean Voltage from Noisy Recordings

A **one-day project** for MAMBA Workshop 2026 participants.
See `student/exercise_handout.docx` (or `.pdf`) for the full brief.

## Scenario

You have voltage recordings from 6 sessions with irregular sample times. Estimate the mean voltage V_ss for each session and check how the estimate depends on how densely each session was sampled.

## Who this is for

Workshop participants who have covered: Basic Python, mean and standard deviation, the standard error of the mean. No material beyond the workshop's
own curriculum is required.

## What you will learn

- Load npz arrays where samples from multiple sessions are stored together.
- Compute mean, standard deviation, and standard error of the mean.
- Understand how the standard error scales with sample size.

## Group size and estimated effort

**1 person (or a pair), about 4–6 hours** — one day.

## Data (in `shared/data/`)

- voltage_sessions.npz — session_id, time_ms, V_obs_mV for every sample.
- session_protocol.csv — sample interval, count, and role for each session.

## Steps

1. Load and inspect the data; print how many samples per session.
2. Plot voltage vs time for one session (notice irregular spacing).
3. For each session, compute the mean, standard deviation, and standard error.
4. Plot standard error vs 1/√n — should be roughly linear.

## What to hand in

- The filled notebook.
- The plot of one session (V vs time).
- Table of V_ss estimates and their standard errors.
- Standard error vs 1/√n plot.
- 3-4 sentence paragraph on how the estimate depends on sample count.

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
