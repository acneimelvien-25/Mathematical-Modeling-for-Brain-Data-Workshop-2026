# Project 01 — Fit a Leaky Neuron

A **one-day project** for MAMBA Workshop 2026 participants.
See `student/exercise_handout.docx` (or `.pdf`) for the full brief.

## Scenario

Fit a Leaky Integrate-and-Fire (LIF) model to voltage recordings of one neuron. Find its two main parameters (τ and R) and check whether the model predicts the firing rate at higher currents.

## Who this is for

Workshop participants who have covered: Basic Python, first-year calculus (exponential functions), a first course in probability. No material beyond the workshop's
own curriculum is required.

## What you will learn

- Load .npz and .csv data files with numpy and pandas.
- Fit a simple exponential model with scipy.optimize.curve_fit.
- Understand how the LIF model links subthreshold dynamics to firing rate.

## Group size and estimated effort

**1 person (or a pair), about 4–6 hours** — one day.

## Data (in `shared/data/`)

- voltage_traces.npz — voltage over time at several input currents.
- spike_times.csv — when the neuron spiked in each trial.
- stimulus_protocol.csv — which current was injected in each trial.

## Steps

1. Load and look — plot one voltage trace per condition.
2. Fit τ and R from subthreshold data using curve_fit.
3. Predict firing rate from the fitted LIF parameters.
4. Compare predicted vs observed firing rates in one plot.

## What to hand in

- The filled notebook.
- One plot: predicted vs observed firing rate across currents.
- Fitted values of τ and R.
- 3-4 sentence paragraph explaining where the model works and where it fails.

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
