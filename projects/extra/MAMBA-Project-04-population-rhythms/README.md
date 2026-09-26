# Project 04 — Population Rhythms in E-I Networks

A **one-day project** for MAMBA Workshop 2026 participants.
See `student/exercise_handout.docx` (or `.pdf`) for the full brief.

## Scenario

Test whether an excitatory-inhibitory (E-I) neural population oscillates. Find the oscillation frequency using a power spectrum, and see how the frequency depends on the external drive.

## Who this is for

Workshop participants who have covered: Basic Python, some familiarity with signals (a signal has a frequency), basic probability. No material beyond the workshop's
own curriculum is required.

## What you will learn

- Read .npz files with numpy arrays.
- Compute a power spectrum with scipy.signal.welch.
- Find the peak of a spectrum with numpy.argmax.

## Group size and estimated effort

**1 person (or a pair), about 4–6 hours** — one day.

## Data (in `shared/data/`)

- population_traces.npz — E and I population activity, 48 trials.
- stimulus_protocol.csv — external drive I_I for each condition.

## Steps

1. Load and look — plot one E and one I trace.
2. Compute the power spectrum of the E signal.
3. Find the peak frequency for every trial.
4. Plot mean peak frequency vs drive I_I.

## What to hand in

- The filled notebook.
- Example E and I trace plot.
- One power-spectrum plot.
- Peak frequency vs I_I plot.
- 3-4 sentence paragraph on how oscillation depends on drive.

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
