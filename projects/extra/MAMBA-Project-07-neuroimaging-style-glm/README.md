# Project 07 — Find Task-Sensitive Channels on a Sensor Grid

A **one-day project** for MAMBA Workshop 2026 participants.
See `student/exercise_handout.docx` (or `.pdf`) for the full brief.

## Scenario

Given 200 trials of an 8x8 sensor grid (64 channels), find which channels respond to the task condition and visualise the result on the grid.

## Who this is for

Workshop participants who have covered: Basic Python, linear regression, the idea of a t-statistic. No material beyond the workshop's
own curriculum is required.

## What you will learn

- Loop over many regressions, one per channel.
- Compute a t-statistic from a regression.
- Reshape a flat list of values into a 2-D grid and plot as heatmap.

## Group size and estimated effort

**1 person (or a pair), about 4–6 hours** — one day.

## Data (in `shared/data/`)

- trials.csv — trial_index, block_index, condition, trial_role.
- channel_responses.csv — trial_index and 64 channel columns.
- channel_positions.csv — grid_row, grid_col for each channel.

## Steps

1. Load all three files and merge.
2. For each channel, fit a linear regression response ~ condition. Save the t-statistic.
3. Reshape the 64 t-stats into an 8x8 grid and plot a heatmap.
4. Mark channels with |t| > 2 as significant.

## What to hand in

- The filled notebook.
- 8x8 heatmap of t-statistics.
- Count of significant channels.
- 3-4 sentence paragraph on where the response sits on the grid.

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
