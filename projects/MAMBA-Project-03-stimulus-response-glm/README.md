# Project 06 — Fit a Linear Model to Neural Responses

A **one-day project** for MAMBA Workshop 2026 participants.
See `student/exercise_handout.docx` (or `.pdf`) for the full brief.

## Scenario

Given a table of trials for one neuron (orientation, contrast, response), fit a linear model predicting the response and check how well it works.

## Who this is for

Workshop participants who have covered: Basic Python, linear regression (y = a + b x), the idea of R². No material beyond the workshop's
own curriculum is required.

## What you will learn

- Build a simple design matrix and fit a linear model.
- Interpret regression coefficients.
- Split data into train and test to check generalisation.

## Group size and estimated effort

**1 person (or a pair), about 4–6 hours** — one day.

## Data (in `shared/data/`)

- trials.csv — trial_index, orientation, contrast, response, trial_role.
- data_dictionary.csv — column descriptions.

## Steps

1. Load the trials table and inspect it.
2. Fit LinearRegression on the fit trials with orientation and contrast as features.
3. Report the coefficients and check the fit on all data (R²).
4. Do an 80/20 train/test split and compare training vs test R².

## What to hand in

- The filled notebook.
- Predicted-vs-observed scatter plot.
- Fitted coefficients (β0, β1, β2) and both R² values.
- 3-4 sentence paragraph: which feature drives the response? Does it generalise?

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
