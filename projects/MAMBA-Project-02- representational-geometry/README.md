# Project 02 — See the Shape of a Neural Code

A **one-day project** for MAMBA Workshop 2026 participants.
See `student/exercise_handout.docx` (or `.pdf`) for the full brief.

## Scenario

Given the responses of 80 neurons to a 2-D grid of visual stimuli, answer: how many dimensions does the neural code really use? Can we decode the stimulus back?

## Who this is for

Workshop participants who have covered: Basic Python, vectors and matrices, some idea of variance and dimensionality. No material beyond the workshop's
own curriculum is required.

## What you will learn

- Run PCA with sklearn.decomposition.PCA.
- Train a linear decoder with sklearn.linear_model.Ridge.
- Split data into train and test with train_test_split.

## Group size and estimated effort

**1 person (or a pair), about 4–6 hours** — one day.

## Data (in `shared/data/`)

- population_responses.npz — 640 trials × 80 neurons of spike counts.
- stimulus_grid.csv — the 2-D stimulus (feature_x, feature_y) on each trial.

## Steps

1. Load data, keep the fit trials for training and holdout for a final check.
2. Run PCA on the responses. How many components explain 80% variance?
3. Project data onto first 2 PCs and colour by stimulus feature.
4. Train a Ridge decoder to predict the stimulus; report test R².

## What to hand in

- The filled notebook.
- Variance-explained bar plot.
- 2-D PCA scatter plot coloured by stimulus.
- Decoder R² on the test set.
- 3-4 sentence paragraph: is the code low-dimensional? Does the decoder work?

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
