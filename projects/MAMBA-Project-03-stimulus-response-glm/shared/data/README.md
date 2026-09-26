# Project 06 data: single-neuron stimulus-evoked responses

## What was recorded

A single 2-photon-imaged neuron in mouse visual cortex, recorded across
one continuous session of 600 trials. On each trial the animal viewed a
drifting grating at one of four orientations (0, 45, 90, 135 degrees)
and one of four contrasts (12.5, 25, 50, 100 percent). `response` is the
trial-evoked response amplitude (peak dF/F, arbitrary units) -- a single
continuous number per trial.

## Files

- `trials.csv` -- one row per trial, in the ORIGINAL SESSION ORDER (do
  not shuffle it): `trial_index`, `orientation_deg`, `contrast_pct`,
  `response`, `trial_role`.
- `data_dictionary.csv` -- column-by-column description.

## The `trial_role` column

The final 15% of trials (in session-time order) are marked `holdout`;
the rest are marked `fit`. Both are equally valid, equally noisy real
recordings. This is an analysis-plan constraint your group must respect:
estimate every parameter using `fit`-role trials only, and use
`holdout`-role trials only to validate predictions afterward -- never to
estimate anything.

## Was the stimulus schedule fully randomized?

`orientation_deg` was randomized in a block design (every 4 consecutive
trials contains one of each orientation, in random order). Whether
`contrast_pct` was drawn independently of session time, or whether its
distribution shifted across the session, is a question your own analysis
of `trials.csv` should be able to answer -- it is not asserted here
either way.

## Noise model (qualitative)

`response` reflects: the stimulus-driven effects of orientation and
contrast; ordinary trial-to-trial measurement noise; and POSSIBLY a
slow, stimulus-independent change in this neuron's baseline
responsiveness over the course of the session (e.g. from adaptation,
imaging-plane drift, or a slow change in behavioral state) -- something
several 2-photon datasets like this one are known to show. Whether it is
present in this particular recording, and if so, how strongly it affects
your conclusions, is exactly what this exercise asks you to determine.
No further detail is given here.
