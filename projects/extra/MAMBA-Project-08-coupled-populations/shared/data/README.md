# Project 08 data: coupled-population ring activity

## What was recorded

Five simulated neural populations ("nodes"), arranged so that each node
receives input from exactly one other node -- a directed ring. Each
node's activity was recorded at several values of the coupling strength
`w`, split into two files:

- `fit_timeseries.csv`: node activity recorded at SUBTHRESHOLD coupling
  strengths (the network sits near a stable equilibrium, with visible
  noisy fluctuations around it).
- `holdout_timeseries.csv`: node activity recorded at coupling strengths
  at or above where the network is expected to become unstable, with
  no added recording noise (any structure you see reflects the
  network's own dynamics, not measurement noise).

Use `fit_timeseries.csv` ONLY for estimation (Tasks T1-T3); use
`holdout_timeseries.csv` ONLY to validate your predictions afterward
(Task T4 onward) -- never to fit or tune anything.

## Files

- `fit_timeseries.csv` -- columns: `w`, `time`, `x0`, `x1`, `x2`, `x3`, `x4`.
- `holdout_timeseries.csv` -- same columns, different `w` values.
- `data_dictionary.csv` -- column-by-column description.

## What is NOT given

The exact functional form of each node's dynamics, the sign or
topology of the coupling, and the gain of the nonlinearity relating one
node's activity to its influence on the next are all things your group
is expected to investigate and/or estimate from the data -- none of
these are stated directly here. (The handout gives you a starting
functional FORM to test against the data; confirming it fits, and
estimating its free parameter(s), is Task T1-T3's job.)
