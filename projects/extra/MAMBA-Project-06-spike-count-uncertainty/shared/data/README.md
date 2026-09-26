# Project 05 data: spike counts across contrast conditions

## What was recorded

A single extracellularly recorded neuron (visual cortex, awake animal),
tested with a drifting-grating stimulus at five contrast levels:
6%, 12%, 25%, 50%, and 100% Michelson contrast. Each contrast was
presented for 250 trials, in randomly interleaved order during the
recording session (trial order does not matter for this exercise and is
not preserved). Every trial is 800 ms long, timed from stimulus onset
(t = 0).

## Files

- `spike_times.csv` -- one row per spike: `trial_id`, `condition_pct`,
  `spike_time_ms` (time from stimulus onset). Trials with zero spikes
  contribute no rows -- do not assume every `trial_id` appears here.
- `trial_protocol.csv` -- one row per trial (1250 rows total: 5 conditions
  x 250 trials): `trial_id`, `condition_pct`, `condition_role`,
  `trial_duration_ms`.
- `data_dictionary.csv` -- column-by-column description of the two files
  above.

## The `condition_role` column

Each contrast condition is labeled `fit` or `holdout` in
`trial_protocol.csv`. This is not a data-quality label -- both roles
contain equally valid, equally noisy real recordings. It is an
analysis-plan constraint your group must respect:

- Any parameter that is shared across conditions (i.e. assumed to be a
  property of the recording session as a whole, not of any one stimulus)
  may be ESTIMATED using `fit`-role trials only.
- `holdout`-role trials must be reserved purely for VALIDATING
  predictions made from parameters estimated on `fit` trials -- never
  used to estimate those parameters in the first place.

The exercise handout also asks you to reserve a portion of the
*counting-window range* the same way (see handout Section 2) -- that
split is a matter of how you choose to analyze this same spike-time
data, not a property of the data files themselves, so it is not encoded
as a column here.

## Noise model (qualitative)

Spike generation on each trial reflects two sources of variability:

1. Ordinary point-process ("shot") noise: even at a perfectly constant
   underlying rate, the exact number and timing of spikes in a fixed
   window varies from trial to trial.
2. A trial-to-trial fluctuation in the neuron's overall excitability
   (sometimes attributed to arousal, attention, or other slow network
   state changes), which very likely differs from trial to trial. Its
   statistical size and character are exactly what this exercise asks
   you to characterize -- no value for it is given here.

No further detail about the noise model is provided in student-facing
files. Deriving what its presence or absence would predict about how
spike-count variability should depend on the counting-window length is
the mathematical heart of this exercise (see the handout).
