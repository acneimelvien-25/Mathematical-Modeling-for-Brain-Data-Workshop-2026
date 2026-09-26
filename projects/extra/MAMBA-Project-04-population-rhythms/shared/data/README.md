# Dataset: Population Rhythms (Project 02)

Recordings of two-population (excitatory E, inhibitory I) activity from a
single "preparation," under 6 levels of a control variable
(external drive to the inhibitory population, `I_I`) -- think of this as a
pharmacological or optogenetic manipulation of inhibitory tone. For each
condition, 8 trials of 10 s were
recorded at a sample interval of 0.5 ms, after a
500 ms burn-in (not included in the recorded data) to let the
population settle onto its natural fluctuation regime.

Files:
- `population_traces.npz` -- noisy recorded (E, I) activity (see data_dictionary.csv)
- `stimulus_protocol.csv` -- the experiment design, including which conditions
  are recommended for **fitting** vs. reserved for **validation**
- `data_dictionary.csv` -- column-by-column description of every field
- `noise_calibration.json` -- the recording system's KNOWN measurement-noise
  standard deviation (from an independent instrument calibration -- see
  below for why this one noise parameter is disclosed when nothing else is)

Noise model: the population activity itself evolves with intrinsic
stochastic fluctuations (present in the underlying dynamics, not just the
recording) -- this part of the noise model is NOT disclosed. On top of
that, the recorded signal carries additional independent observation
(measurement) noise, and THIS noise's standard deviation is disclosed in
`noise_calibration.json`, as it would be from a real instrument's
calibration record. Think about why you might need to know this value
specifically, given what kind of quantity you are trying to estimate from
short, closely-spaced samples of a noisy signal.

No ground-truth model parameters, fixed points, or stability properties are
included in this folder. That is intentional: your task is to infer them
from the observations alone.
