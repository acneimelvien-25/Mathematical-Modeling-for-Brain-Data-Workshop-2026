# Dataset: Membrane to Spike (Project 01)

Recordings from a single neuron under repeated current-clamp stimulation.
For each of 7 stimulus conditions, 10 trials of
200 ms were recorded at a sample interval of 0.05 ms.

Files:
- `voltage_traces.npz` -- noisy recorded membrane potential (see data_dictionary.csv)
- `spike_times.csv` -- spike times from an independent spike detector
- `stimulus_protocol.csv` -- the experiment design, including which conditions
  are recommended for **fitting** vs. reserved for **validation**
- `data_dictionary.csv` -- column-by-column description of every field

Noise model (disclosed; parameters are NOT): the injected current fluctuates
around its commanded DC value (a colored, mean-reverting fluctuation), and the
recorded voltage carries additional independent per-sample measurement noise.
Both are present in every trial and are part of the inference problem -- do not
assume noise-free access to the true membrane potential or the true injected
current at any instant.

No ground-truth model identity or parameter values are included in this
folder. That is intentional: your task is to infer a model from the
observations alone.
