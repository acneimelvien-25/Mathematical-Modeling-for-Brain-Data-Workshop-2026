# Dataset: Noisy Neural Dynamics (Project 04)

Recordings of a single passive (non-spiking) membrane compartment across
6 sessions (4 fit, 2 holdout), each
5 s long. Sessions differ in their TARGET sampling
density (see `session_protocol.csv`'s `mean_dt_ms`) and in their own mean
voltage level (reflecting session-to-session differences in ambient
synaptic drive) -- but the compartment's own intrinsic dynamics (its
membrane time constant and the amplitude of its input-driven fluctuations)
are the SAME physical property in every session, and recovering them
precisely requires appropriately combining information across sessions.

**Samples within a session are NOT evenly spaced in time.** Consecutive
gaps vary around each session's target mean interval. Do not assume a
fixed sample clock anywhere in your analysis -- use the actual recorded
`time_ms` values.

Files:
- `voltage_sessions.npz` -- irregular (time, voltage) samples, all sessions
- `session_protocol.csv` -- the experiment design, including which sessions
  are recommended for **fitting** vs. reserved for **validation**
- `data_dictionary.csv` -- column-by-column description of every field

Noise model (disclosed; parameters are NOT): the true voltage undergoes
its own intrinsic random fluctuations (present in the underlying dynamics,
continuously, not just at the sampled instants), and the recorded signal
carries additional independent observation noise on top of that at each
sampled instant. No ground-truth parameter values, or the identity of the
underlying process, are included in this folder. That is intentional: your
task is to infer them from the observations alone.
