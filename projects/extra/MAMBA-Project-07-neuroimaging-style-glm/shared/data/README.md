# Project 07 data: multi-channel task recording

## What was recorded

A simulated 64-channel recording array (think a small ECoG grid, EEG
montage, or fMRI voxel patch -- arranged as an 8x8 grid, see
`channel_positions.csv`), recorded during a two-condition task. Trials
are grouped into 40 consecutive BLOCKS of 5 trials each; every trial in
a block shares the same condition (a block design). Every channel is
observed on the SAME 200 trials -- there is one shared design (one
`condition` sequence) for the whole array.

## Files

- `trials.csv` -- one row per trial: `trial_index`, `block_index`,
  `condition` (0/1), `trial_role`.
- `channel_responses.csv` -- one row per trial, one column per channel
  (`ch00` .. `ch63`): this trial's response on that channel.
- `channel_positions.csv` -- each channel's position in the 8x8 array,
  for visualization only (spatial position plays no other role in this
  exercise).
- `data_dictionary.csv` -- column-by-column description of the files above.

## The `trial_role` column

Blocks are split by PARITY of block index (even-indexed blocks =
`discovery`, odd-indexed blocks = `replication`) -- an interleaved
split-half, not a first-half/second-half split. Use `discovery`-role
trials for your primary analysis (Tasks T1-T5); use `replication`-role
trials ONLY to check how many of your discovery-stage conclusions hold
up independently (Task T6) -- never to select or threshold channels in
the first place.

## Noise model (qualitative)

Every channel's trial-to-trial response reflects: a possible
condition-related effect (present in some channels, absent in others --
which is which is exactly what this exercise asks you to determine);
and channel noise that may or may not be simply independent from trial
to trial -- something your own residual/diagnostic analysis should be
able to establish, not something asserted here. No further detail about
the noise process, and no information about which or how many channels
carry a real effect, is given in this file.
