# MAMBA Workshop 2026 — Project 01: Membrane to Spike

End-to-end group exercise for the MAMBA Workshop 2026 final day (23-26
September 2026). See `metadata.yaml` for machine-readable metadata.

## Scenario

Infer a leaky integrate-and-fire (LIF) model from noisy current-clamp
recordings of a neuron, and find out how well it predicts firing behavior
at current levels it was never fit against. Full scenario and tasks: 
`student/exercise_handout.md` (also available as `.pdf` and `.docx`).

## Intended audience / prerequisites

Workshop participants who have covered: single-neuron models (LIF,
Hodgkin-Huxley-style dynamics), first-order ODEs and stability, linear
regression / the general linear model, and basic probability (random
variables, noise, expectation/variance). No material beyond the workshop's
own curriculum is required.

## Learning objectives

- Derive a leaky integrate-and-fire model from a circuit-level assumption
  and its closed-form subthreshold and firing-rate solutions.
- Estimate all parameters of a nonlinear dynamical model from noisy,
  multi-condition observational data, making and justifying concrete
  numerical/statistical method choices.
- Quantify parameter uncertainty via resampling and distinguish sampling
  uncertainty from structural non-identifiability.
- Validate a fitted model against held-out conditions and give a
  biologically grounded account of where and why it fails.

## Group size and estimated effort

4-6 participants, roughly 6-8 person-hours (about one workshop day). 

## Software requirements

Python >= 3.10 (developed and tested on 3.12). Dependencies in
`shared/requirements.txt`:

```
pip install -r shared/requirements.txt
```

## Exact run commands

From the package root:

```bash
# 1. Regenerate the synthetic dataset (optional -- data/ is already committed;
#    this reproduces it exactly from the documented seed)
python3 shared/src/generate_data.py

# 2. Run the test suite (unit tests + end-to-end smoke test, ~90s total,
#    mostly spent re-simulating the dataset for the determinism check)
cd shared && python3 -m pytest tests/ -v && cd ..

# 3. (Participants) open the starter notebook
jupyter notebook student/notebooks/00_starter.ipynb
```

## File map

```
README.md                    -- this file
metadata.yaml                -- machine-readable project metadata
student/                      
  exercise_handout.{pdf,docx}
  notebooks/00_starter.ipynb  
  submission_template.md
shared/                       -- code and data used by both student and instructor
  data/                       -- synthetic dataset (see data/README.md)
  src/                        -- models.py, generate_data.py, analysis.py, plotting.py
  tests/                      -- pytest unit + smoke tests
  requirements.txt
  LICENSE
```

## Reproducibility command

```bash
python3 shared/src/generate_data.py && \
  python3 -c "import json,sys; sys.path.insert(0,'shared/src'); \
  print('OK - rerun shared/tests/test_smoke.py for a full check')"
cd shared && python3 -m pytest tests/test_smoke.py -v
```
