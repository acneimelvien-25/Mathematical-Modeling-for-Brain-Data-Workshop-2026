# MAMBA Workshop 2026 — Project 06: Stimulus-to-Response GLM

End-to-end group exercise for the MAMBA Workshop 2026 final day (23-26
September 2026). See `metadata.yaml` for machine-readable metadata.

## Scenario

Build a general linear model of a single neuron's trial-evoked response
as a function of stimulus orientation and contrast; determine whether an
unmodeled, session-time-correlated confound (slow drift in baseline
excitability) is present and, if so, derive and confirm exactly how much
it distorts specific coefficient estimates and why; test several
scientifically interpretable contrasts under both a naive and a
corrected model; and validate out-of-sample. Full scenario and tasks:
`student/exercise_handout.md` (also available as `.pdf` and `.docx`).

## Intended audience / prerequisites

Workshop participants who have covered: vectors, matrices, matrix
multiplication; linear regression and the general linear model (design
matrices, least-squares estimation, residuals, contrasts, links to
t-tests and ANOVA); and basic probability. No material beyond the
workshop's own curriculum is required.

## Learning objectives

- Derive the classical omitted-variable-bias result for ordinary least
  squares from first principles, as an exact property of design
  matrices alone, and correctly distinguish it from single-dataset
  sampling noise.
- Use a design matrix's own correlation structure to predict, before
  fitting anything, which regressors are most at risk from a suspected
  confound.
- Diagnose an omitted, time-correlated confound using residual
  autocorrelation and a residuals-vs-time plot, and confirm it formally
  with a nested-model F-test.
- Construct, test, and correctly interpret contrasts (linear
  combinations of fitted GLM coefficients), including recognizing that a
  smaller standard error around a biased estimate is not a virtue.
- Validate a fitted model out-of-sample and connect predictive
  improvement to earlier diagnostic conclusions.

## Group size and estimated effort

4-6 participants, roughly 6-8 person-hours (about one workshop day). See
`student/exercise_handout.md` Section 6 for role suggestions, including a
4-person adaptation.

## Software requirements

Python >= 3.10 (developed and tested on 3.12). Dependencies in
`shared/requirements.txt`:

```bash
pip install -r shared/requirements.txt --break-system-packages
```

(Drop `--break-system-packages` if not needed in your environment.)

Building the DOCX/PDF handout from source additionally requires Pandoc, a
TeX distribution, and (for DOCX->PDF review rendering) the
`libreoffice-math` system package -- not needed to run any exercise code;
pre-built `student/exercise_handout.pdf` and `.docx` are already included.

## Exact run commands

From the package root:

```bash
# 1. Regenerate the synthetic dataset (optional -- data/ is already committed;
#    this reproduces it exactly from the documented seed, a few seconds)
python3 shared/src/generate_data.py

# 2. Run the test suite (unit tests + end-to-end smoke test)
cd shared && python3 -m pytest tests/ -v && cd ..

# 3. (Instructors) rebuild the reference analysis, figures, and answer-key
#    numbers
python3 instructor/build_reference.py

# 4. (Participants) open the starter notebook
jupyter notebook student/notebooks/00_starter.ipynb
```

## File map

```
README.md                    -- this file
metadata.yaml                -- machine-readable project metadata
THIRD_PARTY_NOTICES.md        -- third-party dependency licenses
MANIFEST.sha256               -- checksums of every file in this package
student/                      -- participant-facing materials only
  exercise_handout.{md,pdf,docx}
  latex/exercise_handout.tex  -- LaTeX source of the handout
  notebooks/00_starter.ipynb  -- starter notebook (no solutions)
  submission_template.md
shared/                       -- code and data used by both student and instructor
  data/                       -- committed synthetic dataset (see data/README.md)
  src/                        -- models.py, generate_data.py, analysis.py, plotting.py
  tests/                      -- pytest unit + smoke tests
  requirements.txt
  LICENSE
instructor/                   -- DO NOT DISTRIBUTE to participants
  instructor_guide.md
  answer_key.md
  grading_rubric.md
  checkpoint_questions.md
  troubleshooting.md
  build_reference.py          -- regenerates everything in reference_outputs/
  reference_outputs/          -- ground_truth.json (hidden parameters), fitted
                                 reference numbers, figures
```

## Expected outputs

Running the full pipeline (steps 1-3 above) reproduces, byte-for-byte or
numerically exactly (see `shared/tests/test_smoke.py`): the committed
`shared/data/` files, and every file in `instructor/reference_outputs/`.

## License map

- Code, prose, and synthetic data in this package: MIT (`shared/LICENSE`).
- Third-party build/run dependencies (not redistributed): see
  `THIRD_PARTY_NOTICES.md`.

## Reproducibility command

```bash
python3 shared/src/generate_data.py
cd shared && python3 -m pytest tests/test_smoke.py -v
```
