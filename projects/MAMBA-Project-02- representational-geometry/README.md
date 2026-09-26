# MAMBA Workshop 2026 — Project 03: Representational Geometry

End-to-end group exercise for the MAMBA Workshop 2026 final day (23-26
September 2026). See `metadata.yaml` for machine-readable metadata.

## Scenario

Given noisy population spike-count recordings from a set of neurons in
response to a structured, 2-D grid of stimuli, determine how many linear
dimensions the population's code actually occupies, test whether it
preserves the geometry of the true stimulus space, and build a
cross-validated linear decoder -- then validate everything against a set
of held-out stimuli never used in fitting. Full scenario and tasks:
`student/exercise_handout.md` (also available as `.pdf` and `.docx`).

## Intended audience / prerequisites

Workshop participants who have covered: vectors and matrices, covariance
and eigendecomposition, principal component analysis, distance/similarity
measures, ordinary least-squares regression, cross-validation, and basic
probability (the Poisson distribution). No material beyond the workshop's
own curriculum is required.

## Learning objectives

- Derive PCA from the eigendecomposition of a covariance matrix, and
  connect eigenvalues to variance explained.
- Distinguish a population's *linear* (PCA) dimensionality from a
  process's *true* (intrinsic) dimensionality, and explain why nonlinear
  tuning can make the former exceed the latter.
- Use pairwise-distance comparisons to test whether a population code
  preserves stimulus-space geometry, and correctly interpret a
  strong-but-imperfect relationship.
- Select a regression model's complexity via cross-validation and
  recognize genuine bias-variance trade-off behavior.
- Validate a complete analysis pipeline against genuinely held-out
  stimuli, not just held-out trials.

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

Building the DOCX/PDF handout from source additionally requires Pandoc and
a TeX distribution (with the `lmodern` and `libreoffice-math` system
packages available if re-rendering DOCX to PDF for review) -- not needed
to run any exercise code; pre-built `student/exercise_handout.pdf` and
`.docx` are already included.

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
                                 reference numbers, bootstrap summary, figures
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
