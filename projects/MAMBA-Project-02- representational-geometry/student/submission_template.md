# Project 03 Submission: Representational Geometry

**Group members:**
**Date:**

Fill in every section below. Where the handout (Section 11) left a method
choice open, say what you chose and why -- do not just report a number.

## 1. Derivation summary

(Attach or paste your full derivation from Task T1: why PCA eigenvalues
equal variance along their eigenvectors, the variance-explained formula,
and the least-squares decoder weight formula. Confirm every assumption is
stated.)

## 2. Method summary

For each of the following, one to three sentences on what you did and why:

- PCA computation method (covariance eigendecomposition vs. SVD):
- Representational-geometry statistic(s) used and why:
- Cross-validation scheme used to select $k$ (folds, whether the PCA
  basis was refit per fold):
- Anything else you did differently from a "default" choice:

## 3. Dimensionality summary

| Quantity | Value |
|---|---|
| True stimulus dimensionality | 2 |
| Components needed for 90% variance | |
| Components needed for 95% variance | |

## 4. Representational geometry

| Stimulus set | Pearson r | Spearman r |
|---|---|---|
| Fit only | | |
| Fit + holdout | | |

## 5. Required figures

(Insert, in order: variance-explained figure; PC1/PC2 projection figure;
neural-vs-stimulus distance figure; model-selection trade-off curve;
decoded-vs-true figure.)

## 6. Decoder and validation

| Quantity | Value |
|---|---|
| Chosen $k$ | |
| Mean decoding error, FIT stimuli | |
| Mean decoding error, HOLDOUT stimuli | |

## 7. Interpretation

(Task T6. Ground every claim in a number or figure from Sections 3-6
above. Address explicitly: what does your dimensionality finding mean
about how this population encodes information, what does your geometry
finding establish (and not establish), and how much do you trust the
decoder for a stimulus unlike anything in the fit set?)

## 8. Reflection

- What would you do differently with another day?
- Which finding are you least confident in, and why?

## Submission checklist (copy from handout Section 9)

- [ ] Derivation document
- [ ] Code runs end-to-end from the package root with a documented command
- [ ] Dimensionality summary
- [ ] All required figures
- [ ] Validation table covering FIT and HOLDOUT stimuli
- [ ] Written interpretation
- [ ] Statement of every open methodological choice and your reasoning
