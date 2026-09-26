# Project 06 Submission: Stimulus-to-Response GLM

**Group members:**
**Date:**

Fill in every section below. Where the handout (Section 11) left a method
choice open, say what you chose and why -- do not just report a number.

## 1. Derivation summary

(Attach or paste your full derivation from Task T1: the omitted-variable-
bias formula, your Section 3.2 hand-check with numbers filled in, and
both verification results -- the repeated-noise-draw average bias
compared to the formula, and the exactly-zero-bias orthogonal check.)

## 2. Method summary

For each of the following, one to three sentences on what you did and why:

- Basis chosen for the drift term $X_2$, and what diagnostic plot justified it:
- What you substituted for the true $\beta_2$ when evaluating the bias
  formula on real data (Task T3), and why that is reasonable here:
- Third contrast chosen for Task T5, and why it is scientifically interesting:
- Resampling method and number of resamples used for uncertainty quantification:
- Lags used for residual autocorrelation, and why:

## 3. Confound check (Task T2)

| Regressor | Correlation with trial index | Predicted bias risk |
|---|---|---|
| D45 | | |
| D90 | | |
| D135 | | |
| log_contrast | | |

## 4. Coefficient table (Task T3)

| Coefficient | True (n/a -- not given) | Naive estimate (SE) | Full estimate (SE) | Predicted bias (formula) | Observed bias (naive) |
|---|---|---|---|---|---|
| intercept | | | | | |
| D45 | | | | | |
| D90 | | | | | |
| D135 | | | | | |
| log_contrast | | | | | |

## 5. Residual diagnostics and nested F-test (Task T4)

| Lag | Naive model autocorrelation | Full model autocorrelation |
|---|---|---|
| | | |

**F-test result:** F( , ) = , p =

**Conclusion (is there evidence of a real session-time confound?):**

## 6. Contrasts (Task T5)

| Contrast | Naive estimate (SE, t, p) | Full estimate (SE, t, p) | Uncertainty range (resampling) |
|---|---|---|---|
| (e.g. 90 vs. 0 deg) | | | |
| (contrast slope) | | | |
| (your third contrast) | | | |

**Any case where the naive vs. full conclusion differs, or where SE alone
would have been misleading:**

## 7. Holdout validation (Task T6)

| Model | RMSE | R^2 |
|---|---|---|
| Naive | | |
| Full | | |

## 8. Required figures

(Insert, in order: residuals-vs-trial-index and autocorrelation-vs-lag
for both models; holdout prediction figure.)

## 9. Interpretation (Task T7)

(Ground every claim in a number or figure from Sections 3-8 above.
Address explicitly: this neuron's orientation and contrast tuning;
whether a session-time confound is present and how it would have
distorted the collaborator's planned analysis if ignored; and a concrete
recommendation for their next recording session's design.)

## 10. Reflection

- What would you do differently with another day?
- Which estimate are you least confident in, and why?

## Submission checklist (copy from handout Section 9)

- [ ] Derivation with hand-check and both verification checks
- [ ] Code runs end-to-end from the package root with a documented command
- [ ] Coefficient table with bias comparison
- [ ] All required figures
- [ ] Nested F-test result and stated conclusion
- [ ] Contrast table with uncertainty range
- [ ] Written interpretation
- [ ] Statement of every open methodological choice and your reasoning
