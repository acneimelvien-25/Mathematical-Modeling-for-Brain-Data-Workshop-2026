"""
analysis.py
===========
Reference analysis pipeline for MAMBA Project 06: Stimulus-to-Response GLM.

Pipeline stages (see student handout Sections 2 and 5 for the full task
description):

1. Build the design matrix for a "naive" model (orientation + contrast
   only) and a "full" model (naive + a drift basis), fit both by
   ordinary least squares on FIT trials only.
2. Compare fitted coefficients between the two models; compute the
   EXACT closed-form omitted-variable-bias prediction from the design
   matrices alone (no fitting/noise involved) and compare it to what is
   actually observed.
3. Residual diagnostics: autocorrelation of residuals at several lags,
   for both models.
4. A nested-model F-test: does adding the drift basis significantly
   improve the fit?
5. Contrasts: point estimate, standard error, t-statistic, and p-value
   for an arbitrary linear combination of fitted coefficients, under
   either model.
6. Holdout validation: out-of-sample RMSE and R^2 on holdout trials.

Every public function takes plain arrays in and returns plain arrays/
dicts/floats out, so it can be unit-tested against data with KNOWN
parameters independently of the committed dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Stage 1: OLS fitting
# ---------------------------------------------------------------------------

@dataclass
class OLSFit:
    beta: np.ndarray
    residuals: np.ndarray
    se: np.ndarray
    cov_beta: np.ndarray
    sigma2_hat: float
    dof: int
    rss: float
    n: int
    p: int


def fit_ols(X: np.ndarray, y: np.ndarray) -> OLSFit:
    """Ordinary least squares fit with the standard homoskedastic-Gaussian
    covariance estimate: cov(beta_hat) = sigma2_hat * (X'X)^-1, where
    sigma2_hat = RSS / (n - p).
    """
    n, p = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta
    dof = n - p
    rss = float(np.sum(residuals ** 2))
    sigma2_hat = rss / dof
    cov_beta = sigma2_hat * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov_beta))
    return OLSFit(beta=beta, residuals=residuals, se=se, cov_beta=cov_beta,
                  sigma2_hat=sigma2_hat, dof=dof, rss=rss, n=n, p=p)


# ---------------------------------------------------------------------------
# Stage 2: omitted-variable-bias formula
# ---------------------------------------------------------------------------

def omitted_variable_bias(X_included: np.ndarray, X_omitted: np.ndarray,
                           beta_omitted_true: np.ndarray) -> np.ndarray:
    """Exact formula for E[beta_hat_included] - beta_included_true, when
    the TRUE model is y = X_included @ beta_included + X_omitted @
    beta_omitted + noise, but only X_included is actually fit:

        bias = (X_included' X_included)^-1  X_included' X_omitted  beta_omitted_true

    This is a property of the two design matrices and the true omitted
    coefficients ONLY -- it does not depend on the noise realization at
    all, and can be computed before any model is ever fit to real data.
    """
    XtX_inv = np.linalg.inv(X_included.T @ X_included)
    return XtX_inv @ X_included.T @ X_omitted @ beta_omitted_true


# ---------------------------------------------------------------------------
# Stage 3: residual autocorrelation
# ---------------------------------------------------------------------------

def residual_autocorrelation(residuals: np.ndarray, lags: Sequence[int]) -> Dict[int, float]:
    """Sample autocorrelation of the (mean-centered) residuals at each
    requested lag: sum(r_t * r_{t+lag}) / sum(r_t^2). Assumes residuals
    are already in their natural TIME order (do not shuffle before
    calling this).
    """
    r = residuals - residuals.mean()
    denom = np.sum(r ** 2)
    out = {}
    for lag in lags:
        out[lag] = float(np.sum(r[:-lag] * r[lag:]) / denom)
    return out


# ---------------------------------------------------------------------------
# Stage 4: nested-model F-test
# ---------------------------------------------------------------------------

@dataclass
class FTestResult:
    F: float
    df1: int
    df2: int
    pvalue: float


def nested_f_test(fit_restricted: OLSFit, fit_full: OLSFit) -> FTestResult:
    """Classical F-test comparing a restricted (fewer regressors) and
    full (superset of regressors, nested) OLS fit on the SAME data:

        F = [(RSS_restricted - RSS_full) / (p_full - p_restricted)]
            / [RSS_full / (n - p_full)]

    under F(p_full - p_restricted, n - p_full).
    """
    df1 = fit_full.p - fit_restricted.p
    df2 = fit_full.dof
    if df1 <= 0:
        raise ValueError("fit_full must have strictly more parameters than fit_restricted")
    F = ((fit_restricted.rss - fit_full.rss) / df1) / (fit_full.rss / df2)
    pvalue = float(1.0 - stats.f.cdf(F, df1, df2))
    return FTestResult(F=float(F), df1=df1, df2=df2, pvalue=pvalue)


# ---------------------------------------------------------------------------
# Stage 5: contrasts
# ---------------------------------------------------------------------------

@dataclass
class ContrastResult:
    estimate: float
    se: float
    t: float
    pvalue: float
    dof: int


def evaluate_contrast(contrast_vector: np.ndarray, fit: OLSFit) -> ContrastResult:
    """Two-sided t-test of a linear contrast c' beta = 0, using an OLS
    fit's coefficient estimate and covariance matrix.
    """
    c = np.asarray(contrast_vector, dtype=float)
    estimate = float(c @ fit.beta)
    se = float(np.sqrt(c @ fit.cov_beta @ c))
    t = estimate / se
    pvalue = float(2 * (1 - stats.t.cdf(abs(t), fit.dof)))
    return ContrastResult(estimate=estimate, se=se, t=t, pvalue=pvalue, dof=fit.dof)


# ---------------------------------------------------------------------------
# Stage 6: holdout validation
# ---------------------------------------------------------------------------

def predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return X @ beta


def holdout_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    return {"rmse": rmse, "r2": r2}


# ---------------------------------------------------------------------------
# Uncertainty quantification: residual bootstrap
# ---------------------------------------------------------------------------

def residual_bootstrap_ci(X: np.ndarray, fit: OLSFit, contrast_vector: np.ndarray,
                           n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Residual (semiparametric) bootstrap: resample the FITTED model's
    own residuals with replacement, add them back onto the fitted values
    to build a synthetic response, refit, and recompute the contrast.
    Returns an array of length n_boot of bootstrap contrast estimates.
    Valid under the assumption that residuals are (approximately) i.i.d.
    -- worth checking (Task-relevant) whether that assumption looks
    reasonable for the model being bootstrapped.
    """
    fitted_values = X @ fit.beta
    c = np.asarray(contrast_vector, dtype=float)
    out = np.empty(n_boot)
    n = len(fit.residuals)
    for b in range(n_boot):
        resampled_resid = rng.choice(fit.residuals, size=n, replace=True)
        y_boot = fitted_values + resampled_resid
        beta_boot, *_ = np.linalg.lstsq(X, y_boot, rcond=None)
        out[b] = c @ beta_boot
    return out
