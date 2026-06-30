"""Steady-state extraction and equilibrium-model fitting.

Bridges the time-domain `TitrationSeries` to the concentration-domain
`EquilibriumModel`s: reduce each trace to a steady-state response R_eq, then fit
R_eq vs. concentration with curve_fit.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit

from banana.models.equilibrium import (
    EquilibriumModel,
    HyperbolicBinding,
    QuadraticBinding,
    TwoSiteMicroscopic,
    TwoSiteMacroscopic,
    HillBinding,
)
from banana.fitting.autotune import estimate_plateau, InitialGuess
from banana.fitting.fit import FitResult

logger = logging.getLogger(__name__)


def steady_state_response(
    titration, frac: float = 0.2
) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce each trace to (concentration_M, R_eq).

    R_eq is the association-phase plateau (mean of the last `frac` of the
    association window). Concentrations are converted nM -> M.
    """
    conc_M, R_eq = [], []
    n = len(titration)
    for i in range(n):
        t, r = titration.get_trace(i)
        t = np.asarray(t, dtype=float)
        if t.size == 0:
            continue
        t1 = t2 = None
        if titration.association_start_t and i < len(titration.association_start_t):
            t1 = float(titration.association_start_t[i])
        if titration.dissociation_start_t and i < len(titration.dissociation_start_t):
            t2 = float(titration.dissociation_start_t[i])
        R_eq.append(estimate_plateau(t, r, t1, t2, frac=frac))
        conc_M.append(float(titration.concentration[i]) * 1e-9)
    order = np.argsort(conc_M)
    return np.asarray(conc_M)[order], np.asarray(R_eq)[order]


_MODEL_FACTORY = {
    "hyperbolic": HyperbolicBinding,
    "quadratic": QuadraticBinding,
    "two_site_microscopic": TwoSiteMicroscopic,
    "two_site_macroscopic": TwoSiteMacroscopic,
    "hill": HillBinding,
}


def _default_guess(model: EquilibriumModel, conc: np.ndarray,
                   signal: np.ndarray) -> InitialGuess:
    """Data-driven p0 + bounds for an equilibrium model (autotuning)."""
    c = np.asarray(conc, dtype=float)
    s = np.asarray(signal, dtype=float)
    r_max = float(np.max(s)) if s.size else 1.0
    r_max = r_max if r_max > 0 else 1.0
    # K_D ~ concentration at half-max signal (geometric midpoint fallback).
    half = 0.5 * r_max
    try:
        kd = float(np.interp(half, s, c))
    except Exception:
        kd = float(np.median(c)) if c.size else 1e-9
    if not np.isfinite(kd) or kd <= 0:
        kd = float(np.median(c[c > 0])) if np.any(c > 0) else 1e-9
    c_lo = float(np.min(c[c > 0])) if np.any(c > 0) else 1e-12
    c_hi = float(np.max(c)) if c.size else 1e-3
    names = model.param_names()
    p0, lo, hi = [], [], []
    for nm in names:
        if nm == "R_max":
            p0.append(r_max); lo.append(0.0); hi.append(5.0 * r_max)
        elif nm in ("K_D", "K_D1", "K_D_M", "K50"):
            p0.append(kd); lo.append(c_lo * 1e-3); hi.append(c_hi * 1e3)
        elif nm == "K_D2":
            p0.append(kd * 10.0); lo.append(c_lo * 1e-3); hi.append(c_hi * 1e4)
        elif nm == "alpha":
            p0.append(1.0); lo.append(1e-3); hi.append(1e3)
        elif nm == "n_Hill":
            p0.append(1.0); lo.append(0.1); hi.append(8.0)
        elif nm == "R_tot":
            p0.append(c_lo); lo.append(0.0); hi.append(c_hi)
        elif nm == "baseline":
            p0.append(float(np.min(s)) if s.size else 0.0)
            lo.append(-abs(r_max)); hi.append(abs(r_max))
        else:
            p0.append(1.0); lo.append(-np.inf); hi.append(np.inf)
    return InitialGuess(p0=p0, lower=lo, upper=hi, param_names=names).clipped()


def fit_equilibrium(
    conc_M: np.ndarray,
    signal: np.ndarray,
    model: str = "hyperbolic",
    p0: Optional[List[float]] = None,
    bounds: Optional[Tuple[List[float], List[float]]] = None,
    model_kwargs: Optional[dict] = None,
    **curve_fit_kw,
) -> FitResult:
    """Fit an equilibrium binding model to a signal-vs-concentration curve.

    Parameters
    ----------
    conc_M : array
        Total titrant concentration in molar.
    signal : array
        Steady-state observable (e.g. R_eq).
    model : str
        One of: hyperbolic, quadratic, two_site_microscopic,
        two_site_macroscopic, hill.
    """
    conc_M = np.asarray(conc_M, dtype=float)
    signal = np.asarray(signal, dtype=float)
    key = str(model).strip().lower()
    if key not in _MODEL_FACTORY:
        return FitResult(False, np.array([]), [], None, None, None,
                         f"Unknown equilibrium model: {model}")
    mdl: EquilibriumModel = _MODEL_FACTORY[key](**(model_kwargs or {}))

    if conc_M.size < mdl.n_params() + 1:
        return FitResult(False, np.array([]), mdl.param_names(), None, None, None,
                         f"Too few points ({conc_M.size}) for {key}")

    guess = _default_guess(mdl, conc_M, signal)
    if p0 is not None:
        guess = InitialGuess(p0=list(p0), lower=guess.lower, upper=guess.upper,
                             param_names=guess.param_names).clipped()
    use_bounds = bounds if bounds is not None else guess.as_bounds()

    def f(c, *params):
        return mdl(c, *params)

    try:
        popt, pcov = curve_fit(f, conc_M, signal, p0=guess.p0,
                               bounds=use_bounds, maxfev=20000, **curve_fit_kw)
        resid = signal - mdl(conc_M, *popt)
        chi2 = float(np.sum(resid ** 2))
        extra = {"model": key}
        if key == "two_site_macroscopic":
            extra["K_D_M"] = float(abs(popt[1]))
            extra["alpha"] = float(abs(popt[2]))
        return FitResult(True, popt, mdl.param_names(), pcov, resid, chi2,
                         "Equilibrium fit converged", extra=extra)
    except Exception as e:
        logger.warning("fit_equilibrium (%s) failed: %s", key, e)
        return FitResult(False, np.array([]), mdl.param_names(), None, None, None,
                         str(e))


def fit_titration_equilibrium(
    titration, model: str = "hyperbolic", frac: float = 0.2,
    model_kwargs: Optional[dict] = None, **kw,
) -> FitResult:
    """Convenience: extract steady-state R_eq from a TitrationSeries and fit it."""
    conc_M, R_eq = steady_state_response(titration, frac=frac)
    res = fit_equilibrium(conc_M, R_eq, model=model,
                          model_kwargs=model_kwargs, **kw)
    res.extra = dict(res.extra or {})
    res.extra["concentrations_M"] = conc_M
    res.extra["R_eq"] = R_eq
    return res
