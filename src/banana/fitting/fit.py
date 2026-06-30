"""Fitting routines for kinetic models."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit, least_squares

from banana.models.kinetics import KineticModel, OneToOneBinding, PiecewiseExponential
from banana.io.titration import TitrationSeries, association_dissociation_start_times
from banana.fitting.autotune import (
    InitialGuess,
    piecewise_initial_guess,
    one_to_one_initial_guess,
)

logger = logging.getLogger(__name__)


def _safe_float(x: Any) -> Optional[float]:
    """Convert to float if possible; return None on failure."""
    if x is None:
        return None
    try:
        if isinstance(x, (np.floating, np.integer)):
            v = float(x)
        elif isinstance(x, np.ndarray):
            if x.size != 1:
                return None
            v = float(x.flat[0])
        else:
            v = float(x)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def _extra_for_export(extra: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only CSV-safe scalar entries from extra."""
    out = {}
    if not extra:
        return out
    skip_keys = {"trace_results", "concentrations_M", "k_obs_fitted", "k_obs_observed"}
    for k, v in extra.items():
        if k in skip_keys:
            continue
        if isinstance(v, (np.ndarray, list)) and k not in ("A0", "t1", "t2"):
            # Skip large arrays; optional small lists could be stringified
            continue
        sf = _safe_float(v)
        if sf is not None:
            out[k] = sf
        elif isinstance(v, str):
            out[k] = v
        elif isinstance(v, bool):
            out[k] = v
    return out


@dataclass
class FitResult:
    """Result of a kinetic fit."""

    success: bool
    params: np.ndarray
    param_names: List[str]
    cov: Optional[np.ndarray]
    residuals: Optional[np.ndarray]
    chi2: Optional[float]
    message: str
    extra: Dict[str, Any] = None

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}
        if self.params is None:
            self.params = np.array([])
        self.params = np.asarray(self.params, dtype=float).ravel()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export; never raises on missing params."""
        d: Dict[str, Any] = {
            "success": bool(self.success),
            "message": str(self.message) if self.message else "",
        }
        n_params = len(self.params)
        n_names = len(self.param_names)
        n = min(n_params, n_names)
        if n_params != n_names and (n_params > 0 or n_names > 0):
            logger.debug(
                "FitResult.to_dict: param count (%s) != param_names count (%s); exporting min slice",
                n_params,
                n_names,
            )
        for i in range(n):
            name = self.param_names[i]
            val = _safe_float(self.params[i])
            if val is not None:
                d[name] = val
            else:
                d[name] = None
            if self.cov is not None and self.cov.size > 0:
                try:
                    if i < self.cov.shape[0]:
                        stderr = _safe_float(np.sqrt(np.abs(self.cov[i, i])))
                        if stderr is not None:
                            d[f"{name}_stderr"] = stderr
                except Exception:
                    pass
        if self.chi2 is not None:
            c2 = _safe_float(self.chi2)
            if c2 is not None:
                d["chi2"] = c2
        d.update(_extra_for_export(self.extra))
        return d


def fit_single_trace(
    time: np.ndarray,
    response: np.ndarray,
    model: KineticModel,
    p0: Optional[List[float]] = None,
    bounds: Optional[Tuple[List[float], List[float]]] = None,
    **curve_fit_kw,
) -> FitResult:
    """
    Fit a kinetic model to a single time-response trace.

    Returns
    -------
    FitResult
        On failure, success=False and params may be empty.
    """
    try:
        if len(time) < model.n_params() + 1:
            return FitResult(
                success=False,
                params=np.array([]),
                param_names=model.param_names(),
                cov=None,
                residuals=None,
                chi2=None,
                message=f"Too few points ({len(time)}) for model",
            )
        popt, pcov = curve_fit(
            model,
            time,
            response,
            p0=p0 or [0.05, 10, 0.04, 10][: model.n_params()],
            bounds=bounds or (-np.inf, np.inf),
            **curve_fit_kw,
        )
        resid = response - model(time, *popt)
        chi2 = float(np.sum(resid**2))
        return FitResult(
            success=True,
            params=popt,
            param_names=model.param_names(),
            cov=pcov,
            residuals=resid,
            chi2=chi2,
            message="Optimization converged",
        )
    except Exception as e:
        logger.warning("fit_single_trace failed: %s", e)
        return FitResult(
            success=False,
            params=np.array([]),
            param_names=model.param_names(),
            cov=None,
            residuals=None,
            chi2=None,
            message=str(e),
        )


def fit_titration_global(
    titration: TitrationSeries,
    model_type: str = "piecewise",
    shared_params: Optional[List[str]] = None,
    p0: Optional[Dict[str, float]] = None,
) -> Tuple[FitResult, List[FitResult]]:
    """
    Global fit of a kinetic model across a titration series.
    """
    p0 = p0 or {}
    trace_results: List[FitResult] = []

    if len(titration) == 0:
        return FitResult(
            success=False,
            params=np.array([]),
            param_names=["k_on", "k_off"],
            cov=None,
            residuals=None,
            chi2=None,
            message="Empty titration series",
        ), trace_results

    if model_type == "piecewise":
        k_obs_list = []
        k_off_list = []
        conc_list = []

        for i in range(len(titration)):
            t, r = titration.get_trace(i)
            conc = titration.concentration[i]

            if len(t) < 10:
                trace_results.append(
                    FitResult(
                        success=False,
                        params=np.array([]),
                        param_names=PiecewiseExponential(0, 0, 1).param_names(),
                        cov=None,
                        residuals=None,
                        chi2=None,
                        message="Too few time points",
                    )
                )
                continue

            A0 = float(np.mean(r[: min(50, max(1, len(r) // 10))]))

            am = (
                titration.assoc_mask[i]
                if titration.assoc_mask and i < len(titration.assoc_mask)
                else None
            )
            dm = (
                titration.dissoc_mask[i]
                if titration.dissoc_mask and i < len(titration.dissoc_mask)
                else None
            )
            if (
                titration.association_start_t is not None
                and titration.dissociation_start_t is not None
                and i < len(titration.association_start_t)
                and i < len(titration.dissociation_start_t)
            ):
                t1 = float(titration.association_start_t[i])
                t2 = float(titration.dissociation_start_t[i])
            else:
                t1, t2 = association_dissociation_start_times(t, am, dm)

            model = PiecewiseExponential(A0=A0, t1=t1, t2=t2)
            # Data-driven initial guess + physical bounds (autotuning), with
            # optional per-parameter overrides from p0.
            guess = piecewise_initial_guess(t, r, A0=A0, t1=t1, t2=t2)
            init = list(guess.p0)
            for j, name in enumerate(["A1", "tau1", "A2", "tau2"]):
                if name in p0:
                    init[j] = p0[name]
            init_guess = InitialGuess(
                p0=init, lower=guess.lower, upper=guess.upper,
                param_names=guess.param_names,
            ).clipped()
            res = fit_single_trace(
                t, r, model, p0=init_guess.p0, bounds=init_guess.as_bounds()
            )
            trace_results.append(res)

            if res.success and len(res.params) >= 4:
                res.extra = dict(res.extra or {})
                res.extra["A0"] = A0
                res.extra["t1"] = t1
                res.extra["t2"] = t2
                tau1, tau2 = res.params[1], res.params[3]
                k_obs = 1.0 / max(np.abs(tau1), 1e-12)
                k_off = 1.0 / max(np.abs(tau2), 1e-12)
                if conc > 0 and k_obs > k_off:
                    k_obs_list.append(k_obs)
                    k_off_list.append(k_off)
                    conc_list.append(conc * 1e-9)

        if len(conc_list) < 2:
            return FitResult(
                success=False,
                params=np.array([np.nan, np.nan]),
                param_names=["k_on", "k_off"],
                cov=None,
                residuals=None,
                chi2=None,
                message="Insufficient valid traces for global fit",
                extra={"trace_results": trace_results},
            ), trace_results

        conc_arr = np.array(conc_list)
        k_obs_arr = np.array(k_obs_list)
        A = np.column_stack([conc_arr, np.ones_like(conc_arr)])
        try:
            x, _, _, _ = np.linalg.lstsq(A, k_obs_arr, rcond=None)
            k_on, k_off = float(x[0]), float(x[1])
        except Exception as e:
            logger.warning("Global linear fit failed: %s", e)
            return FitResult(
                success=False,
                params=np.array([]),
                param_names=["k_on", "k_off"],
                cov=None,
                residuals=None,
                chi2=None,
                message=str(e),
                extra={"trace_results": trace_results},
            ), trace_results

        K_d = k_off / k_on if k_on > 0 else np.nan
        return FitResult(
            success=True,
            params=np.array([k_on, k_off]),
            param_names=["k_on", "k_off"],
            cov=None,
            residuals=np.array([]),
            chi2=float(np.sum((k_obs_arr - (k_on * conc_arr + k_off)) ** 2)),
            message="Global fit converged",
            extra={
                "K_d": K_d,
                "concentrations_M": conc_arr,
                "k_obs_fitted": k_on * conc_arr + k_off,
                "k_obs_observed": k_obs_arr,
                "trace_results": trace_results,
            },
        ), trace_results

    elif model_type == "one_to_one":
        def residuals(params):
            k_on, k_off = params[0], params[1]
            R_max = params[2:]
            res = []
            for j in range(len(titration)):
                t, r = titration.get_trace(j)
                conc = titration.concentration[j] * 1e-9
                R_m = R_max[j] if j < len(R_max) else R_max[-1]
                pred = OneToOneBinding(phase="association").association(
                    t, k_on, k_off, R_m, conc * 1e9
                )
                res.extend((r - pred).ravel())
            return np.array(res)

        n_traces = len(titration)
        times = [titration.get_trace(j)[0] for j in range(n_traces)]
        resps = [titration.get_trace(j)[1] for j in range(n_traces)]
        guess = one_to_one_initial_guess(
            times, resps, list(titration.concentration),
            t1_list=titration.association_start_t,
            t2_list=titration.dissociation_start_t,
        )
        x0 = list(guess.p0)
        # Honor explicit overrides for the shared rate constants.
        if "k_on" in p0:
            x0[0] = p0["k_on"]
        if "k_off" in p0:
            x0[1] = p0["k_off"]
        guess = InitialGuess(
            p0=x0, lower=guess.lower, upper=guess.upper,
            param_names=guess.param_names,
        ).clipped()

        try:
            ls = least_squares(residuals, guess.p0, bounds=tuple(guess.as_bounds()))
            popt = ls.x
            k_on, k_off = popt[0], popt[1]
            K_d = k_off / k_on if k_on > 0 else np.nan
            return FitResult(
                success=bool(ls.success),
                params=popt,
                param_names=["k_on", "k_off"] + [f"R_max_{j}" for j in range(n_traces)],
                cov=None,
                residuals=ls.fun,
                chi2=float(np.sum(ls.fun**2)),
                message=str(ls.message or "Optimization finished"),
                extra={"K_d": K_d, "trace_results": trace_results},
            ), trace_results
        except Exception as e:
            logger.warning("one_to_one global fit failed: %s", e)
            return FitResult(
                success=False,
                params=np.array([]),
                param_names=[],
                cov=None,
                residuals=None,
                chi2=None,
                message=str(e),
                extra={"trace_results": trace_results},
            ), trace_results

    elif model_type == "avidity":
        # Per-trace bivalent/avidity ODE fit (Vauquelin 2013). Scheme and fixed
        # quantities (local conc L, penalty mode) come from p0/options.
        from banana.models.avidity import AvidityModel

        scheme = str(p0.get("scheme", "heterobivalent"))
        symmetric = bool(p0.get("symmetric", scheme == "heterobivalent"))
        L = float(p0.get("L", 1e-3))
        rebind_k = float(p0.get("rebind_k", 0.0))

        for i in range(len(titration)):
            t, r = titration.get_trace(i)
            conc = titration.concentration[i]
            if len(t) < 10:
                trace_results.append(FitResult(
                    False, np.array([]), [], None, None, None, "Too few points"))
                continue
            t1 = float(titration.association_start_t[i]) \
                if titration.association_start_t else float(t[0])
            t2 = float(titration.dissociation_start_t[i]) \
                if titration.dissociation_start_t else float(t[-1])
            model = AvidityModel(
                scheme=scheme, t1=t1, t2=t2, conc_M=conc * 1e-9,
                L=L, rebind_k=rebind_k, symmetric=symmetric,
            )
            R_max0 = float(np.max(r)) if len(r) else 1.0
            init_map = {
                "R_max": R_max0, "k1": p0.get("k1", 1e5),
                "k_off1": p0.get("k_off1", 1e-2), "k2": p0.get("k2", 1e5),
                "k_off2": p0.get("k_off2", 1e-2), "f": p0.get("f", 1.0),
            }
            init = [init_map[n] for n in model.param_names()]
            lower = [0.0] + [1e-3 if "k1" in n or "k2" in n else 1e-6
                             for n in model.param_names()[1:]]
            upper = [5 * R_max0] + [1e9 if n in ("k1", "k2") else
                                    (1e3 if n == "f" else 1e2)
                                    for n in model.param_names()[1:]]
            res = fit_single_trace(t, r, model, p0=init, bounds=(lower, upper))
            if res.success:
                res.extra = dict(res.extra or {})
                res.extra.update({"scheme": scheme, "conc_nM": conc})
                res.extra.update(model.avidity_metrics(*res.params))
            trace_results.append(res)

        ok = [r for r in trace_results if r.success]
        return FitResult(
            success=bool(ok),
            params=np.array([]),
            param_names=[],
            cov=None,
            residuals=None,
            chi2=float(np.nansum([r.chi2 for r in ok if r.chi2 is not None])) if ok else None,
            message=f"Avidity ({scheme}) fit: {len(ok)}/{len(trace_results)} traces",
            extra={"trace_results": trace_results, "scheme": scheme},
        ), trace_results

    else:
        return FitResult(
            success=False,
            params=np.array([]),
            param_names=[],
            cov=None,
            residuals=None,
            chi2=None,
            message=f"Unknown model_type: {model_type}",
        ), trace_results
