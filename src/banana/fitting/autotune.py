"""Data-driven initial guesses and bounds for kinetic/equilibrium fits.

Instead of hardcoded starting parameters, estimate sensible values directly from
each trace (or titration series): association plateau, dissociation-tail decay,
association curvature, and titration midpoint. All routines are pure NumPy and
degrade gracefully to conservative defaults when the data are too short or noisy.

Scope (per project decision): produce p0 + bounds for a single optimization run.
No multi-start / global optimization here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class InitialGuess:
    """A starting point for a fit: values plus matching lower/upper bounds."""

    p0: List[float]
    lower: List[float]
    upper: List[float]
    param_names: List[str] = field(default_factory=list)

    def as_bounds(self) -> Tuple[List[float], List[float]]:
        return self.lower, self.upper

    def clipped(self) -> "InitialGuess":
        """Return a copy with p0 strictly inside [lower, upper]."""
        p0 = []
        for v, lo, hi in zip(self.p0, self.lower, self.upper):
            if not np.isfinite(v):
                v = lo if np.isfinite(lo) else (hi if np.isfinite(hi) else 0.0)
            lo_f = lo if np.isfinite(lo) else -np.inf
            hi_f = hi if np.isfinite(hi) else np.inf
            # Nudge off the exact boundary so least-squares has room to move.
            if hi_f > lo_f:
                span = hi_f - lo_f if np.isfinite(hi_f - lo_f) else 0.0
                eps = 1e-9 * (abs(span) if span else max(abs(v), 1.0))
                v = min(max(v, lo_f + eps), hi_f - eps)
            p0.append(float(v))
        return InitialGuess(p0=p0, lower=list(self.lower), upper=list(self.upper),
                            param_names=list(self.param_names))


# --------------------------------------------------------------------------- #
# Low-level single-trace estimators
# --------------------------------------------------------------------------- #

def _finite(*arrays: np.ndarray) -> bool:
    return all(a is not None and len(a) > 0 for a in arrays)


def estimate_plateau(
    t: np.ndarray, r: np.ndarray, t1: Optional[float], t2: Optional[float],
    frac: float = 0.2,
) -> float:
    """Association plateau (signal asymptote): mean of the last `frac` of the
    association window [t1, t2). Falls back to the overall max."""
    t = np.asarray(t, dtype=float)
    r = np.asarray(r, dtype=float)
    if not _finite(t, r):
        return 0.0
    if t1 is not None and t2 is not None and t2 > t1:
        in_assoc = (t >= t1) & (t < t2)
        seg = r[in_assoc]
        if seg.size >= 3:
            n = max(3, int(round(seg.size * frac)))
            return float(np.mean(seg[-n:]))
    n = max(3, int(round(len(r) * 0.05)))
    return float(np.mean(np.sort(r)[-n:]))


def estimate_baseline(t: np.ndarray, r: np.ndarray, n: int = 50) -> float:
    """Pre-association baseline: mean of the first few points."""
    r = np.asarray(r, dtype=float)
    if not _finite(r):
        return 0.0
    n = max(1, min(n, len(r) // 10 or 1))
    return float(np.mean(r[:n]))


def estimate_koff(
    t: np.ndarray, r: np.ndarray, t2: Optional[float],
    baseline: float = 0.0,
) -> Optional[float]:
    """Dissociation rate from the log-slope of (R - baseline) over the
    dissociation tail (t >= t2). Returns None if not estimable."""
    t = np.asarray(t, dtype=float)
    r = np.asarray(r, dtype=float)
    if not _finite(t, r) or t2 is None:
        return None
    mask = t >= t2
    td, rd = t[mask], r[mask]
    if td.size < 5:
        return None
    y = rd - baseline
    # Keep the decaying, positive part; need a meaningful amplitude.
    y0 = y[0] if y[0] != 0 else (np.max(np.abs(y)) or 1.0)
    sign = np.sign(y0) or 1.0
    yy = sign * y
    good = yy > (0.02 * abs(y0))
    if good.sum() < 5:
        return None
    tt = td[good] - td[good][0]
    ly = np.log(yy[good])
    try:
        slope, _ = np.polyfit(tt, ly, 1)
    except Exception:
        return None
    koff = -float(slope)
    if not np.isfinite(koff) or koff <= 0:
        return None
    return koff


def estimate_tau_association(
    t: np.ndarray, r: np.ndarray, t1: Optional[float], t2: Optional[float],
    A0: float, A1: float,
) -> Optional[float]:
    """Association time constant tau1 from the 1/e crossing of the rise.

    R(t) = A1 + (A0 - A1) exp(-(t-t1)/tau1); the response reaches
    A0 + (1 - 1/e)(A1 - A0) at t = t1 + tau1.
    """
    t = np.asarray(t, dtype=float)
    r = np.asarray(r, dtype=float)
    if not _finite(t, r) or t1 is None or t2 is None or t2 <= t1:
        return None
    mask = (t >= t1) & (t < t2)
    ta, ra = t[mask], r[mask]
    if ta.size < 5 or A1 == A0:
        return None
    target = A0 + (1.0 - 1.0 / np.e) * (A1 - A0)
    # First crossing of target (works for rise or fall).
    rising = A1 > A0
    crossed = (ra >= target) if rising else (ra <= target)
    idx = np.argmax(crossed) if crossed.any() else -1
    if idx <= 0:
        # Fall back to half the association window.
        return float((ta[-1] - ta[0]) / 2.0) or None
    tau = float(ta[idx] - t1)
    if not np.isfinite(tau) or tau <= 0:
        return None
    return tau


# --------------------------------------------------------------------------- #
# Model-level initial guesses
# --------------------------------------------------------------------------- #

def piecewise_initial_guess(
    t: np.ndarray, r: np.ndarray, A0: float, t1: float, t2: float,
) -> InitialGuess:
    """p0 + bounds for PiecewiseExponential params [A1, tau1, A2, tau2]."""
    t = np.asarray(t, dtype=float)
    r = np.asarray(r, dtype=float)
    rng = float(np.max(r) - np.min(r)) if _finite(r) else 1.0
    rng = rng if rng > 0 else 1.0

    A1 = estimate_plateau(t, r, t1, t2)            # association asymptote
    A2 = estimate_plateau(t, r, t2, (t[-1] if len(t) else t2) + 1, frac=0.3) \
        if len(t) else A0                           # dissociation asymptote
    tau1 = estimate_tau_association(t, r, t1, t2, A0, A1) or 10.0
    koff = estimate_koff(t, r, t2, baseline=A2)
    tau2 = (1.0 / koff) if koff else max(tau1, 10.0)

    span_lo = float(np.min(r)) - rng if _finite(r) else -np.inf
    span_hi = float(np.max(r)) + rng if _finite(r) else np.inf
    dur = float(t[-1] - t[0]) if len(t) > 1 else 1e4
    tau_hi = max(dur * 10.0, 1e3)

    return InitialGuess(
        p0=[A1, tau1, A2, tau2],
        lower=[span_lo, 1e-3, span_lo, 1e-3],
        upper=[span_hi, tau_hi, span_hi, tau_hi],
        param_names=["A1", "tau1", "A2", "tau2"],
    ).clipped()


def kobs_koff_from_titration(
    times: Sequence[np.ndarray],
    responses: Sequence[np.ndarray],
    concentrations_nM: Sequence[float],
    t1_list: Optional[Sequence[float]] = None,
    t2_list: Optional[Sequence[float]] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Estimate (k_on, k_off, K_D) from a titration series.

    Per trace: k_obs = 1/tau1, k_off from the dissociation tail. Then regress
    k_obs ~ k_on * [conc] + k_off across traces.
    """
    kobs, koff_vals, conc_M = [], [], []
    n = len(times)
    for i in range(n):
        t = np.asarray(times[i], dtype=float)
        r = np.asarray(responses[i], dtype=float)
        if t.size < 5:
            continue
        t1 = float(t1_list[i]) if t1_list is not None and i < len(t1_list) else float(t[0])
        t2 = float(t2_list[i]) if t2_list is not None and i < len(t2_list) else float(t[-1])
        A0 = estimate_baseline(t, r)
        A1 = estimate_plateau(t, r, t1, t2)
        tau1 = estimate_tau_association(t, r, t1, t2, A0, A1)
        koff = estimate_koff(t, r, t2, baseline=estimate_plateau(t, r, t2, t[-1] + 1, frac=0.3))
        c = float(concentrations_nM[i]) * 1e-9
        if tau1 and c > 0:
            kobs.append(1.0 / tau1)
            conc_M.append(c)
        if koff:
            koff_vals.append(koff)

    koff_est = float(np.median(koff_vals)) if koff_vals else None
    if len(conc_M) >= 2:
        A = np.column_stack([np.asarray(conc_M), np.ones(len(conc_M))])
        try:
            x, *_ = np.linalg.lstsq(A, np.asarray(kobs), rcond=None)
            k_on = float(x[0])
            k_off_fit = float(x[1])
        except Exception:
            k_on, k_off_fit = None, None
    else:
        k_on, k_off_fit = None, None

    koff_final = koff_est if koff_est else (k_off_fit if (k_off_fit and k_off_fit > 0) else None)
    if k_on is not None and koff_final and k_on > 0:
        K_D = koff_final / k_on
    else:
        K_D = None
    return k_on, koff_final, K_D


def one_to_one_initial_guess(
    times: Sequence[np.ndarray],
    responses: Sequence[np.ndarray],
    concentrations_nM: Sequence[float],
    t1_list: Optional[Sequence[float]] = None,
    t2_list: Optional[Sequence[float]] = None,
) -> InitialGuess:
    """p0 + bounds for global 1:1 association fit: [k_on, k_off, R_max_0, ...]."""
    n = len(times)
    k_on, k_off, _ = kobs_koff_from_titration(
        times, responses, concentrations_nM, t1_list, t2_list
    )
    k_on = k_on if (k_on and k_on > 0) else 1e4          # M^-1 s^-1
    k_off = k_off if (k_off and k_off > 0) else 1e-3      # s^-1
    r_max = []
    for i in range(n):
        t = np.asarray(times[i], dtype=float)
        r = np.asarray(responses[i], dtype=float)
        t1 = float(t1_list[i]) if t1_list is not None and i < len(t1_list) else None
        t2 = float(t2_list[i]) if t2_list is not None and i < len(t2_list) else None
        r_max.append(max(estimate_plateau(t, r, t1, t2), 1e-6))
    r_hi = max(r_max) * 5.0 if r_max else np.inf

    p0 = [k_on, k_off] + r_max
    lower = [1.0, 1e-6] + [0.0] * n
    upper = [1e9, 1e2] + [r_hi] * n
    names = ["k_on", "k_off"] + [f"R_max_{i}" for i in range(n)]
    return InitialGuess(p0=p0, lower=lower, upper=upper, param_names=names).clipped()
