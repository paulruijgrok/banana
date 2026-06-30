"""Bivalent / avidity binding kinetics (Vauquelin 2013, Appendix S1).

Time-domain ODE models for mono-, homo-, and heterobivalent ligands binding to
target sites, integrated numerically. Unlike the simple kinetic models, these
capture avidity (raised functional affinity) and prolonged residence time that
arise when a bivalent ligand engages two proximate sites.

Schemes (Appendix S1, Table A1)
-------------------------------
- "monovalent"     : a + A <-> aA                              (Eqs. 1-2)
- "homobivalent"   : aa + A <-> aaA  (isolated site, fwd x2)   (Eqs. 3-4)
- "heterobivalent" : ab + AB target-pair, 5 species, avidity   (Eqs. 5-9)

The heterobivalent scheme is the avidity model of interest for BLI of a bivalent
binder to a target-pair. Set k2=k1, k_off2=k_off1 to model a symmetric
(homobivalent) ligand binding bivalently.

Rate-constant units follow the paper conventions but are time-unit agnostic:
k_on (k1,k2) in M^-1 (time)^-1, k_off (k_off1,k_off2) in (time)^-1, with the
trace time axis defining the time unit. Local concentration `L` is in M.

Ligand concentration is held constant during the association window [t1, t2) and
set to zero during dissociation (t >= t2), matching a BLI assoc/dissoc cycle.

Production integration uses scipy.integrate.solve_ivp; the RHS functions are pure
NumPy so they can be integrated by any solver.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from banana.models.kinetics import KineticModel


# --------------------------------------------------------------------------- #
# Right-hand sides (pure NumPy). State vectors documented per scheme.
# --------------------------------------------------------------------------- #

def rhs_monovalent(y, ligand, k1, k_off1, A_tot, rebind_k=0.0):
    """State y = [A, aA].  Eqs. 1-2 (optional hindered-diffusion rebinding)."""
    A, aA = y
    kon, koff = k1, k_off1
    if rebind_k:
        denom = 1.0 + k1 * A * rebind_k
        kon = k1 / denom
        koff = k_off1 / denom
    d_aA = kon * ligand * A - koff * aA
    return np.array([-d_aA, d_aA])


def rhs_homobivalent(y, ligand, k1, k_off1, A_tot, rebind_k=0.0):
    """State y = [A, aaA].  Eqs. 3-4 (forward rate doubled)."""
    A, aaA = y
    kon, koff = k1, k_off1
    if rebind_k:
        denom = 1.0 + 2.0 * k1 * A * rebind_k
        kon = k1 / denom
        koff = k_off1 / denom
    d_aaA = 2.0 * kon * ligand * A - koff * aaA
    return np.array([-d_aaA, d_aaA])


def rhs_heterobivalent(y, ligand, k1, k_off1, k2, k_off2, f, L):
    """State y = [AB, aAB, ABb, aABb, apABbp].  Eqs. 5-9.

    aAB   : 'a' arm bound (site A occupied)
    ABb   : 'b' arm bound (site B occupied)
    aABb  : both arms of the SAME ligand bound (ring-closed / avidity)
    apABbp: two SEPARATE ligands bridging (a' and b')
    L     : local concentration driving intramolecular ring closure.
    """
    AB, aAB, ABb, aABb, apABbp = y
    ab = ligand
    d_AB = k_off1 * aAB + k_off2 * ABb - (k1 + k2) * AB * ab
    d_aAB = (k1 * AB * ab + k_off2 * (aABb + apABbp) - k_off1 * aAB
             - (L * k2 / f) * aAB - k2 * aAB * ab)
    d_ABb = (k2 * AB * ab + k_off1 * (aABb + apABbp) - k_off2 * ABb
             - (L * k1 / f) * ABb - k1 * ABb * ab)
    d_aABb = (L * k1 / f) * ABb + (L * k2 / f) * aAB - (k_off1 + k_off2) * aABb
    d_apABbp = k1 * ABb * ab + k2 * aAB * ab - (k_off1 + k_off2) * apABbp
    return np.array([d_AB, d_aAB, d_ABb, d_aABb, d_apABbp])


_SCHEMES = {
    "monovalent": {"n_state": 2, "bound_idx": [1], "stoich": [1]},
    "homobivalent": {"n_state": 2, "bound_idx": [1], "stoich": [1]},
    "heterobivalent": {"n_state": 5, "bound_idx": [1, 2, 3, 4], "stoich": [1, 1, 1, 1]},
}


def _rk4(rhs: Callable, y0: np.ndarray, t_eval: np.ndarray, ligand: float,
         args: tuple, max_rate: float = 1.0) -> np.ndarray:
    """Fixed-step RK4 fallback integrator (used if scipy is unavailable).

    Substep size is chosen for stability from `max_rate` (largest first-order
    rate coefficient): h * max_rate <= 0.1. Capped to bound the work; clips
    negatives. Returns array shape (len(t_eval), len(y0)). For very stiff
    systems prefer the scipy (LSODA) path.
    """
    out = np.empty((len(t_eval), len(y0)))
    out[0] = y0
    y = y0.astype(float)
    h_target = 0.1 / max(max_rate, 1e-12)
    for i in range(1, len(t_eval)):
        span = t_eval[i] - t_eval[i - 1]
        nsub = int(np.ceil(span / h_target)) if span > 0 else 1
        nsub = max(1, min(nsub, 200000))
        h = span / nsub
        for _ in range(nsub):
            k1 = rhs(y, ligand, *args)
            k2 = rhs(y + 0.5 * h * k1, ligand, *args)
            k3 = rhs(y + 0.5 * h * k2, ligand, *args)
            k4 = rhs(y + h * k3, ligand, *args)
            y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            y = np.maximum(y, 0.0)
        out[i] = y
    return out


def _max_rate_hint(args: tuple, ligand: float) -> float:
    """Upper bound on first-order rate coefficients, for RK4 step sizing."""
    rates = [0.0]
    if len(args) == 4:                       # mono/homo: (k1, k_off1, A_tot, k)
        k1, k_off1, A_tot, _ = args
        rates += [abs(k1) * abs(ligand), abs(k_off1)]
    else:                                     # hetero: (k1,koff1,k2,koff2,f,L)
        k1, k_off1, k2, k_off2, f, L = args
        f = abs(f) or 1.0
        rates += [abs(k1) * abs(ligand), abs(k2) * abs(ligand),
                  abs(k_off1), abs(k_off2), L * abs(k1) / f, L * abs(k2) / f]
    return max(rates)


def _integrate_segment(rhs, y0, t_eval, ligand, args, use_scipy=True):
    if use_scipy:
        try:
            from scipy.integrate import solve_ivp
            sol = solve_ivp(
                lambda t, y: rhs(y, ligand, *args),
                (t_eval[0], t_eval[-1]), y0, t_eval=t_eval,
                method="LSODA", rtol=1e-7, atol=1e-12,
            )
            if sol.success:
                return np.clip(sol.y.T, 0.0, None)
        except Exception:
            pass
    max_rate = _max_rate_hint(args, ligand)
    return _rk4(rhs, np.asarray(y0, float), np.asarray(t_eval, float),
                ligand, args, max_rate=max_rate)


def simulate_avidity(
    t: np.ndarray,
    t1: float,
    t2: float,
    conc_M: float,
    scheme: str,
    *,
    k1: float, k_off1: float,
    k2: float = None, k_off2: float = None,
    f: float = 1.0, L: float = 1e-3, rebind_k: float = 0.0,
    target_tot: float = 1.0,
    use_scipy: bool = True,
) -> Dict[str, np.ndarray]:
    """Integrate an avidity scheme over the trace time grid.

    Ligand concentration is `conc_M` during [t1, t2) and 0 elsewhere. Returns a
    dict with the species arrays (on the t grid) and 'occupancy' (0..1) and
    'bound' (stoichiometry-weighted) signals scaled to total target `target_tot`.
    """
    t = np.asarray(t, dtype=float)
    scheme = scheme.lower()
    if scheme not in _SCHEMES:
        raise ValueError(f"Unknown avidity scheme: {scheme}")
    info = _SCHEMES[scheme]
    n = info["n_state"]

    if scheme == "monovalent":
        rhs = rhs_monovalent
        args = (k1, k_off1, target_tot, rebind_k)
    elif scheme == "homobivalent":
        rhs = rhs_homobivalent
        args = (k1, k_off1, target_tot, rebind_k)
    else:
        k2 = k1 if k2 is None else k2
        k_off2 = k_off1 if k_off2 is None else k_off2
        rhs = rhs_heterobivalent
        args = (k1, k_off1, k2, k_off2, f, L)

    # Initial state: all target free.
    y0 = np.zeros(n)
    y0[0] = target_tot

    # Build the three phases against the actual grid points.
    states = np.zeros((len(t), n))
    # Baseline phase t < t1 : ligand 0, nothing happens (state constant).
    pre = t < t1
    states[pre] = y0
    # Association phase t1 <= t < t2.
    assoc = (t >= t1) & (t < t2)
    # Dissociation phase t >= t2.
    dissoc = t >= t2

    y_cur = y0.copy()
    # Integrate association on its own grid (prepend t1 as the segment start).
    if assoc.any():
        ta = t[assoc]
        grid = np.concatenate(([t1], ta)) if ta[0] > t1 else ta
        seg = _integrate_segment(rhs, y_cur, grid, conc_M, args, use_scipy)
        seg = seg[-len(ta):]
        states[assoc] = seg
        y_cur = seg[-1].copy()
    if dissoc.any():
        td = t[dissoc]
        start = t2
        grid = np.concatenate(([start], td)) if td[0] > start else td
        seg = _integrate_segment(rhs, y_cur, grid, 0.0, args, use_scipy)
        seg = seg[-len(td):]
        states[dissoc] = seg

    result = {}
    if scheme in ("monovalent", "homobivalent"):
        result["A"] = states[:, 0]
        result["bound_complex"] = states[:, 1]
    else:
        for name, col in zip(["AB", "aAB", "ABb", "aABb", "apABbp"], range(5)):
            result[name] = states[:, col]
    bound_idx = info["bound_idx"]
    stoich = info["stoich"]
    bound = sum(s * states[:, j] for s, j in zip(stoich, bound_idx))
    occ = sum(states[:, j] for j in bound_idx) / target_tot
    result["bound"] = bound
    result["occupancy"] = occ
    return result


# --------------------------------------------------------------------------- #
# KineticModel wrapper for fitting a single trace
# --------------------------------------------------------------------------- #

class AvidityModel(KineticModel):
    """Fittable avidity model for one assoc/dissoc trace.

    Fixed at construction: scheme, phase boundaries (t1, t2), analyte
    concentration (conc_M), local concentration L, rebinding k, target total.
    Free parameters depend on scheme:
      - monovalent / homobivalent : R_max, k1, k_off1
      - heterobivalent            : R_max, k1, k_off1, k2, k_off2, f
        (set symmetric=True to share k2=k1, k_off2=k_off1 -> R_max, k1, k_off1, f)

    Response = R_max * occupancy(t).
    """

    def __init__(self, scheme: str, t1: float, t2: float, conc_M: float,
                 L: float = 1e-3, rebind_k: float = 0.0, target_tot: float = 1.0,
                 symmetric: bool = False, use_scipy: bool = True):
        self.scheme = scheme.lower()
        self.t1 = float(t1)
        self.t2 = float(t2)
        self.conc_M = float(conc_M)
        self.L = float(L)
        self.rebind_k = float(rebind_k)
        self.target_tot = float(target_tot)
        self.symmetric = symmetric
        self.use_scipy = use_scipy

    def __call__(self, t: np.ndarray, *params) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        if self.scheme in ("monovalent", "homobivalent"):
            R_max, k1, k_off1 = params[:3]
            sim = simulate_avidity(
                t, self.t1, self.t2, self.conc_M, self.scheme,
                k1=abs(k1), k_off1=abs(k_off1), rebind_k=self.rebind_k,
                target_tot=self.target_tot, use_scipy=self.use_scipy,
            )
        elif self.symmetric:
            R_max, k1, k_off1, f = params[:4]
            sim = simulate_avidity(
                t, self.t1, self.t2, self.conc_M, "heterobivalent",
                k1=abs(k1), k_off1=abs(k_off1), k2=abs(k1), k_off2=abs(k_off1),
                f=abs(f), L=self.L, target_tot=self.target_tot,
                use_scipy=self.use_scipy,
            )
        else:
            R_max, k1, k_off1, k2, k_off2, f = params[:6]
            sim = simulate_avidity(
                t, self.t1, self.t2, self.conc_M, "heterobivalent",
                k1=abs(k1), k_off1=abs(k_off1), k2=abs(k2), k_off2=abs(k_off2),
                f=abs(f), L=self.L, target_tot=self.target_tot,
                use_scipy=self.use_scipy,
            )
        return R_max * sim["occupancy"]

    def n_params(self) -> int:
        if self.scheme in ("monovalent", "homobivalent"):
            return 3
        return 4 if self.symmetric else 6

    def param_names(self) -> list:
        if self.scheme in ("monovalent", "homobivalent"):
            return ["R_max", "k1", "k_off1"]
        if self.symmetric:
            return ["R_max", "k1", "k_off1", "f"]
        return ["R_max", "k1", "k_off1", "k2", "k_off2", "f"]

    def avidity_metrics(self, *params) -> Dict[str, float]:
        """Functional affinity and residence-time style summaries.

        Reports per-arm K_D's and, for bivalent schemes, an avidity enhancement
        from the ring-closure term (L/f relative to k_off).
        """
        out: Dict[str, float] = {}
        if self.scheme in ("monovalent", "homobivalent"):
            _, k1, k_off1 = params[:3]
            out["K_D"] = abs(k_off1) / abs(k1) if k1 else np.nan
            return out
        if self.symmetric:
            _, k1, k_off1, f = params[:4]
            k2, k_off2 = k1, k_off1
        else:
            _, k1, k_off1, k2, k_off2, f = params[:6]
        out["K_D1"] = abs(k_off1) / abs(k1) if k1 else np.nan
        out["K_D2"] = abs(k_off2) / abs(k2) if k2 else np.nan
        # Ring-closure forward rates vs. off rates -> qualitative avidity factor.
        if f:
            out["ring_closure_a"] = (self.L * abs(k2) / abs(f))
            out["ring_closure_b"] = (self.L * abs(k1) / abs(f))
        return out
