"""Equilibrium (steady-state) binding models: signal vs. concentration.

These differ from the time-domain kinetic models in `kinetics.py`: the
independent variable is **total titrant concentration** (molar), and the
dependent variable is a steady-state observable (e.g. BLI plateau response,
SPR R_eq, or MST thermophoresis).

Models implemented
------------------
- HyperbolicBinding   : 1:1 Langmuir, Jarmoskaite & Herschlag 2020 Eq. 4b.
- QuadraticBinding    : depletion-aware 1:1, Jarmoskaite 2020 Eq. 5.
- TwoSiteMicroscopic  : 1:2 with two micro K_D, Tso et al. 2018 (Eqs. 7-16).
- TwoSiteMacroscopic  : 1:2 symmetric, K_D,M + cooperativity alpha (Eqs. 17-22).
- HillBinding         : Hill equation, Tso 2018 Eq. 23.

Concentrations are in molar (M) throughout unless a function says otherwise.
All model evaluation is pure NumPy; fitting lives in `banana.fitting`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class EquilibriumModel(ABC):
    """Base class: signal as a function of total titrant concentration (M)."""

    @abstractmethod
    def __call__(self, conc: np.ndarray, *params) -> np.ndarray:
        ...

    @abstractmethod
    def n_params(self) -> int:
        ...

    @abstractmethod
    def param_names(self) -> list:
        ...


# --------------------------------------------------------------------------- #
# Jarmoskaite & Herschlag 2020
# --------------------------------------------------------------------------- #

class HyperbolicBinding(EquilibriumModel):
    """1:1 Langmuir binding (binding regime), Eq. 4b.

        signal = baseline + R_max * C / (C + K_D)

    Assumes the limiting species concentration << K_D (no depletion). Use when
    the titrant is in large excess across the curve.
    Parameters: R_max, K_D [, baseline].
    """

    def __init__(self, fit_baseline: bool = False):
        self.fit_baseline = fit_baseline

    def __call__(self, conc: np.ndarray, *params) -> np.ndarray:
        c = np.asarray(conc, dtype=float)
        R_max, K_D = params[0], params[1]
        base = params[2] if self.fit_baseline else 0.0
        K_D = abs(K_D)
        return base + R_max * c / (c + K_D)

    def n_params(self) -> int:
        return 3 if self.fit_baseline else 2

    def param_names(self) -> list:
        names = ["R_max", "K_D"]
        if self.fit_baseline:
            names.append("baseline")
        return names


class QuadraticBinding(EquilibriumModel):
    """Depletion-aware 1:1 binding (intermediate regime), Eq. 5.

        f_bound = ((R_t + C + K_D) - sqrt((R_t + C + K_D)^2 - 4 R_t C)) / (2 R_t)
        signal  = baseline + R_max * f_bound

    where C is total titrant concentration and R_t is the total concentration of
    the limiting (labeled/immobilized) species. R_t is a known constant by
    default; set fit_Rtot=True to refine it (e.g. active-fraction estimation).
    Parameters: R_max, K_D [, R_t] [, baseline].
    """

    def __init__(self, R_tot: float = 1e-9, fit_Rtot: bool = False,
                 fit_baseline: bool = False):
        self.R_tot = float(R_tot)
        self.fit_Rtot = fit_Rtot
        self.fit_baseline = fit_baseline

    def __call__(self, conc: np.ndarray, *params) -> np.ndarray:
        c = np.asarray(conc, dtype=float)
        R_max, K_D = params[0], abs(params[1])
        idx = 2
        R_t = abs(params[idx]) if self.fit_Rtot else self.R_tot
        if self.fit_Rtot:
            idx += 1
        base = params[idx] if self.fit_baseline else 0.0
        R_t = max(R_t, 1e-30)
        s = c + R_t + K_D
        disc = np.maximum(s * s - 4.0 * R_t * c, 0.0)
        f_bound = (s - np.sqrt(disc)) / (2.0 * R_t)
        return base + R_max * f_bound

    def n_params(self) -> int:
        return 2 + int(self.fit_Rtot) + int(self.fit_baseline)

    def param_names(self) -> list:
        names = ["R_max", "K_D"]
        if self.fit_Rtot:
            names.append("R_tot")
        if self.fit_baseline:
            names.append("baseline")
        return names


# --------------------------------------------------------------------------- #
# Two-site cubic free-ligand solvers (Tso et al. 2018)
# --------------------------------------------------------------------------- #

def _cubic_physical_root(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Physically realistic root of A^3 + p A^2 + q A + r = 0 (Tso Eqs. 9-10).

    Uses the trigonometric form. Falls back to numpy root-finding when the
    discriminant goes negative (e.g. strong positive cooperativity), selecting
    the smallest non-negative real root.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    r = np.asarray(r, dtype=float)
    out = np.empty(np.broadcast(p, q, r).shape, dtype=float)
    pf, qf, rf = np.broadcast_arrays(p, q, r)
    it = np.nditer(out, flags=["multi_index"], op_flags=["writeonly"])
    while not it.finished:
        i = it.multi_index
        pp, qq, rr = float(pf[i]), float(qf[i]), float(rf[i])
        base = pp * pp - 3.0 * qq
        val = np.nan
        if base > 0:
            arg = (-2.0 * pp**3 + 9.0 * pp * qq - 27.0 * rr) / (2.0 * np.sqrt(base**3))
            arg = min(1.0, max(-1.0, arg))
            theta = np.arccos(arg)
            cand = -pp / 3.0 + (2.0 / 3.0) * np.sqrt(base) * np.cos(theta / 3.0)
            if np.isfinite(cand) and cand >= -1e-18:
                val = max(cand, 0.0)
        if not np.isfinite(val):
            roots = np.roots([1.0, pp, qq, rr])
            real = roots[np.abs(roots.imag) < 1e-9].real
            real = real[real >= -1e-18]
            val = float(np.min(real)) if real.size else 0.0
            val = max(val, 0.0)
        it[0][...] = val
        it.iternext()
    return out


def solve_free_A_microscopic(A_tot, B_tot, K_D1, K_D2):
    """Free ligand A given totals and micro K_D's (Tso Eqs. 7-10)."""
    A_tot = np.asarray(A_tot, dtype=float)
    p = K_D1 + K_D2 + 2.0 * B_tot - A_tot
    q = K_D1 * K_D2 + (B_tot - A_tot) * (K_D1 + K_D2)
    r = -K_D1 * K_D2 * A_tot
    return _cubic_physical_root(p, q, r)


def species_microscopic(A_tot, B_tot, K_D1, K_D2):
    """Concentrations of B*, AB*, B*A, AB*A (Tso Eqs. 11-14).

    Returns dict of arrays. AB* and B*A are returned separately; AB*A uses
    Eq. 12. Robust to A->0.
    """
    A_tot = np.asarray(A_tot, dtype=float)
    A = solve_free_A_microscopic(A_tot, B_tot, K_D1, K_D2)
    denom = (A * A) / (K_D1 * K_D2) - 1.0
    # B* from Eq. 14; guard the near-singular denominator.
    with np.errstate(divide="ignore", invalid="ignore"):
        Bstar = (A_tot - B_tot - A) / denom
    # When denom ~ 0 (A^2 ~ KD1 KD2), fall back via mass balance later.
    Bstar = np.where(np.isfinite(Bstar), Bstar, 0.0)
    Bstar = np.clip(Bstar, 0.0, B_tot)
    ABA = (A * A) * Bstar / (K_D1 * K_D2)        # Eq. 12
    AB = A * Bstar / K_D1                          # site 1 singly bound
    BA = A * Bstar / K_D2                          # site 2 singly bound
    return {"B": Bstar, "AB": AB, "BA": BA, "ABA": ABA, "A_free": A}


def solve_free_A_macroscopic(A_tot, B_tot, beta1, beta2):
    """Free ligand A for the macroscopic model (Tso Eqs. 18-19)."""
    A_tot = np.asarray(A_tot, dtype=float)
    p = (beta1 + beta2 * (2.0 * B_tot - A_tot)) / beta2
    q = (1.0 + beta1 * (B_tot - A_tot)) / beta2
    r = -A_tot / beta2
    return _cubic_physical_root(p, q, r)


def species_macroscopic(A_tot, B_tot, beta1, beta2):
    """Concentrations of B*, AB*, AB*A for the macroscopic model (Eqs. 17, 20)."""
    A_tot = np.asarray(A_tot, dtype=float)
    A = solve_free_A_macroscopic(A_tot, B_tot, beta1, beta2)
    with np.errstate(divide="ignore", invalid="ignore"):
        Bstar = (A_tot - A) / (beta1 * A + 2.0 * beta2 * A * A)   # Eq. 20
    Bstar = np.where(np.isfinite(Bstar), Bstar, 0.0)
    Bstar = np.clip(Bstar, 0.0, B_tot)
    AB = beta1 * A * Bstar                # Eq. 17
    ABA = beta2 * A * A * Bstar           # Eq. 17
    return {"B": Bstar, "AB": AB, "ABA": ABA, "A_free": A}


def kd_alpha_from_beta(beta1, beta2):
    """Macroscopic K_D,M and cooperativity alpha (Tso Eq. 22).

    alpha > 1 negative cooperativity; alpha < 1 positive cooperativity.
    """
    K_DM = 1.0 / beta1 if beta1 > 0 else np.inf
    alpha = (beta1 * beta1) / (4.0 * beta2) if beta2 > 0 else np.nan
    return K_DM, alpha


def beta_from_kd_alpha(K_DM, alpha):
    """Inverse of `kd_alpha_from_beta`: beta1, beta2 from K_D,M and alpha."""
    beta1 = 1.0 / K_DM
    beta2 = (beta1 * beta1) / (4.0 * alpha)
    return beta1, beta2


class TwoSiteMicroscopic(EquilibriumModel):
    """1:2 microscopic binding, two distinct micro K_D's (Tso 2018).

    Signal is a weighted sum of labeled-species concentrations. For BLI/SPR the
    natural weighting is bound-analyte mass: w_AB = w_BA = 1, w_ABA = 2. The
    default reproduces site occupancy scaled by R_max.

        signal = baseline + R_max * (AB + BA + 2 AB*A) / (2 B_tot)

    B_tot (receptor/surface total, M) is a fixed constant; in the BLI
    no-depletion limit (B_tot << K_D) this reduces to two independent sites.
    Parameters: R_max, K_D1, K_D2 [, baseline].
    """

    def __init__(self, B_tot: float = 1e-12, fit_baseline: bool = False):
        self.B_tot = float(B_tot)
        self.fit_baseline = fit_baseline

    def __call__(self, conc: np.ndarray, *params) -> np.ndarray:
        c = np.asarray(conc, dtype=float)
        R_max, K_D1, K_D2 = params[0], abs(params[1]), abs(params[2])
        base = params[3] if self.fit_baseline else 0.0
        sp = species_microscopic(c, self.B_tot, K_D1, K_D2)
        occ = (sp["AB"] + sp["BA"] + 2.0 * sp["ABA"]) / (2.0 * self.B_tot)
        return base + R_max * occ

    def n_params(self) -> int:
        return 4 if self.fit_baseline else 3

    def param_names(self) -> list:
        names = ["R_max", "K_D1", "K_D2"]
        if self.fit_baseline:
            names.append("baseline")
        return names


class TwoSiteMacroscopic(EquilibriumModel):
    """1:2 symmetric (macroscopic) binding with cooperativity (Tso 2018).

    Refines K_D,M and cooperativity alpha (converted to/from beta1, beta2).

        signal = baseline + R_max * (AB + 2 AB*A) / (2 B_tot)

    Parameters: R_max, K_D,M, alpha [, baseline].
    """

    def __init__(self, B_tot: float = 1e-12, fit_baseline: bool = False):
        self.B_tot = float(B_tot)
        self.fit_baseline = fit_baseline

    def __call__(self, conc: np.ndarray, *params) -> np.ndarray:
        c = np.asarray(conc, dtype=float)
        R_max, K_DM, alpha = params[0], abs(params[1]), abs(params[2])
        base = params[3] if self.fit_baseline else 0.0
        beta1, beta2 = beta_from_kd_alpha(K_DM, alpha)
        sp = species_macroscopic(c, self.B_tot, beta1, beta2)
        occ = (sp["AB"] + 2.0 * sp["ABA"]) / (2.0 * self.B_tot)
        return base + R_max * occ

    def n_params(self) -> int:
        return 4 if self.fit_baseline else 3

    def param_names(self) -> list:
        names = ["R_max", "K_D_M", "alpha"]
        if self.fit_baseline:
            names.append("baseline")
        return names


class HillBinding(EquilibriumModel):
    """Hill equation (Tso 2018 Eq. 23).

        signal = baseline + R_max * C^n / (K50^n + C^n)

    Parameters: R_max, K50, n [, baseline].
    """

    def __init__(self, fit_baseline: bool = False):
        self.fit_baseline = fit_baseline

    def __call__(self, conc: np.ndarray, *params) -> np.ndarray:
        c = np.asarray(conc, dtype=float)
        R_max, K50, n = params[0], abs(params[1]), abs(params[2])
        base = params[3] if self.fit_baseline else 0.0
        cn = np.power(c, n)
        return base + R_max * cn / (np.power(K50, n) + cn)

    def n_params(self) -> int:
        return 4 if self.fit_baseline else 3

    def param_names(self) -> list:
        names = ["R_max", "K50", "n_Hill"]
        if self.fit_baseline:
            names.append("baseline")
        return names


# --------------------------------------------------------------------------- #
# QC: equilibration time (Jarmoskaite Eqs. 1-2)
# --------------------------------------------------------------------------- #

def equilibration_time(k_on: float, k_off: float, conc_M: float,
                       n_halflives: float = 5.0) -> float:
    """Time to reach equilibrium: t = n * ln2 / (k_on*[C] + k_off).

    Defaults to 5 half-lives (~96.6% equilibration), the paper's recommendation.
    """
    k_equil = k_on * conc_M + k_off
    if k_equil <= 0:
        return np.inf
    return n_halflives * np.log(2.0) / k_equil
