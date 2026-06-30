"""Tests for new fitting models (Jarmoskaite, Tso, Vauquelin) + autotuning.

Pure-NumPy model math runs everywhere; fit tests require scipy.
"""

import numpy as np
import pytest

from banana.models.equilibrium import (
    HyperbolicBinding, QuadraticBinding, HillBinding,
    TwoSiteMicroscopic, TwoSiteMacroscopic,
    species_microscopic, species_macroscopic,
    kd_alpha_from_beta, beta_from_kd_alpha, equilibration_time,
)
from banana.models.avidity import simulate_avidity, AvidityModel
from banana.fitting.autotune import (
    piecewise_initial_guess, kobs_koff_from_titration,
)

C = np.array([1e-9, 3e-9, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6])
KD = 3e-8


# --------------------------- equilibrium models --------------------------- #

def test_hyperbolic_half_max():
    h = HyperbolicBinding()
    assert abs(h(np.array([KD]), 1.0, KD)[0] - 0.5) < 1e-9


def test_quadratic_reduces_to_hyperbolic():
    h = HyperbolicBinding()
    q = QuadraticBinding(R_tot=KD * 1e-6)
    assert np.allclose(q(C, 1.0, KD), h(C, 1.0, KD), atol=1e-4)


def test_quadratic_fraction_in_range():
    q = QuadraticBinding(R_tot=1e-7)
    fb = q(C, 1.0, KD)
    assert np.all(fb >= -1e-9) and np.all(fb <= 1 + 1e-9)


def test_microscopic_mass_balance():
    Btot, KD1, KD2 = 5e-8, 1e-8, 1e-6
    sp = species_microscopic(C, Btot, KD1, KD2)
    A_bal = sp["A_free"] + sp["AB"] + sp["BA"] + 2 * sp["ABA"]
    B_bal = sp["B"] + sp["AB"] + sp["BA"] + sp["ABA"]
    assert np.allclose(A_bal, C, rtol=1e-3)
    assert np.allclose(B_bal, Btot, rtol=1e-3)


def test_microscopic_no_depletion_limit():
    Btot, KD1, KD2 = 1e-12, 1e-8, 1e-6
    sp = species_microscopic(C, Btot, KD1, KD2)
    occ = (sp["AB"] + sp["BA"] + 2 * sp["ABA"]) / (2 * Btot)
    indep = 0.5 * (C / (C + KD1) + C / (C + KD2))
    assert np.allclose(occ, indep, atol=2e-3)


def test_macroscopic_mass_balance():
    Btot = 5e-8
    b1, b2 = beta_from_kd_alpha(3e-8, 1.0)
    sp = species_macroscopic(C, Btot, b1, b2)
    A_bal = sp["A_free"] + sp["AB"] + 2 * sp["ABA"]
    B_bal = sp["B"] + sp["AB"] + sp["ABA"]
    assert np.allclose(A_bal, C, rtol=1e-3)
    assert np.allclose(B_bal, Btot, rtol=1e-3)


def test_beta_kd_alpha_roundtrip():
    b1, b2 = beta_from_kd_alpha(3e-8, 2.0)
    kdm, alpha = kd_alpha_from_beta(b1, b2)
    assert abs(kdm - 3e-8) < 1e-15 and abs(alpha - 2.0) < 1e-9


def test_hill_reduces_to_hyperbolic_when_n1():
    h = HyperbolicBinding()
    hill = HillBinding()
    assert np.allclose(hill(C, 1.0, KD, 1.0), h(C, 1.0, KD))


def test_equilibration_time_positive():
    assert equilibration_time(1e5, 1e-3, 1e-8) > 0


# ------------------------------ avidity ODE ------------------------------ #

def _grid():
    return np.arange(0, 1200, 1.0), 100.0, 600.0


def test_monovalent_mass_conservation_and_kinetics():
    t, t1, t2 = _grid()
    k1, koff, conc = 1e5, 1e-2, 1e-7
    sim = simulate_avidity(t, t1, t2, conc, "monovalent",
                           k1=k1, k_off1=koff, target_tot=1.0, use_scipy=False)
    mass = sim["A"] + sim["bound_complex"]
    assert np.allclose(mass, 1.0, atol=1e-6)
    eq = conc / (conc + koff / k1)
    kobs = k1 * conc + koff
    ta = t[(t >= t1) & (t < t2)]
    oa = sim["occupancy"][(t >= t1) & (t < t2)]
    assert np.allclose(oa, eq * (1 - np.exp(-kobs * (ta - t1))), atol=1e-2)


def test_heterobivalent_conservation_and_avidity():
    t, t1, t2 = _grid()
    k1, koff, conc = 1e5, 1e-2, 1e-7
    base = simulate_avidity(t, t1, t2, conc, "heterobivalent", k1=k1, k_off1=koff,
                            k2=k1, k_off2=koff, f=10.0, L=1e-5, use_scipy=False)
    tot = sum(base[s] for s in ["AB", "aAB", "ABb", "aABb", "apABbp"])
    assert np.allclose(tot, 1.0, atol=1e-5)
    # Stronger ring closure -> more bound remains after washout.
    av = simulate_avidity(t, t1, t2, conc, "heterobivalent", k1=k1, k_off1=koff,
                          k2=k1, k_off2=koff, f=10.0, L=1e-4, use_scipy=False)
    i = np.argmax(t >= 1100)
    assert av["occupancy"][i] > base["occupancy"][i]


# ------------------------------ autotuning ------------------------------- #

def test_piecewise_initial_guess_recovers_params():
    A0, A1, tau1, A2, tau2 = 0.0, 1.0, 30.0, 0.05, 120.0
    t1, t2 = 60.0, 360.0
    t = np.arange(0, 900, 2.0)
    r = np.empty_like(t)
    for i, ti in enumerate(t):
        if ti < t1:
            r[i] = A0
        elif ti < t2:
            r[i] = A1 + (A0 - A1) * np.exp(-(ti - t1) / tau1)
        else:
            A1e = A1 + (A0 - A1) * np.exp(-(t2 - t1) / tau1)
            r[i] = A2 + (A1e - A2) * np.exp(-(ti - t2) / tau2)
    g = piecewise_initial_guess(t, r, A0=A0, t1=t1, t2=t2)
    assert abs(g.p0[0] - A1) < 0.05            # A1 plateau
    assert 0.5 * tau1 < g.p0[1] < 2 * tau1     # tau1 in right ballpark
    assert all(lo <= v <= hi for v, lo, hi in zip(g.p0, g.lower, g.upper))


def test_kobs_koff_from_titration_ballpark():
    k_on, k_off = 2e5, 1e-3
    concs = [6.25, 12.5, 25, 50, 100, 200]
    t1, t2 = 60.0, 360.0
    times, resps, t1l, t2l = [], [], [], []
    for c in concs:
        cM = c * 1e-9
        kobs = k_on * cM + k_off
        Req = cM / ((k_off / k_on) + cM)
        tt = np.arange(0, 900, 2.0)
        rr = np.empty_like(tt)
        for i, ti in enumerate(tt):
            if ti < t1:
                rr[i] = 0
            elif ti < t2:
                rr[i] = Req * (1 - np.exp(-kobs * (ti - t1)))
            else:
                Re = Req * (1 - np.exp(-kobs * (t2 - t1)))
                rr[i] = Re * np.exp(-k_off * (ti - t2))
        times.append(tt); resps.append(rr); t1l.append(t1); t2l.append(t2)
    kon_e, koff_e, KD_e = kobs_koff_from_titration(times, resps, concs, t1l, t2l)
    # Initial-guess accuracy: within an order of magnitude is sufficient.
    assert 0.2 * k_on < kon_e < 5 * k_on


# --------------------------- scipy-backed fits --------------------------- #

def test_fit_equilibrium_recovers_KD():
    pytest.importorskip("scipy")
    from banana.fitting import fit_equilibrium
    true_KD, true_Rmax = 4e-8, 1.2
    h = HyperbolicBinding()
    signal = h(C, true_Rmax, true_KD) + np.random.default_rng(0).normal(0, 1e-3, C.shape)
    res = fit_equilibrium(C, signal, model="hyperbolic")
    assert res.success
    d = res.to_dict()
    assert abs(d["K_D"] - true_KD) / true_KD < 0.1
    assert abs(d["R_max"] - true_Rmax) / true_Rmax < 0.1


def test_avidity_model_callable_with_scipy():
    pytest.importorskip("scipy")
    t, t1, t2 = _grid()
    m = AvidityModel("heterobivalent", t1, t2, conc_M=1e-7, L=1e-5, symmetric=True)
    y = m(t, 1.0, 1e5, 1e-2, 10.0)
    assert y.shape == t.shape and np.all(np.isfinite(y))
