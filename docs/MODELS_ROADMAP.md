# Implementation roadmap: new fitting models + autotuning

> **Status (implemented):** autotuning (`fitting/autotune.py`), equilibrium models
> + steady-state extraction (`models/equilibrium.py`, `fitting/equilibrium_fit.py`),
> and the Vauquelin avidity ODE (`models/avidity.py`) are all built, registered,
> and unit-tested (`tests/test_new_models.py`). Model math verified to machine
> precision (mass balances, limiting cases, analytic kinetics). Run
> `pytest tests/` in the scipy-enabled `banana` env for the full suite incl. fits.


Prepared from the three papers in `literature/`. Scope confirmed with author:
no priority preference between model families; autotuning = **data-driven initial
guesses + bounds** (single optimization run, no multi-start for now).

## Current architecture (extension points)

- `models/kinetics.py` — `KineticModel` ABC (`__call__`, `n_params`, `param_names`).
  Existing: `OneToOneBinding`, `PiecewiseExponential`. **All current models are
  time-domain** (fit response vs. time per trace).
- `fitting/fit.py` — `fit_single_trace` (`curve_fit`), `fit_titration_global`
  (branches on `model_type`). Initial guesses are hardcoded
  (`p0=[0.05,10,0.04,10]`), bounds `(-inf, inf)`.
- `cli/run_bli.py` — selects model via `kinetic_model.type` in YAML.
- Register new models in `models/__init__.py` and `banana/__init__.py`.

The papers fall into **two families** that plug in differently.

---

## Family A — Equilibrium (steady-state) binding models

These fit a **signal-vs-concentration** curve, not a time trace. The package has
no such axis yet, so they share one new piece of machinery.

### A0. Shared prerequisite: steady-state response extraction
New module `models/equilibrium.py` + helper to reduce each `TitrationSeries`
trace to an equilibrium observable `R_eq(conc)`:
- For each trace, take the mean response over the last N points of the
  association phase (plateau), or the fitted `R_eq` from the kinetic fit.
- Produce arrays `(concentration_M, R_eq)` to feed equilibrium fitters.
- Add `EquilibriumModel` base (or reuse `KineticModel` with an `x = concentration`
  convention) and a `fit_equilibrium(conc, signal, model, ...)` routine.

### A1. Jarmoskaite & Herschlag 2020 — hyperbolic + quadratic
- **Hyperbolic / Langmuir (Eq 4b):**
  `f_bound = [P]_tot / ([P]_tot + K_D)`  → signal form `R = R_max·[P]/([P]+K_D)`.
- **Quadratic / depletion-aware (Eq 5):**
  `f_bound = ((R_t+P_t+K_D) − sqrt((R_t+P_t+K_D)² − 4·R_t·P_t)) / (2·R_t)`
  Params: `K_D`, plus `R_max` (signal scale) and optionally active-fraction
  coefficient. Use when limiting-species conc is not << K_D.
- Also expose the equilibration-time check: `k_equil = k_on·[P] + k_off`,
  `t_equil ≈ 5·ln2/k_equil` (QC warning, not a fit).

### A2. Tso et al. 2018 — two-site (1:2) MST models
All solve for free ligand `A` from a cubic `A³ + pA² + qA + r = 0` via the
trig root (Eqs 9–10), then compute species fractions and a linear signal
combination. Implement a shared cubic root solver.
- **1:2 microscopic** — two micro constants `K_D(1)`, `K_D(2)` (Eqs 7–8, 11–16).
  Signal = Σ species·F_species.
- **1:2 macroscopic** — overall β1, β2 (Eqs 17–22); report `K_D,M = 1/β1` and
  cooperativity `α = β1²/(4β2)`. Constraint `K_D,M2 = 4α·K_D,M1`. Handle the
  positive-cooperativity imaginary-root branch (alt root-finding).
- **Hill (Eq 23):** `F = F_B* + F_AB*A·Aⁿ/(K50ⁿ + Aⁿ)`.

> Note: Tso models are written for MST (thermophoresis `Fn`). For BLI/SPR reuse
> the same species math; map `F_species` to per-species response weights.

---

## Family B — Kinetic avidity / bivalent model (time-domain)

### B1. Vauquelin 2013 — bivalent / heterobivalent two-step binding
ODE-based, fits the association+dissociation **trace** like existing kinetic
models. Equations now in hand from **Appendix S1** (`literature/bph12106-sup-0001-s/`,
extracted Tables A1 & A2). Integrate with `scipy.integrate.solve_ivp`; fit
response = weighted sum of bound species. Ligand assumed in large excess (free
ligand conc held constant). Implement the three Table A1 schemes:

**(1) Monovalent a→A** (Eqs 1–2):
```
d[aA]/dt = k1·[a]·[A] − k-1·[aA]
d[A]/dt  = k-1·[aA] − k1·[a]·[A]
```

**(2) Homobivalent aa→A**, isolated site, forward ×2 (Eqs 3–4):
```
d[aaA]/dt = 2·k1·[aa]·[A] − k-1·[aaA]
d[A]/dt   = k-1·[aaA] − 2·k1·[aa]·[A]
```

**(3) Heterobivalent ab→AB target-pair** — the avidity model (Eqs 5–9).
Species: `AB`, `aAB`, `ABb`, `aABb`, `a'ABb'`. `[L]` = local conc of the free
pharmacophore once one arm is bound (ring-closure driver), `f` = penalty factor:
```
d[AB]/dt    = k-1·[aAB] + k-2·[ABb] − (k1+k2)·[AB]·[ab]
d[aAB]/dt   = k1·[AB]·[ab] + k-2·([aABb]+[a'ABb']) − k-1·[aAB]
              − ([L]·k2/f)·[aAB] − k2·[aAB]·[ab]
d[ABb]/dt   = k2·[AB]·[ab] + k-1·([aABb]+[a'ABb']) − k-2·[ABb]
              − ([L]·k1/f)·[ABb] − k1·[ABb]·[ab]
d[aABb]/dt  = ([L]·k1/f)·[ABb] + ([L]·k2/f)·[aAB] − (k-1+k-2)·[aABb]
d[a'ABb']/dt= k1·[ABb]·[ab] + k2·[aAB]·[ab] − (k-1+k-2)·[a'ABb']
```
Occupancy (Eq 10): `[AB]occ = 100·(aAB+aABb+ABb+a'ABb') / (AB+aAB+aABb+ABb+a'ABb')`.

Optional **hindered-diffusion rebinding** variant: replace `k1→k1/(1+k1·[A]·k)`
and `k-1→k-1/(1+k1·[A]·k)` (×2 in the homobivalent case). `k` is a rebinding
parameter. Make this a toggle.

**Table A2** extends every scheme with a competitive ligand `c` (k3/k-3, species
`cA`, `cAB`, `cABb`). Defer to a later phase — adds a competition flag only if needed.

For BLI of a bivalent binder to immobilized antigen, **scheme (3)** is the avidity
case of interest; scheme (2) is the no-crosslink baseline. Fit params: `k1, k-1,
k2, k-2, f` (and `[L]`, `k` as fixed/optional), plus response scale per species.

---

## Autotuning (data-driven p0 + bounds)

New `fitting/autotune.py`. For a given model + trace(s), derive starting values
and bounds from the data instead of hardcoded constants:
- `R_max` ← association-phase plateau (max/last-N mean).
- `k_off` ← slope of `log(response)` over the dissociation tail.
- `k_obs` ← association curvature (1/τ from early-phase fit); `k_on ≈
  (k_obs − k_off)/[conc]`.
- `K_D` ← concentration at half-maximal `R_eq` (titration midpoint).
- Bounds: physically constrained (rates > 0, R_max in [0, ~1.5·max signal],
  K_D within the titrated concentration span ± decade).
- Wire into `fit_single_trace` / `fit_titration_global` so each `model_type`
  pulls its own autotuned `p0`/`bounds`; keep current hardcoded values as
  fallback.

---

## Suggested build order (no hard dependency on author's preference)

1. `autotune.py` + retrofit existing piecewise/1:1 fits (immediate robustness win,
   no new model risk).
2. A0 steady-state extraction + A1 (hyperbolic, quadratic) — smallest new family.
3. A2 Tso two-site models (reuse A0 machinery + cubic solver).
4. B1 Vauquelin bivalent ODE (most complex; needs Appendix S1 verification).

Each step: add model/routine → register in `__init__`s → add `kinetic_model.type`
(or `equilibrium_model.type`) config option → unit test against a known curve →
run on `example_data/20260311 Nb6` and eyeball the fit PDF.
