#!/usr/bin/env python3
"""Compare data-driven autotuning against the old hardcoded initial guesses.

Testing harness (not part of the package). Runs the piecewise and 1:1 global
fits on real Octet titration data TWO ways, isolating the effect of the
autotune p0/bounds retrofit:

  * LEGACY   -- the pre-autotune behavior from the initial commit:
                  piecewise per-trace:  p0 = [0.05, 10, 0.04, 10], bounds = (-inf, inf)
                  1:1 global:           x0 = [1e4, 1e-3, max(r)...], bounds = (0, inf)
  * AUTOTUNE  -- the current behavior: data-driven p0 + physical bounds from
                  banana.fitting.autotune (piecewise_initial_guess /
                  one_to_one_initial_guess).

For each dataset it reports, per fit type:
  - convergence / success rate
  - total and median chi^2 (sum of squared residuals)
  - how many traces pass the k_obs > k_off gate that feeds the global regression
  - the resulting k_on, k_off, K_D and the R^2 of the k_obs~[conc] line (piecewise)
  - side-by-side against the Octet instrument's own per-trace fits, when present.

Usage (from the repo root, inside the conda `banana` env):

    python autotune_vs_legacy.py
    python autotune_vs_legacy.py --data "example_data/20260311 Nb6"
    python autotune_vs_legacy.py --csv autotune_compare   # also write per-trace CSVs

Everything is read-only; nothing in the package is modified.
"""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent  # scripts/ -> repo root
SRC = REPO / "src"
if SRC.is_dir():
    sys.path.insert(0, str(SRC))

from banana.config.load_config import load_config  # noqa: E402
from banana.io.bli_experiment import load_bli_experiment_from_config  # noqa: E402
from banana.io.titration import association_dissociation_start_times  # noqa: E402
from banana.models.kinetics import PiecewiseExponential, OneToOneBinding  # noqa: E402
from banana.fitting.fit import fit_single_trace  # noqa: E402
from banana.fitting.autotune import (  # noqa: E402
    InitialGuess,
    piecewise_initial_guess,
    one_to_one_initial_guess,
)
from scipy.optimize import least_squares  # noqa: E402

CONFIG = REPO / "src" / "banana" / "configs" / "bli" / \
    "octet_384_16sensor_titration_concentration_matched.yaml"

LEGACY_PW_P0 = [0.05, 10.0, 0.04, 10.0]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _trace_setup(titration, i):
    """Replicate fit_titration_global's per-trace preamble (A0, t1, t2)."""
    t, r = titration.get_trace(i)
    A0 = float(np.mean(r[: min(50, max(1, len(r) // 10))]))
    am = (titration.assoc_mask[i]
          if titration.assoc_mask and i < len(titration.assoc_mask) else None)
    dm = (titration.dissoc_mask[i]
          if titration.dissoc_mask and i < len(titration.dissoc_mask) else None)
    if (titration.association_start_t is not None
            and titration.dissociation_start_t is not None
            and i < len(titration.association_start_t)
            and i < len(titration.dissociation_start_t)):
        t1 = float(titration.association_start_t[i])
        t2 = float(titration.dissociation_start_t[i])
    else:
        t1, t2 = association_dissociation_start_times(t, am, dm)
    return t, r, A0, t1, t2


def _global_linear(conc_M, k_obs):
    """k_obs = k_on*[conc] + k_off via lstsq; returns k_on, k_off, K_D, R^2."""
    conc_M = np.asarray(conc_M, float)
    k_obs = np.asarray(k_obs, float)
    A = np.column_stack([conc_M, np.ones_like(conc_M)])
    x, *_ = np.linalg.lstsq(A, k_obs, rcond=None)
    k_on, k_off = float(x[0]), float(x[1])
    pred = k_on * conc_M + k_off
    ss_res = float(np.sum((k_obs - pred) ** 2))
    ss_tot = float(np.sum((k_obs - np.mean(k_obs)) ** 2)) or np.nan
    r2 = 1.0 - ss_res / ss_tot if ss_tot and np.isfinite(ss_tot) else np.nan
    K_D = k_off / k_on if k_on > 0 else np.nan
    return k_on, k_off, K_D, r2


def piecewise_run(titration, mode):
    """Run piecewise per-trace fits. mode in {'legacy','autotune'}.

    Returns (per_trace rows, global dict).
    """
    rows = []
    conc_M, k_obs_list = [], []
    for i in range(len(titration)):
        t, r, A0, t1, t2 = _trace_setup(titration, i)
        conc = titration.concentration[i]
        label = (titration.labels[i]
                 if titration.labels and i < len(titration.labels) else f"#{i}")
        if len(t) < 10:
            rows.append(dict(label=label, conc_nM=conc, ok=False, chi2=np.nan,
                             tau1=np.nan, tau2=np.nan, gated=False, note="too few pts"))
            continue
        model = PiecewiseExponential(A0=A0, t1=t1, t2=t2)
        if mode == "legacy":
            res = fit_single_trace(t, r, model, p0=list(LEGACY_PW_P0))  # bounds default -inf/inf
        else:
            g = piecewise_initial_guess(t, r, A0=A0, t1=t1, t2=t2)
            init = InitialGuess(p0=list(g.p0), lower=g.lower, upper=g.upper,
                                param_names=g.param_names).clipped()
            res = fit_single_trace(t, r, model, p0=init.p0, bounds=init.as_bounds())

        row = dict(label=label, conc_nM=conc, ok=bool(res.success),
                   chi2=(res.chi2 if res.chi2 is not None else np.nan),
                   tau1=np.nan, tau2=np.nan, gated=False, note=res.message[:40])
        if res.success and len(res.params) >= 4:
            tau1, tau2 = float(res.params[1]), float(res.params[3])
            row["tau1"], row["tau2"] = tau1, tau2
            k_obs = 1.0 / max(abs(tau1), 1e-12)
            k_off = 1.0 / max(abs(tau2), 1e-12)
            if conc > 0 and k_obs > k_off:
                row["gated"] = True
                conc_M.append(conc * 1e-9)
                k_obs_list.append(k_obs)
        rows.append(row)

    g = dict(n=len(rows),
             n_ok=sum(1 for x in rows if x["ok"]),
             n_gated=len(conc_M),
             chi2_total=float(np.nansum([x["chi2"] for x in rows])),
             chi2_median=float(np.nanmedian([x["chi2"] for x in rows
                                             if np.isfinite(x["chi2"])] or [np.nan])),
             k_on=np.nan, k_off=np.nan, K_D=np.nan, r2=np.nan)
    if len(conc_M) >= 2:
        g["k_on"], g["k_off"], g["K_D"], g["r2"] = _global_linear(conc_M, k_obs_list)
    return rows, g


def one_to_one_run(titration, mode):
    """Global 1:1 association fit. mode in {'legacy','autotune'}."""
    n = len(titration)

    def residuals(params):
        k_on, k_off = params[0], params[1]
        R_max = params[2:]
        out = []
        for j in range(n):
            t, r = titration.get_trace(j)
            conc = titration.concentration[j] * 1e-9
            R_m = R_max[j] if j < len(R_max) else R_max[-1]
            pred = OneToOneBinding(phase="association").association(
                t, k_on, k_off, R_m, conc * 1e9)
            out.extend((r - pred).ravel())
        return np.array(out)

    if mode == "legacy":
        R_max_0 = [float(np.max(titration.response[j])) for j in range(n)]
        x0 = [1e4, 1e-3] + R_max_0
        bounds = (0, np.inf)
    else:
        times = [titration.get_trace(j)[0] for j in range(n)]
        resps = [titration.get_trace(j)[1] for j in range(n)]
        g = one_to_one_initial_guess(
            times, resps, list(titration.concentration),
            t1_list=titration.association_start_t,
            t2_list=titration.dissociation_start_t)
        g = InitialGuess(p0=list(g.p0), lower=g.lower, upper=g.upper,
                         param_names=g.param_names).clipped()
        x0 = g.p0
        bounds = tuple(g.as_bounds())

    try:
        ls = least_squares(residuals, x0, bounds=bounds)
        k_on, k_off = float(ls.x[0]), float(ls.x[1])
        return dict(ok=bool(ls.success), k_on=k_on, k_off=k_off,
                    K_D=(k_off / k_on if k_on > 0 else np.nan),
                    chi2=float(np.sum(ls.fun ** 2)), nfev=int(ls.nfev),
                    msg=str(ls.message)[:50])
    except Exception as e:  # noqa: BLE001
        return dict(ok=False, k_on=np.nan, k_off=np.nan, K_D=np.nan,
                    chi2=np.nan, nfev=0, msg=str(e)[:50])


def load_instrument_reference(data_dir: Path):
    """Instrument KD/kon/kdis per trace, if the Results CSV is present."""
    csv_path = data_dir / "Results" / "kineticanalysistableresults.csv"
    if not csv_path.is_file():
        return None
    out = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            out.append(dict(sample=row.get("Sample ID", ""),
                            conc=row.get("Conc. (nM)", ""),
                            KD=row.get("KD (M)", ""),
                            kon=row.get("kon(1/Ms)", ""),
                            kdis=row.get("kdis(1/s)", ""),
                            r2=row.get("Full R^2", "")))
    return out


def _fmt(x, sci=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "  --  "
    return f"{x:.3e}" if sci else f"{x:.4g}"


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def analyze(data_dir: Path, cfg, write_csv_prefix=None):
    print("=" * 78)
    print(f"DATASET: {data_dir.name}")
    print("=" * 78)
    titration, _ = load_bli_experiment_from_config(data_dir, config_dict=cfg)
    print(f"traces loaded: {len(titration)}   "
          f"conc (nM): {', '.join(f'{c:g}' for c in titration.concentration)}")

    # ---- piecewise ----
    leg_rows, leg_g = piecewise_run(titration, "legacy")
    aut_rows, aut_g = piecewise_run(titration, "autotune")

    print("\n-- PIECEWISE per-trace (chi^2 = sum sq resid; lower is better) --")
    print(f"{'trace':<14}{'conc(nM)':>10}{'legacy chi2':>14}{'auto chi2':>14}"
          f"{'leg ok':>8}{'auto ok':>9}{'leg gate':>10}{'auto gate':>10}")
    for lr, ar in zip(leg_rows, aut_rows):
        print(f"{str(lr['label']):<14}{lr['conc_nM']:>10.4g}"
              f"{_fmt(lr['chi2']):>14}{_fmt(ar['chi2']):>14}"
              f"{str(lr['ok']):>8}{str(ar['ok']):>9}"
              f"{str(lr['gated']):>10}{str(ar['gated']):>10}")

    def summarize(tag, g):
        print(f"  {tag:<9} converged {g['n_ok']}/{g['n']} | "
              f"gated {g['n_gated']}/{g['n']} | "
              f"chi2 total {_fmt(g['chi2_total'])} median {_fmt(g['chi2_median'])}")
        print(f"  {'':<9} GLOBAL  k_on={_fmt(g['k_on'], 1)} 1/Ms  "
              f"k_off={_fmt(g['k_off'], 1)} 1/s  "
              f"K_D={_fmt(g['K_D'], 1)} M  (k_obs~conc R^2={_fmt(g['r2'])})")

    print("\n-- PIECEWISE global summary --")
    summarize("LEGACY", leg_g)
    summarize("AUTOTUNE", aut_g)

    # ---- 1:1 ----
    print("\n-- 1:1 GLOBAL association fit --")
    leg1 = one_to_one_run(titration, "legacy")
    aut1 = one_to_one_run(titration, "autotune")
    for tag, r in (("LEGACY", leg1), ("AUTOTUNE", aut1)):
        print(f"  {tag:<9} ok={str(r['ok']):<5} "
              f"k_on={_fmt(r['k_on'], 1)} 1/Ms  k_off={_fmt(r['k_off'], 1)} 1/s  "
              f"K_D={_fmt(r['K_D'], 1)} M  chi2={_fmt(r['chi2'])}  nfev={r['nfev']}")

    # ---- instrument reference ----
    ref = load_instrument_reference(data_dir)
    if ref:
        def _num(s):
            s = (s or "").strip()
            if not s or s.startswith("<"):
                return None
            try:
                return float(s)
            except ValueError:
                return None
        kds = [v for r in ref if (v := _num(r["KD"])) is not None]
        print("\n-- Octet instrument per-trace reference (its own 1:1 fits) --")
        for r in ref:
            print(f"  {r['sample']:<18} conc={r['conc']:>7} nM  "
                  f"KD={r['KD']:>10}  kon={r['kon']:>9}  kdis={r['kdis']:>10}  "
                  f"R^2={r['r2']}")
        if kds:
            vals = np.array(kds)
            print(f"  instrument KD range: {vals.min():.2e} - {vals.max():.2e} M "
                  f"(median {np.median(vals):.2e} M)  [{len(kds)}/{len(ref)} traces reported]")
        else:
            print("  instrument KD: no numeric values reported for this dataset")

    if write_csv_prefix:
        for tag, rows in (("legacy", leg_rows), ("autotune", aut_rows)):
            out = REPO / f"{write_csv_prefix}_{data_dir.name.replace(' ', '_')}_{tag}.csv"
            with open(out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"  wrote {out}")
    print()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", action="append", default=None,
                    help="Data folder(s); repeatable. Default: both example_data sets.")
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--csv", default=None,
                    help="Prefix for per-trace CSV output (optional).")
    args = ap.parse_args(argv)

    warnings.filterwarnings("ignore")  # curve_fit covariance / overflow noise
    cfg = load_config(Path(args.config))

    if args.data:
        dirs = [Path(d) for d in args.data]
    else:
        # Default: only example folders that are actually loadable
        # (have a Settings_WellInfo.xml and at least one .frd).
        base = REPO / "example_data"
        dirs = sorted(p for p in base.iterdir() if p.is_dir()
                      and (p / "Settings_WellInfo.xml").is_file()
                      and any(p.glob("*.frd")))
        skipped = sorted(p.name for p in base.iterdir() if p.is_dir()
                         and not ((p / "Settings_WellInfo.xml").is_file()
                                  and any(p.glob("*.frd"))))
        if skipped:
            print("(skipping non-loadable folders: "
                  + "; ".join(skipped) + ")\n")
    for d in dirs:
        d = d if d.is_absolute() else (REPO / d)
        if not d.is_dir():
            print(f"skip (not a dir): {d}")
            continue
        try:
            analyze(d, cfg, write_csv_prefix=args.csv)
        except FileNotFoundError as e:
            print(f"skip {d.name}: {e}\n")
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"ERROR on {d}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
