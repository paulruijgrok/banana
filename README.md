# Banana

Kinetic binding analysis (BLI, SPR). Sensor-block aware import from ForteBio **Settings_WellInfo.xml** + **YAML** config.

## Config-driven BLI pipeline

1. **`load_well_info(data_dir)`** → **`sensor_map`**, **`sample_map`**
   - **sensor_map**: `SensorPlate_K` entries with `<Empty>false</Empty>` → `{ "A1": "Ligand Sensor", "A2": "Reference Sensor", ... }`
   - **sample_map**: **plate-indexed** SamplePlate wells →  
     `sample_map[1]["A1"]`, `sample_map[2]["A1"]`, … (plate 1, 2, … as detected from repeated `WellLocation`):
     ```python
     {
       1: {"A1": {"well_type": "Sample", "concentration_nM": 3000.0, "sample_id": "Nb6 3000 nM", ...}, ...},
       2: {"A1": {"well_type": "Sample", "concentration_nM": 3000.0, "sample_id": "Nb6 3000 nM", ...}, ...},
     }
     ```

2. **`load_titration_info(sample_map, type="titration")`** → list of sample wells with concentrations (nM).

3. **Per ligand `.frd`**: **Association** step → `MolarConcentration` + `MolarConcUnits` → concentration in **nM** (`build_sensor_frd_concentration_map`).

4. **`reference_type: concentration_matched`**: ligand **A1** → reference **A2** (same row, column + 1). Processing default: **L − R**.

5. **Each run** writes **`processing_settings.yaml`**: `sensor_map`, `sample_map`, `concentration_map`, `ligand_reference_map`, config snapshot.

### How the `banana-bli` command works

There is **no** `banana-bli` script file in the repo. The command is created by **setuptools** when you install the package:

- In **`pyproject.toml`**: `[project.scripts]` defines `banana-bli = "banana.cli.run_bli:main"`.
- After **`pip install -e .`**, setuptools puts a small launcher (e.g. `venv/bin/banana-bli` or `~/.local/bin/banana-bli`) that imports **`banana.cli.run_bli`** and calls **`main()`**.
- The real code lives in **`src/banana/cli/run_bli.py`**, function **`main()`**.

**Ways to run the same CLI:**

1. **After install** (if `banana` is on your PATH):
   ```bash
   pip install -e .
   banana-bli /path/to/data -c config.yaml -o output
   ```

2. **Without installing**, from the project root:
   ```bash
   PYTHONPATH=src python -m banana.cli.run_bli /path/to/data -c src/banana/configs/bli/BLI_default.yaml -o output
   ```

3. **Example wrapper** (same behavior):
   ```bash
   python examples/run_BLI.py /path/to/data -c config.yaml
   ```
   (`examples/run_BLI.py` just calls `banana.cli.run_bli.main()`.)

### CLI usage

```bash
pip install -e .
banana-bli /path/to/Nb6_folder -c src/banana/configs/bli/octet_384_16sensor_titration_concentration_matched.yaml -o output
# Import + QC only (no fitting/plotting):
banana-bli /path/to/Nb6 --no-fit
```

### Python

```python
from banana import load_well_info, load_bli_experiment_from_config, write_processing_settings, load_config

sensor_map, sample_map = load_well_info("example_data/20260311 Nb6")
cfg = load_config("src/banana/configs/bli/BLI_default.yaml")
titration, bundle = load_bli_experiment_from_config("example_data/20260311 Nb6", config_dict=cfg)
write_processing_settings("output/processing_settings.yaml", **{k: bundle[k] for k in ["sensor_map","sample_map","concentration_map","ligand_reference_map"]}, config=cfg,
    titration_from_sample_plate=bundle["titration_from_sample_plate"])
```

## YAML configs

- **`configs/bli/BLI_default.yaml`** – defaults
- **`configs/bli/octet_384_16sensor_titration_concentration_matched.yaml`** – 16-sensor block + concentration-matched references

## Fitting models

Banana fits both **time-domain** kinetic traces and **steady-state** binding curves.

### Time-domain (response vs. time)

- **`piecewise`** – association (rise) + dissociation (decay) exponentials. Default.
- **`1_to_1_binding`** – global 1:1 with shared `k_on`/`k_off` and per-trace `R_max`.
- **`avidity`** – bivalent / heterobivalent ODE model (Vauquelin, *Br J Pharmacol*
  2013, Appendix S1). Monovalent, homobivalent, and the 5-species heterobivalent
  avidity scheme (ring closure via local concentration `L` and penalty factor `f`),
  integrated with `scipy.integrate.solve_ivp` (stiff LSODA), with a NumPy RK4
  fallback. Captures avidity (raised functional affinity) and prolonged residence
  time. Configure under `kinetic_model:` (`scheme`, `symmetric`, `L`, `rebind_k`).

```python
from banana.fitting import fit_titration_global
g, traces = fit_titration_global(titration, model_type="avidity",
                                 p0={"scheme": "heterobivalent", "symmetric": True})
```

### Steady-state (signal vs. concentration)

Per-trace equilibrium response `R_eq` is extracted from the association plateau and
fit against concentration:

- **`hyperbolic`** – 1:1 Langmuir (Jarmoskaite & Herschlag, *eLife* 2020, Eq. 4b).
- **`quadratic`** – depletion-aware 1:1 (Jarmoskaite 2020, Eq. 5); use when the
  limiting species is not ≪ K_D.
- **`two_site_microscopic`** – 1:2 with two micro K_D's (Tso et al., *Anal Biochem*
  2018).
- **`two_site_macroscopic`** – 1:2 symmetric, reports K_D,M and cooperativity α.
- **`hill`** – Hill equation.

```python
from banana.fitting import fit_titration_equilibrium
res = fit_titration_equilibrium(titration, model="hyperbolic")
print(res.to_dict())          # R_max, K_D, stderrs, chi2
```

Also: `equilibration_time(k_on, k_off, conc_M)` returns the time to reach
equilibrium (Jarmoskaite Eqs. 1–2), a QC check for insufficient incubation.

### Autotuning (data-driven initial guesses)

Fits no longer rely on hardcoded starting parameters. `fitting/autotune.py` derives
initial guesses and physical bounds directly from each trace — `R_max` from the
plateau, `k_off` from the dissociation-tail log-slope, `k_obs` from association
curvature, `K_D` from the titration midpoint. This is wired into the piecewise,
1:1, and equilibrium fits automatically; explicit `p0` values still override.

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

`tests/test_new_models.py` covers the model math (mass balances, limiting cases,
analytic kinetics, cubic solver) and the scipy-backed fits.

## License

MIT
