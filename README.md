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

## License

MIT
