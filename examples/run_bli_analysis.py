#!/usr/bin/env python3
"""Legacy example; prefer run_BLI.py + YAML config."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
logging.basicConfig(level=logging.INFO)

from banana.io.well_info import load_well_info, load_titration_info
from banana.config.load_config import default_config_path, load_config
from banana.io.bli_experiment import load_bli_experiment_from_config
from banana.io.qc_export import write_processing_settings

def main():
    root = Path(__file__).parent.parent
    data_dir = root / "example_data" / "20260311 Nb6"
    if not data_dir.exists():
        data_dir = root / "example_data" / "20260311 Nb8"
    if not data_dir.exists():
        print("No example data")
        return
    cfg = load_config(root / "src" / "banana" / "configs" / "bli" / "octet_384_16sensor_titration_concentration_matched.yaml")
    titration, bundle = load_bli_experiment_from_config(data_dir, config_dict=cfg)
    out = root / "output"
    out.mkdir(exist_ok=True)
    write_processing_settings(
        out / "processing_settings.yaml",
        config=cfg,
        sensor_map=bundle["sensor_map"],
        sample_map=bundle["sample_map"],
        concentration_map=bundle["concentration_map"],
        ligand_reference_map=bundle["ligand_reference_map"],
        titration_from_sample_plate=bundle["titration_from_sample_plate"],
    )
    print("Traces:", len(titration), "QC:", out / "processing_settings.yaml")

if __name__ == "__main__":
    main()
