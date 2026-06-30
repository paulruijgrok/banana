#!/usr/bin/env python3
"""CLI: load BLI experiment from config, write processing_settings.yaml, optional fit/plot."""

import argparse
import logging
import re
import sys
from datetime import datetime
from pathlib import Path


def _safe_data_folder_name(path: Path) -> str:
    """Last path component of data dir, sanitized for use in a directory name."""
    name = path.resolve().name or "data"
    # Avoid empty or odd names from edge cases
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip()
    return name or "data"


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Banana BLI pipeline")
    parser.add_argument(
        "data_dir",
        nargs="?",
        default=".",
        help="Folder with .frd + Settings_WellInfo.xml",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="YAML config (default: configs/bli/BLI_default.yaml)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: data_dir or config output_dir)",
    )
    parser.add_argument(
        "--no-fit",
        action="store_true",
        help="Only import + write processing_settings.yaml",
    )
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help="Write outputs directly into the main output folder (no timestamped subfolder).",
    )
    args = parser.parse_args(argv)

    root = Path(args.data_dir).resolve()
    if not root.is_dir():
        print("Not a directory:", root, file=sys.stderr)
        return 1

    try:
        from banana.config.load_config import load_config, default_config_path
        from banana.io.bli_experiment import load_bli_experiment_from_config
        from banana.io.qc_export import write_processing_settings
    except ImportError as e:
        print(e, file=sys.stderr)
        return 1

    cfg_path = Path(args.config) if args.config else default_config_path()
    if not cfg_path.is_file():
        print("Config not found:", cfg_path, file=sys.stderr)
        return 1
    cfg = load_config(cfg_path)

    base_out = Path(args.output_dir or cfg.get("output", {}).get("output_dir") or root)
    base_out.mkdir(parents=True, exist_ok=True)

    if args.flat_output:
        out_dir = base_out
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sub = f"{_safe_data_folder_name(root)}_{stamp}"
        out_dir = base_out / sub
        out_dir.mkdir(parents=True, exist_ok=True)
        print("Output folder:", out_dir)

    qc_name = cfg.get("output", {}).get("processing_settings_file", "processing_settings.yaml")
    qc_path = out_dir / qc_name

    try:
        titration, bundle = load_bli_experiment_from_config(root, config_dict=cfg)
    except Exception as e:
        logging.exception("Load failed: %s", e)
        return 1

    write_processing_settings(
        qc_path,
        config=cfg,
        sensor_map=bundle["sensor_map"],
        sample_map=bundle["sample_map"],
        concentration_map=bundle["concentration_map"],
        ligand_reference_map=bundle["ligand_reference_map"],
        titration_from_sample_plate=bundle["titration_from_sample_plate"],
        extra={"n_traces": len(titration), "labels": titration.labels},
    )
    print("Wrote", qc_path)
    print("Titration traces:", len(titration))

    if args.no_fit:
        return 0

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from matplotlib.backends.backend_pdf import PdfPages
        from banana import (
            fit_titration_global,
            plot_titration_fits,
            plot_titration_fits_summary_table_page,
            plot_titration_trace_detailed_page,
            export_titration_data,
        )
        from banana.io.titration import TitrationSeries

        # Temporary debug output: write trace time/response arrays to CSV before fitting.
        debug_csv_path = out_dir / "titration_debug_traces.csv"
        debug_cols = {}
        for i in range(len(titration)):
            t, r = titration.get_trace(i)
            t_arr = np.asarray(t)
            r_arr = np.asarray(r)
            conc = titration.concentration[i]
            conc_label = f"{conc:g}{titration.concentration_units}"
            debug_cols[f"trace_{i}_{conc_label}_time_sec"] = pd.Series(t_arr)
            debug_cols[f"trace_{i}_{conc_label}_response"] = pd.Series(r_arr)
        pd.DataFrame(debug_cols).to_csv(debug_csv_path, index=False)
        print("Wrote", debug_csv_path)

        g, traces = fit_titration_global(titration, model_type="piecewise")
        plot_cfg = cfg.get("output", {}).get("titration_fits_plots", {}) or {}
        layout = str(plot_cfg.get("layout", "single_plot")).strip().lower()
        if layout == "detailed_single_page":
            layout = "detailed_single"
        if layout not in {"single_plot", "detailed_single"}:
            logging.warning("Unknown titration_fits_plots.layout=%r; using single_plot", layout)
            layout = "single_plot"

        plot_path = out_dir / "titration_fits.pdf"
        page_width = 6.0
        page_height = page_width * 1.0  # Keep height/width ratio at 1.
        raw_df = bundle.get("df")
        ligand_reference_map = bundle.get("ligand_reference_map", {}) or {}
        with PdfPages(plot_path) as pdf:
            for i in range(max(1, len(titration))):
                if len(titration) == 0 or layout == "single_plot":
                    fig = plot_titration_fits(
                        titration if len(titration) == 0 else TitrationSeries(
                            time=[titration.get_trace(i)[0]],
                            response=[titration.get_trace(i)[1]],
                            concentration=[titration.concentration[i]],
                            concentration_units=titration.concentration_units,
                            response_units=titration.response_units,
                            labels=(
                                [titration.labels[i]]
                                if titration.labels and i < len(titration.labels)
                                else None
                            ),
                            assoc_mask=(
                                [titration.assoc_mask[i]]
                                if titration.assoc_mask and i < len(titration.assoc_mask)
                                else None
                            ),
                            dissoc_mask=(
                                [titration.dissoc_mask[i]]
                                if titration.dissoc_mask and i < len(titration.dissoc_mask)
                                else None
                            ),
                            time_from_assoc_start=(
                                [titration.time_from_assoc_start[i]]
                                if titration.time_from_assoc_start and i < len(titration.time_from_assoc_start)
                                else None
                            ),
                            association_start_t=(
                                [titration.association_start_t[i]]
                                if titration.association_start_t and i < len(titration.association_start_t)
                                else None
                            ),
                            dissociation_start_t=(
                                [titration.dissociation_start_t[i]]
                                if titration.dissociation_start_t and i < len(titration.dissociation_start_t)
                                else None
                            ),
                            metadata=titration.metadata,
                        ),
                        traces if len(titration) == 0 else (traces[i : i + 1] if i < len(traces) else []),
                        model_type="piecewise",
                        figsize=(page_width, page_height),
                        ncols=1,
                    )
                else:
                    ligand_sensor = titration.labels[i] if titration.labels and i < len(titration.labels) else None
                    reference_sensor = (
                        ligand_reference_map.get(ligand_sensor)
                        if isinstance(ligand_reference_map, dict) and ligand_sensor
                        else None
                    )
                    fig = plot_titration_trace_detailed_page(
                        titration=titration,
                        trace_results=traces,
                        trace_index=i,
                        raw_df=raw_df,
                        ligand_sensor=ligand_sensor,
                        reference_sensor=reference_sensor,
                        model_type="piecewise",
                        figsize=(8.5, 11.0),
                    )
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
            fig_sum = plot_titration_fits_summary_table_page(
                titration,
                traces,
                g,
                model_type="piecewise",
            )
            pdf.savefig(fig_sum, bbox_inches="tight")
            plt.close(fig_sum)
        export_titration_data(titration, out_dir / "titration_data.csv")
        print("Saved titration_fits.pdf, titration_data.csv")
    except Exception as e:
        logging.warning("Fit/plot skipped: %s", e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
