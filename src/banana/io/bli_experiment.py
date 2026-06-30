"""
Load a BLI titration experiment from directory + YAML config (sensor block aware).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from banana.config.load_config import load_config
from banana.io.bli import load_bli_frd
from banana.io.frd_meta import build_sensor_frd_concentration_map
from banana.io.processing import (
    ProcessingMode,
    ProcessingSpec,
    apply_association_dissociation_postprocessing,
)
from banana.io.titration import (
    TitrationSeries,
    association_dissociation_start_times,
    extract_processed_trace,
)
from banana.io.well_info import (
    load_well_info,
    load_titration_info,
    ligand_reference_pairs_concentration_matched,
    get_sensor_frd_basenames,
)

logger = logging.getLogger(__name__)


def load_bli_experiment_from_config(
    data_dir: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
    config_dict: Optional[Dict[str, Any]] = None,
) -> Tuple[TitrationSeries, Dict[str, Any]]:
    """
    Load titration series using config + Settings_WellInfo.xml + .frd files.

    Returns
    -------
    titration : TitrationSeries
    qc_bundle : dict with keys for write_processing_settings + combined df
    """
    data_dir = Path(data_dir)
    if config_dict is None:
        if config_path is None:
            from banana.config.load_config import default_config_path
            config_path = default_config_path()
        cfg = load_config(config_path)
    else:
        cfg = dict(config_dict)

    sensor_map, sample_map = load_well_info(data_dir)

    frd_basenames = get_sensor_frd_basenames(data_dir)

    meas = cfg.get("measurement") or {}
    mtype = (meas.get("type") or "titration").lower()
    ref_type = (meas.get("reference_type") or "none").lower()

    titration_samples = load_titration_info(sample_map, mtype) if mtype == "titration" else []

    conc_from_frd = build_sensor_frd_concentration_map(data_dir, sensor_map, frd_basenames)

    ligand_ref: Dict[str, str] = {}
    if ref_type == "concentration_matched":
        ligand_ref = ligand_reference_pairs_concentration_matched(sensor_map)
    elif ref_type == "none":
        ligand_ref = {}

    proc = cfg.get("processing") or {}
    align_to_association_phase = bool(proc.get("align_to_association_phase", False))
    step_correction = bool(proc.get("step_correction", False))
    spec = ProcessingSpec(
        mode=ProcessingMode.LINEAR,
        alpha_ligand=float(proc.get("alpha_ligand", 1.0)),
        beta_reference=float(proc.get("beta_reference", -1.0 if ref_type != "none" else 0.0)),
    )
    if ref_type == "none":
        spec = ProcessingSpec(mode=ProcessingMode.RAW)

    # Build combined DataFrame from all relevant .frd
    frames = []
    wells_to_load = set(sensor_map.keys())
    for well in wells_to_load:
        bn = frd_basenames.get(well)
        if not bn:
            continue
        p = data_dir / bn
        if p.exists():
            try:
                frames.append(load_bli_frd(p))
            except Exception as e:
                logger.warning("Failed load %s: %s", p, e)
    if not frames:
        for f in sorted(data_dir.glob("*.frd")):
            try:
                frames.append(load_bli_frd(f))
            except Exception:
                pass
    if not frames:
        raise FileNotFoundError(f"No loadable .frd in {data_dir}")

    df = pd.concat(frames, ignore_index=True)
    if "Step Type" in df.columns:
        df = df[df["Step Type"].isin(["ASSOC", "DISASSOC"])].copy()

    concentration_map: Dict[str, Any] = {}
    for well, meta in conc_from_frd.items():
        concentration_map[well] = dict(meta)

    time_list: List = []
    response_list: List = []
    concentration_list: List[float] = []
    labels_list: List[str] = []
    assoc_mask_list: List[np.ndarray] = []
    dissoc_mask_list: List[np.ndarray] = []
    time_from_assoc_start_list: List[np.ndarray] = []
    association_start_t_list: List[float] = []
    dissociation_start_t_list: List[float] = []

    for well, st in sensor_map.items():
        if st != "Ligand Sensor":
            continue
        meta = conc_from_frd.get(well) or {}
        conc = meta.get("concentration_nM")
        if conc is None:
            # Fallback: first plate that has this well in sample_map
            for plate_idx, wells in sample_map.items():
                if well in wells:
                    conc = wells[well].get("concentration_nM")
                    if conc is not None:
                        break
        if conc is None:
            logger.warning("No concentration for ligand %s; skip", well)
            continue
        ref_well = ligand_ref.get(well) if ref_type == "concentration_matched" else None
        try:
            t, r = extract_processed_trace(
                df,
                ligand_sensor=well,
                reference_sensor=ref_well,
                spec=spec,
                t_start=None,
                t_end=None,
            )
            if len(t) == 0:
                continue
        except Exception as e:
            logger.warning("Trace %s: %s", well, e)
            continue

        # Build phase masks aligned to the ligand time grid.
        ligand_rows = df[df["Sensor"] == well].copy()
        if "Time (sec)" in ligand_rows.columns and "Step Type" in ligand_rows.columns:
            ligand_rows = ligand_rows.sort_values("Time (sec)")
            t_sensor = ligand_rows["Time (sec)"].to_numpy()
            step_sensor = ligand_rows["Step Type"].astype(str).to_numpy()
            if len(t_sensor) == len(t):
                assoc_mask = (step_sensor == "ASSOC")
                dissoc_mask = (step_sensor == "DISASSOC")
            else:
                logger.warning(
                    "Trace %s: phase mask length mismatch (time=%d, step=%d); using empty masks",
                    well,
                    len(t),
                    len(t_sensor),
                )
                assoc_mask = np.zeros(len(t), dtype=bool)
                dissoc_mask = np.zeros(len(t), dtype=bool)
        else:
            assoc_mask = np.zeros(len(t), dtype=bool)
            dissoc_mask = np.zeros(len(t), dtype=bool)

        r = apply_association_dissociation_postprocessing(
            r,
            assoc_mask,
            dissoc_mask,
            align_to_association_phase=align_to_association_phase,
            step_correction=step_correction,
        )

        if np.any(assoc_mask):
            assoc_start_idx = int(np.argmax(assoc_mask))
            assoc_start_t = float(t[assoc_start_idx])
        else:
            assoc_start_t = float(t[0])
        t_assoc0 = np.asarray(t, dtype=float) - assoc_start_t

        t1_phase, t2_phase = association_dissociation_start_times(
            t, assoc_mask, dissoc_mask
        )

        time_list.append(t)
        response_list.append(r)
        concentration_list.append(float(conc))
        labels_list.append(well)
        assoc_mask_list.append(np.asarray(assoc_mask, dtype=bool))
        dissoc_mask_list.append(np.asarray(dissoc_mask, dtype=bool))
        time_from_assoc_start_list.append(t_assoc0)
        association_start_t_list.append(t1_phase)
        dissociation_start_t_list.append(t2_phase)

    titration = TitrationSeries(
        time=time_list,
        response=response_list,
        concentration=concentration_list,
        labels=labels_list,
        assoc_mask=assoc_mask_list,
        dissoc_mask=dissoc_mask_list,
        time_from_assoc_start=time_from_assoc_start_list,
        association_start_t=association_start_t_list,
        dissociation_start_t=dissociation_start_t_list,
        metadata={
            "config_name": cfg.get("name"),
            "reference_type": ref_type,
            "processing": spec.describe(),
            "align_to_association_phase": align_to_association_phase,
            "step_correction": step_correction,
        },
    )

    qc_bundle = {
        "config": cfg,
        "sensor_map": sensor_map,
        "sample_map": sample_map,
        "concentration_map": concentration_map,
        "ligand_reference_map": ligand_ref,
        "titration_from_sample_plate": titration_samples,
        "df": df,
    }
    return titration, qc_bundle
