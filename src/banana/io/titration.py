"""Titration series data structure for kinetic binding experiments."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import logging
import numpy as np
import pandas as pd

from banana.io.bli import load_bli_directory, load_bli_xls
from banana.io.processing import ProcessingSpec, ProcessingMode, extract_processed_trace

logger = logging.getLogger(__name__)


def association_dissociation_start_times(
    t: np.ndarray,
    assoc_mask: Optional[np.ndarray],
    dissoc_mask: Optional[np.ndarray],
) -> Tuple[float, float]:
    """
    Fixed phase boundaries for PiecewiseExponential: t1 = association phase start,
    t2 = dissociation phase start (instrument time, same axis as trace time).

    Taken from the first ASSOC and first DISASSOC sample in the aligned masks.
    Fallbacks apply when masks are missing or empty.
    """
    t = np.asarray(t, dtype=float)
    n = len(t)
    if n == 0:
        return 0.0, 0.0
    if assoc_mask is not None and len(assoc_mask) == n and np.any(assoc_mask):
        t1 = float(t[np.argmax(np.asarray(assoc_mask, dtype=bool))])
    else:
        t1 = float(t[0])
    if dissoc_mask is not None and len(dissoc_mask) == n and np.any(dissoc_mask):
        t2 = float(t[np.argmax(np.asarray(dissoc_mask, dtype=bool))])
    else:
        t2 = float(t[-1])
    if t2 < t1:
        logger.warning(
            "Dissociation start (%.6g) before association start (%.6g); ordering corrected",
            t2,
            t1,
        )
        t1, t2 = min(t1, t2), max(t1, t2)
    return t1, t2


@dataclass
class TitrationSeries:
    """
    Container for a titration series: multiple concentrations with time-resolved binding data.
    """

    time: List[np.ndarray]
    response: List[np.ndarray]
    concentration: List[float]
    concentration_units: str = "nM"
    response_units: str = "nm"
    labels: Optional[List[str]] = None
    assoc_mask: Optional[List[np.ndarray]] = None
    dissoc_mask: Optional[List[np.ndarray]] = None
    time_from_assoc_start: Optional[List[np.ndarray]] = None
    association_start_t: Optional[List[float]] = None
    dissociation_start_t: Optional[List[float]] = None
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.time)
        if len(self.response) != n or len(self.concentration) != n:
            raise ValueError("time, response, and concentration must have same length")
        if self.labels is not None and len(self.labels) != n:
            raise ValueError("labels length must match number of traces")
        if self.assoc_mask is not None and len(self.assoc_mask) != n:
            raise ValueError("assoc_mask length must match number of traces")
        if self.dissoc_mask is not None and len(self.dissoc_mask) != n:
            raise ValueError("dissoc_mask length must match number of traces")
        if self.time_from_assoc_start is not None and len(self.time_from_assoc_start) != n:
            raise ValueError("time_from_assoc_start length must match number of traces")
        if self.association_start_t is not None and len(self.association_start_t) != n:
            raise ValueError("association_start_t length must match number of traces")
        if self.dissociation_start_t is not None and len(self.dissociation_start_t) != n:
            raise ValueError("dissociation_start_t length must match number of traces")
        if (self.association_start_t is None) != (self.dissociation_start_t is None):
            raise ValueError("association_start_t and dissociation_start_t must both be set or both omitted")

        for i in range(n):
            t_len = len(self.time[i])
            if len(self.response[i]) != t_len:
                raise ValueError(f"response length mismatch at trace {i}")
            if self.assoc_mask is not None and len(self.assoc_mask[i]) != t_len:
                raise ValueError(f"assoc_mask length mismatch at trace {i}")
            if self.dissoc_mask is not None and len(self.dissoc_mask[i]) != t_len:
                raise ValueError(f"dissoc_mask length mismatch at trace {i}")
            if self.time_from_assoc_start is not None and len(self.time_from_assoc_start[i]) != t_len:
                raise ValueError(f"time_from_assoc_start length mismatch at trace {i}")

    def __len__(self) -> int:
        return len(self.time)

    def get_trace(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.time[index], self.response[index]

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for i, (t, r, c) in enumerate(zip(self.time, self.response, self.concentration)):
            label = self.labels[i] if self.labels else str(i)
            for ti, ri in zip(t, r):
                rows.append({
                    "Time (sec)": ti,
                    "Response": ri,
                    "Concentration (nM)": c,
                    "Trace": label,
                })
        return pd.DataFrame(rows)


def extract_assoc_dissoc(
    df: pd.DataFrame,
    sensor: str,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    reference_sensor: Optional[str] = None,
    reference_df: Optional[pd.DataFrame] = None,
    baseline_points: int = 100,
    time_col: str = "Time (sec)",
    binding_col: str = "Binding (nm)",
    processing: Optional[ProcessingSpec] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract trace; legacy API. Uses ProcessingSpec (default SUBTRACT if ref given).
    """
    spec = processing or ProcessingSpec(
        mode=ProcessingMode.SUBTRACT if reference_sensor else ProcessingMode.RAW
    )
    work_df = df if reference_df is None else pd.concat([df, reference_df], ignore_index=True)
    ref = reference_sensor
    use_spec = spec
    if spec.needs_reference() and not ref:
        use_spec = ProcessingSpec(mode=ProcessingMode.RAW)
    return extract_processed_trace(
        work_df,
        ligand_sensor=sensor,
        reference_sensor=ref,
        spec=use_spec,
        t_start=t_start,
        t_end=t_end,
        baseline_points=baseline_points,
        time_col=time_col,
        binding_col=binding_col,
    )


def build_titration_from_bli_frd(
    directory: Union[str, Path],
    sensor_concentration_map: Dict[str, float],
    reference_sensor: Optional[str] = None,
    ligand_reference_map: Optional[Dict[str, str]] = None,
    processing: Optional[ProcessingSpec] = None,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    step_types: Optional[List[str]] = None,
) -> TitrationSeries:
    """
    Build TitrationSeries from a directory of .frd files.

    Parameters
    ----------
    sensor_concentration_map : dict
        Ligand sensor ID -> concentration (nM). Only these sensors become traces.
    reference_sensor : str, optional
        Single reference for all ligands (same as mapping each ligand to this sensor).
    ligand_reference_map : dict, optional
        Per-ligand reference: {ligand_sensor: reference_sensor}. Overrides reference_sensor
        for listed ligands.
    processing : ProcessingSpec, optional
        How to combine ligand and reference (default SUBTRACT = L − R).
        Use ProcessingMode.RAW for no reference; ADD for L+R; LINEAR for α·L+β·R.
    """
    df = load_bli_directory(directory)
    return _build_titration_from_dataframe(
        df,
        sensor_concentration_map,
        reference_sensor=reference_sensor,
        ligand_reference_map=ligand_reference_map,
        processing=processing,
        t_start=t_start,
        t_end=t_end,
        step_types=step_types,
    )


def build_titration_from_bli_xls(
    file_path: Union[str, Path],
    sensor_concentration_map: Dict[str, float],
    reference_sensor: Optional[str] = None,
    ligand_reference_map: Optional[Dict[str, str]] = None,
    processing: Optional[ProcessingSpec] = None,
) -> TitrationSeries:
    df = load_bli_xls(file_path)
    return _build_titration_from_dataframe(
        df,
        sensor_concentration_map,
        reference_sensor=reference_sensor,
        ligand_reference_map=ligand_reference_map,
        processing=processing,
    )


def _build_titration_from_dataframe(
    df: pd.DataFrame,
    sensor_concentration_map: Dict[str, float],
    reference_sensor: Optional[str] = None,
    ligand_reference_map: Optional[Dict[str, str]] = None,
    processing: Optional[ProcessingSpec] = None,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    step_types: Optional[List[str]] = None,
) -> TitrationSeries:
    spec = processing or ProcessingSpec(mode=ProcessingMode.SUBTRACT)
    if step_types and "Step Type" in df.columns:
        df = df[df["Step Type"].isin(step_types)].copy()

    time_list: List[np.ndarray] = []
    response_list: List[np.ndarray] = []
    concentration_list: List[float] = []
    labels_list: List[str] = []
    association_start_t_list: List[float] = []
    dissociation_start_t_list: List[float] = []

    for sensor, conc in sensor_concentration_map.items():
        if sensor not in df["Sensor"].values:
            logger.debug("Sensor %s not in data; skip", sensor)
            continue
        ref = None
        if ligand_reference_map and sensor in ligand_reference_map:
            ref = ligand_reference_map[sensor]
        elif reference_sensor:
            ref = reference_sensor
        per_spec = spec
        if spec.needs_reference() and not ref:
            logger.warning(
                "Sensor %s: %s needs reference; none set → using RAW (ligand only)",
                sensor,
                spec.describe(),
            )
            per_spec = ProcessingSpec(mode=ProcessingMode.RAW)
        try:
            t, r = extract_processed_trace(
                df,
                ligand_sensor=sensor,
                reference_sensor=ref,
                spec=per_spec,
                t_start=t_start,
                t_end=t_end,
            )
            if len(t) > 0:
                lig = df[df["Sensor"] == sensor].copy() if "Sensor" in df.columns else pd.DataFrame()
                assoc_mask = None
                dissoc_mask = None
                if (
                    "Step Type" in df.columns
                    and not lig.empty
                    and "Time (sec)" in lig.columns
                ):
                    lig = lig.sort_values("Time (sec)")
                    ts = lig["Time (sec)"].to_numpy()
                    if len(ts) == len(t):
                        st = lig["Step Type"].astype(str).to_numpy()
                        assoc_mask = st == "ASSOC"
                        dissoc_mask = st == "DISASSOC"
                t1_phase, t2_phase = association_dissociation_start_times(
                    t, assoc_mask, dissoc_mask
                )
                time_list.append(t)
                response_list.append(r)
                concentration_list.append(conc)
                labels_list.append(sensor)
                association_start_t_list.append(t1_phase)
                dissociation_start_t_list.append(t2_phase)
        except Exception as e:
            logger.warning("Skip sensor %s: %s", sensor, e)
            continue

    meta = {"processing": spec.describe()}
    return TitrationSeries(
        time=time_list,
        response=response_list,
        concentration=concentration_list,
        labels=labels_list,
        association_start_t=association_start_t_list or None,
        dissociation_start_t=dissociation_start_t_list or None,
        metadata=meta,
    )
