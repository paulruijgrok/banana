"""Biolayer Interferometry (BLI) data import for ForteBio/Octet instruments."""

import base64
import glob
import os
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET


def _decode_base64(data: str) -> np.ndarray:
    """Decode base64 encoded float32 data from ForteBio .frd files."""
    decoded_bytes = base64.b64decode(data)
    return np.frombuffer(decoded_bytes, dtype=np.float32)


def load_bli_frd(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a single ForteBio/Octet .frd file (XML format with base64 encoded data).

    Returns a DataFrame with columns: Sensor, Step Name, Step Type, Time (sec), Binding (nm).
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    root = ET.fromstring(content)

    all_data = []
    exp_info = root.find(".//ExperimentInfo")
    sensor_name = "Unknown"
    if exp_info is not None and exp_info.find(".//SensorName") is not None:
        sensor_name = exp_info.find(".//SensorName").text or "Unknown"

    for step in root.findall(".//Step"):
        step_type_elem = step.find(".//StepType")
        step_name_elem = step.find(".//StepName")
        time_elem = step.find(".//AssayXData")
        binding_elem = step.find(".//AssayYData")

        if any(elem is None for elem in [step_type_elem, step_name_elem, time_elem, binding_elem]):
            continue

        step_type = step_type_elem.text or "UNKNOWN"
        step_name = step_name_elem.text or "Unknown"
        time_data = _decode_base64(time_elem.text)
        binding_data = _decode_base64(binding_elem.text)

        for t, b in zip(time_data, binding_data):
            all_data.append([sensor_name, step_name, step_type, float(t), float(b)])

    return pd.DataFrame(
        all_data,
        columns=["Sensor", "Step Name", "Step Type", "Time (sec)", "Binding (nm)"],
    )


def load_bli_directory(
    directory: Union[str, Path],
    pattern: str = "*.frd",
    sensor_filter: Optional[list] = None,
) -> pd.DataFrame:
    """
    Load all .frd files from a directory and combine into a single DataFrame.

    Parameters
    ----------
    directory : str or Path
        Path to the directory containing .frd files.
    pattern : str
        Glob pattern for files (default: *.frd).
    sensor_filter : list, optional
        If provided, only include data from these sensor names (e.g. ['A1', 'B1']).

    Returns
    -------
    pd.DataFrame
        Combined data from all files.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    files = sorted(glob.glob(str(directory / pattern)))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {directory}")

    frames = [load_bli_frd(f) for f in files]
    combined = pd.concat(frames, ignore_index=True)

    if sensor_filter is not None:
        combined = combined[combined["Sensor"].isin(sensor_filter)]

    return combined


def load_bli_xls(
    file_path: Union[str, Path],
    sensor: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load BLI data from Octet .xls export (tab-separated with sensor columns).

    RawData0.xls format: header A3, '', B3, ''... ; data rows have time,binding pairs.
    Single-sensor files (e.g. B3.xls): Time1, Data1 columns, possibly with metadata rows.

    Parameters
    ----------
    file_path : str or Path
        Path to the .xls file.
    sensor : str, optional
        If provided and file has multiple sensors, extract only this sensor.

    Returns
    -------
    pd.DataFrame
        Columns: Time (sec), Binding (nm), Sensor (if multiple).
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        df = pd.read_csv(file_path, sep="\t", header=None)
    except Exception:
        df = pd.read_excel(file_path, header=None)

    # Single-sensor format: 2+ columns, first rows may be metadata (Vers, Conc1, etc.)
    if df.shape[1] >= 2:
        # Find first numeric row
        data_start = 0
        for i in range(min(15, len(df))):
            try:
                float(df.iloc[i, 0])
                data_start = i
                break
            except (ValueError, TypeError):
                continue

        # If we have few columns and header row doesn't look like sensor names
        first_val = str(df.iloc[0, 0]).strip()
        if df.shape[1] <= 4 and first_val not in ("A1", "A3", "B1", "B3", "C1", "C3"):
            out = pd.DataFrame({
                "Time (sec)": pd.to_numeric(df.iloc[data_start:, 0], errors="coerce"),
                "Binding (nm)": pd.to_numeric(df.iloc[data_start:, 1], errors="coerce"),
            }).dropna()
            out["Sensor"] = sensor or file_path.stem
            return out

    # Multi-sensor RawData0 format: A3, '', B3, '', ... header
    header = df.iloc[0].astype(str)
    sensors = []
    for i in range(0, len(header), 2):
        if i < len(header):
            h = header.iloc[i].strip()
            if h and h not in ("", "nan") and not _is_numeric_string(h):
                sensors.append(h)
            else:
                sensors.append(f"S{i//2}")

    all_rows = []
    for r in range(1, len(df)):
        row = df.iloc[r]
        for i in range(0, min(len(row) - 1, len(sensors) * 2), 2):
            try:
                t = float(row.iloc[i])
                b = float(row.iloc[i + 1])
                sens = sensors[i // 2] if i // 2 < len(sensors) else f"S{i//2}"
                if sensor is None or sens == sensor:
                    all_rows.append({"Time (sec)": t, "Binding (nm)": b, "Sensor": sens})
            except (ValueError, TypeError, IndexError):
                continue

    if not all_rows:
        raise ValueError(f"Could not parse BLI data from {file_path}")

    out = pd.DataFrame(all_rows)
    if sensor:
        out = out[out["Sensor"] == sensor]
    return out


def _is_numeric_string(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


