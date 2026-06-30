"""Metadata from ForteBio .frd files (Association step concentration, sensor name)."""

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

# nM multipliers from MolarConcUnits
_TO_NM = {
    "nm": 1.0,
    "nM": 1.0,
    "µm": 1000.0,
    "um": 1000.0,
    "μm": 1000.0,
    "mM": 1e6,
    "MM": 1e6,
    "pm": 1e-3,
    "pM": 1e-3,
}


def _to_nM(value: float, units: str) -> float:
    u = (units or "nM").strip()
    mult = _TO_NM.get(u, 1.0)
    if mult == 1.0 and u.lower() not in ("nm", "nm"):
        # try case-insensitive
        for k, m in _TO_NM.items():
            if k.lower() == u.lower():
                return value * m
    return value * mult


def parse_frd_association_concentration(
    frd_path: Union[str, Path],
) -> Tuple[Optional[float], str, Optional[str]]:
    """
    Read Association step from .frd: MolarConcentration → nM.

    Returns
    -------
    concentration_nM : float or None
    units_raw : str
    sample_id : optional SampleID from that step
    """
    path = Path(frd_path)
    if not path.exists():
        return None, "", None
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except OSError:
        return None, "", None
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return None, "", None

    sensor = "Unknown"
    exp = root.find(".//ExperimentInfo")
    if exp is not None:
        sn = exp.find("SensorName")
        if sn is not None and sn.text:
            sensor = sn.text.strip()

    for step in root.findall(".//Step"):
        name_el = step.find(".//StepName")
        if name_el is None or (name_el.text or "").strip() != "Association":
            continue
        cd = step.find(".//CommonData")
        if cd is None:
            continue
        mc = cd.find("MolarConcentration")
        mu = cd.find("MolarConcUnits")
        sid = cd.find("SampleID")
        if mc is None or mc.text is None:
            continue
        try:
            v = float(mc.text)
        except ValueError:
            continue
        if v < 0:
            continue
        units = _text(mu)
        sample_id = _text(sid) or None
        return _to_nM(v, units), units, sample_id
    return None, "", None


def _text(elem) -> str:
    if elem is None or elem.text is None:
        return ""
    return str(elem.text).strip()


def resolve_frd_path(
    directory: Path,
    sensor_well: str,
    sensor_frd_basename: Optional[str] = None,
) -> Optional[Path]:
    """
    Find .frd file for a sensor well in directory.
    Prefer basename from Settings if present; else match ExperimentInfo SensorName.
    """
    directory = Path(directory)
    if sensor_frd_basename:
        p = directory / sensor_frd_basename
        if p.exists():
            return p
    # glob by sensor name in file — load each frd's SensorName
    for f in sorted(directory.glob("*.frd")):
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fp:
                head = fp.read(8000)
            if f"<SensorName>{sensor_well}</SensorName>" in head:
                return f
        except OSError:
            continue
    return None


def build_sensor_frd_concentration_map(
    directory: Union[str, Path],
    sensor_map: Dict[str, str],
    sensor_frd_basenames: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, Union[float, str, None]]]:
    """
    For each ligand sensor, locate .frd and read Association MolarConcentration → nM.

    Returns
    -------
    dict[ligand_well] -> {"concentration_nM", "molar_conc_units", "sample_id", "frd_file"}
    """
    directory = Path(directory)
    basenames = sensor_frd_basenames or {}
    out: Dict[str, Dict[str, Union[float, str, None]]] = {}
    for well, st in sensor_map.items():
        if st != "Ligand Sensor":
            continue
        frd = resolve_frd_path(directory, well, basenames.get(well))
        if frd is None:
            out[well] = {
                "concentration_nM": None,
                "molar_conc_units": None,
                "sample_id": None,
                "frd_file": None,
            }
            continue
        cn, units, sid = parse_frd_association_concentration(frd)
        out[well] = {
            "concentration_nM": cn,
            "molar_conc_units": units or None,
            "sample_id": sid,
            "frd_file": str(frd.name),
        }
    return out
