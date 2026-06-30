"""
Parse ForteBio Settings_WellInfo.xml: sensor plate + sample plate.

sensor_map: non-empty SensorPlate_K wells → Ligand Sensor | Reference Sensor
sample_map: all SamplePlate_K wells → type, concentration (nM), metadata
"""

import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

SENSOR_TYPES = frozenset({"Ligand Sensor", "Reference Sensor"})


def _text(elem, default: str = "") -> str:
    if elem is None or elem.text is None:
        return default
    return str(elem.text).strip()


def _parse_empty(elem) -> bool:
    """True if slot is empty (no sensor)."""
    t = _text(elem.find("Empty")).lower()
    return t == "true"


def _well_row_col(well: str) -> Tuple[str, int]:
    """Parse 'A1' -> ('A', 1)."""
    m = re.match(r"^([A-Ha-h])(\d+)$", well.strip())
    if not m:
        return "", -1
    return m.group(1).upper(), int(m.group(2))


def concentration_matched_reference_well(ligand_well: str) -> Optional[str]:
    """
    Default concentration_matched pairing: same row, reference column = ligand column + 1.
    e.g. A1 → A2, B1 → B2.
    """
    row, col = _well_row_col(ligand_well)
    if not row or col < 1:
        return None
    return f"{row}{col + 1}"


def load_well_info(
    directory_or_xml: Union[str, Path],
) -> Tuple[Dict[str, str], Dict[int, Dict[str, Dict[str, Any]]]]:
    """
    Load Settings_WellInfo.xml from a directory or explicit path.

    Returns
    -------
    sensor_map : dict[str, str]
        WellLocation → "Ligand Sensor" | "Reference Sensor"
        Only SensorPlate_K entries with Empty == false (active sensors).
    sample_map : dict[int, dict[str, dict]]
        Plate-indexed sample info (supports multiple sample plates).

        sample_map[plate_index][well_location] = {
            "well_type": "Sample" | "Buffer" | "unset",
            "concentration_nM": float | None,  # YValue when Sample
            "sample_id": str,
            "empty": bool,
            "row": str,
            "column": str,
        }

        Plates are numbered 1,2,3,... in the order encountered. If the XML
        contains N copies of a given WellLocation (e.g. A1 on two plates),
        those are assigned to plate 1 and plate 2 respectively.
    """
    path = Path(directory_or_xml)
    if path.is_dir():
        xml_path = path / "Settings_WellInfo.xml"
    else:
        xml_path = path
    if not xml_path.exists():
        logger.warning("Settings_WellInfo.xml not found: %s", xml_path)
        return {}, {}

    tree = ET.parse(xml_path)
    root = tree.getroot()

    sensor_map: Dict[str, str] = {}
    sensor_frd_files: Dict[str, str] = {}  # well -> basename for QC

    for elem in root.findall(".//SensorPlate_K"):
        if _parse_empty(elem):
            continue
        well = _text(elem.find("WellLocation"))
        if not well:
            continue
        st = _text(elem.find("Type"))
        if st not in SENSOR_TYPES:
            logger.debug("Sensor %s: unknown Type %r, skipping", well, st)
            continue
        sensor_map[well] = st
        fn = _text(elem.find("FileName"))
        if fn:
            sensor_frd_files[well] = Path(fn).name

    # Sample plates: allow multiple plates. We infer plate index from how many
    # times a WellLocation has appeared so far (first A1 = plate 1, second A1 = plate 2, ...).
    sample_map: Dict[int, Dict[str, Any]] = {}
    well_counts: Dict[str, int] = {}
    for elem in root.findall(".//SamplePlate_K"):
        well = _text(elem.find("WellLocation"))
        if not well:
            continue
        plate_index = well_counts.get(well, 0) + 1
        well_counts[well] = plate_index
        raw_type = _text(elem.find("Type"))
        if raw_type in ("Sample",):
            well_type = "Sample"
        elif raw_type in ("Buffer",):
            well_type = "Buffer"
        else:
            well_type = "unset"
        yv = elem.find("YValue")
        conc = None
        if yv is not None and yv.text and str(yv.text).strip().lower() not in ("nan", ""):
            try:
                conc = float(yv.text)
                if conc != conc:  # NaN
                    conc = None
            except ValueError:
                conc = None
        plate_dict = sample_map.setdefault(plate_index, {})
        plate_dict[well] = {
            "well_type": well_type,
            "concentration_nM": conc if well_type == "Sample" else None,
            "sample_id": _text(elem.find("ID")),
            "empty": _parse_empty(elem),
            "row": _text(elem.find("Row")),
            "column": _text(elem.find("Column")),
        }

    # Attach frd hints into a side channel via sample_map meta — store on sensor_map wells only in processing_settings
    # Expose frd basename via extended load if needed; processing pipeline resolves paths
    load_well_info._sensor_frd_basenames = sensor_frd_files  # noqa
    return sensor_map, sample_map


def get_sensor_frd_basenames(directory_or_xml: Union[str, Path]) -> Dict[str, str]:
    """WellLocation → .frd basename from SensorPlate_K FileName."""
    load_well_info(directory_or_xml)
    return dict(getattr(load_well_info, "_sensor_frd_basenames", {}))


def load_titration_info(
    sample_map: Dict[int, Dict[str, Dict[str, Any]]],
    measurement_type: str = "titration",
) -> List[Dict[str, Any]]:
    """
    For measurement type titration: list sample wells with concentration (nM).

    Each entry: well, concentration_nM, sample_id
    """
    if measurement_type.lower() != "titration":
        return []
    out: List[Dict[str, Any]] = []
    for plate_idx, wells in sample_map.items():
        for well, info in wells.items():
            if info.get("well_type") != "Sample":
                continue
            c = info.get("concentration_nM")
            if c is None:
                continue
            out.append({
                "plate": plate_idx,
                "well": well,
                "concentration_nM": float(c),
                "sample_id": info.get("sample_id") or "",
            })
    # Sort by plate, then decreasing concentration, then well location
    out.sort(key=lambda x: (x["plate"], -x["concentration_nM"], x["well"]))
    return out


def ligand_reference_pairs_concentration_matched(
    sensor_map: Dict[str, str],
) -> Dict[str, str]:
    """
    Map each Ligand Sensor well → Reference Sensor well (same row, col+1).
    """
    pairs = {}
    for well, st in sensor_map.items():
        if st != "Ligand Sensor":
            continue
        ref_well = concentration_matched_reference_well(well)
        if ref_well and sensor_map.get(ref_well) == "Reference Sensor":
            pairs[well] = ref_well
        else:
            logger.warning(
                "No Reference Sensor at %s for ligand %s", ref_well, well
            )
    return pairs
