"""Write processing_settings.yaml for QC (sensor_map, sample_map, concentration map)."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


def write_processing_settings(
    path: Union[str, Path],
    *,
    config: Optional[Dict[str, Any]] = None,
    sensor_map: Optional[Dict[str, str]] = None,
    sample_map: Optional[Dict[str, Any]] = None,
    concentration_map: Optional[Dict[str, Any]] = None,
    ligand_reference_map: Optional[Dict[str, str]] = None,
    titration_from_sample_plate: Optional[list] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Serialize analysis inputs for quality control. Safe round-trip via YAML.
    """
    if yaml is None:
        raise RuntimeError("PyYAML required: pip install pyyaml")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _clean(d):
        if d is None:
            return {}
        if isinstance(d, dict):
            return {str(k): _clean(v) for k, v in d.items()}
        if isinstance(d, list):
            return [_clean(x) for x in d]
        if isinstance(d, (str, int, float, bool)) or d is None:
            return d
        return str(d)

    doc = {
        "config": _clean(config or {}),
        "sensor_map": _clean(sensor_map or {}),
        "sample_map": _clean(sample_map or {}),
        "concentration_map": _clean(concentration_map or {}),
        "ligand_reference_map": _clean(ligand_reference_map or {}),
        "titration_from_sample_plate": _clean(titration_from_sample_plate or []),
    }
    if extra:
        doc["extra"] = _clean(extra)

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            doc,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
    return path
