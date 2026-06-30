"""Load BLI / experiment YAML config."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML required: pip install pyyaml")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def default_config_path(name: str = "BLI_default.yaml") -> Path:
    return Path(__file__).resolve().parent.parent / "configs" / "bli" / name
