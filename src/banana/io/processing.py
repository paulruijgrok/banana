"""
Processing pipeline: combine ligand and reference sensor traces before fitting.

Supports mathematical relations between ligand (L) and reference (R) responses
interpolated onto the ligand time grid: processed = α·L + β·R (and presets).
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ProcessingMode(str, Enum):
    """Named presets for ligand/reference combination."""

    RAW = "raw"  # ligand only (no reference)
    SUBTRACT = "subtract"  # L - R (reference subtraction)
    ADD = "add"  # L + R
    LINEAR = "linear"  # α·L + β·R (set alpha_ligand, beta_reference)


@dataclass
class ProcessingSpec:
    """
    Defines how ligand and reference BLI traces are combined.

    After interpolation of reference onto ligand times:
        response_processed = alpha_ligand * R_ligand + beta_reference * R_ref

    Presets (set mode and optionally override alphas):
        - raw: β=0 effectively (reference ignored); same as α=1, β=0
        - subtract: α=1, β=-1  →  L - R
        - add: α=1, β=1  →  L + R
        - linear: use alpha_ligand, beta_reference explicitly

    Examples
    --------
    >>> ProcessingSpec(mode=ProcessingMode.SUBTRACT)  # default BLI reference subtraction
    >>> ProcessingSpec(mode=ProcessingMode.RAW)       # no reference
    >>> ProcessingSpec(mode=ProcessingMode.LINEAR, alpha_ligand=1.0, beta_reference=-0.5)
    """

    mode: Union[ProcessingMode, str] = ProcessingMode.SUBTRACT
    alpha_ligand: float = 1.0
    beta_reference: float = -1.0

    def __post_init__(self):
        if isinstance(self.mode, str):
            self.mode = ProcessingMode(self.mode.lower())

    def effective_coefficients(self) -> Tuple[float, float]:
        """Return (alpha, beta) for processed = alpha*L + beta*R."""
        if self.mode == ProcessingMode.RAW:
            return 1.0, 0.0
        if self.mode == ProcessingMode.SUBTRACT:
            return 1.0, -1.0
        if self.mode == ProcessingMode.ADD:
            return 1.0, 1.0
        # LINEAR: user alphas
        return float(self.alpha_ligand), float(self.beta_reference)

    def needs_reference(self) -> bool:
        alpha, beta = self.effective_coefficients()
        return beta != 0.0

    def describe(self) -> str:
        a, b = self.effective_coefficients()
        if b == 0:
            return "raw (ligand only)"
        if a == 1 and b == -1:
            return "ligand − reference"
        if a == 1 and b == 1:
            return "ligand + reference"
        return f"{a:g}·ligand + ({b:g})·reference"


def combine_ligand_reference(
    time_ligand: np.ndarray,
    response_ligand: np.ndarray,
    time_ref: np.ndarray,
    response_ref: np.ndarray,
    spec: ProcessingSpec,
) -> np.ndarray:
    """
    Combine ligand and reference responses on the ligand time grid.

    Reference is linearly interpolated onto time_ligand. If reference is not
    needed (beta=0), only ligand is returned (copy).
    """
    a, b = spec.effective_coefficients()
    if b == 0.0:
        return np.asarray(response_ligand, dtype=float).copy()
    ref_on_ligand = np.interp(
        np.asarray(time_ligand, dtype=float),
        np.asarray(time_ref, dtype=float),
        np.asarray(response_ref, dtype=float),
    )
    return a * np.asarray(response_ligand, dtype=float) + b * ref_on_ligand


def extract_sensor_trace(
    df: pd.DataFrame,
    sensor: str,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    time_col: str = "Time (sec)",
    binding_col: str = "Binding (nm)",
) -> Tuple[np.ndarray, np.ndarray]:
    """Raw time and binding for one sensor (optionally time-windowed), sorted by time."""
    sub = df[df["Sensor"] == sensor].copy()
    if t_start is not None:
        sub = sub[sub[time_col] >= t_start]
    if t_end is not None:
        sub = sub[sub[time_col] <= t_end]
    if sub.empty:
        return np.array([]), np.array([])
    t = sub[time_col].values
    y = sub[binding_col].values.astype(float)
    order = np.argsort(t)
    return t[order], y[order]


def extract_processed_trace(
    df: pd.DataFrame,
    ligand_sensor: str,
    reference_sensor: Optional[str],
    spec: ProcessingSpec,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    baseline_points: int = 100,
    time_col: str = "Time (sec)",
    binding_col: str = "Binding (nm)",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract ligand trace, optionally combine with reference per ProcessingSpec,
    then baseline-subtract (mean of first baseline_points).
    """
    t_l, y_l = extract_sensor_trace(
        df, ligand_sensor, t_start, t_end, time_col, binding_col
    )
    if len(t_l) == 0:
        return t_l, y_l

    if spec.needs_reference():
        if not reference_sensor:
            raise ValueError(
                f"ProcessingSpec '{spec.describe()}' requires a reference sensor; "
                "set ligand_reference_map or reference_sensor."
            )
        t_r, y_r = extract_sensor_trace(
            df, reference_sensor, t_start, t_end, time_col, binding_col
        )
        if len(t_r) == 0:
            raise ValueError(f"Reference sensor {reference_sensor!r} has no data in window.")
        y_proc = combine_ligand_reference(t_l, y_l, t_r, y_r, spec)
    else:
        y_proc = y_l.copy()

    if len(y_proc) >= baseline_points:
        y_proc = y_proc - np.mean(y_proc[:baseline_points])

    return t_l, y_proc


def apply_association_dissociation_postprocessing(
    response: np.ndarray,
    assoc_mask: np.ndarray,
    dissoc_mask: np.ndarray,
    *,
    align_to_association_phase: bool = False,
    step_correction: bool = False,
) -> np.ndarray:
    """
    Optional corrections after ligand–reference combination and baseline subtraction.

    Parameters
    ----------
    align_to_association_phase
        If True, subtract a constant so the first association sample is 0 (applies to the
        full trace: ``r -= r[first_assoc]``).
    step_correction
        If True, subtract a constant from all dissociation samples so the first dissociation
        value equals the last association value (removes the step at the phase boundary).

    Order: alignment first, then step correction (when both are enabled).
    """
    r = np.asarray(response, dtype=float).copy()
    n = len(r)
    am = np.asarray(assoc_mask, dtype=bool)
    dm = np.asarray(dissoc_mask, dtype=bool)
    if n == 0:
        return r
    if len(am) != n or len(dm) != n:
        logger.warning(
            "Phase masks length (%s, %s) != response length (%s); skipping assoc/dissoc postprocessing",
            len(am),
            len(dm),
            n,
        )
        return r

    if align_to_association_phase and np.any(am):
        i_first_assoc = int(np.argmax(am))
        r -= r[i_first_assoc]

    if step_correction and np.any(am) and np.any(dm):
        a_idx = np.flatnonzero(am)
        d_idx = np.flatnonzero(dm)
        if a_idx.size and d_idx.size:
            last_assoc = int(a_idx[-1])
            first_dissoc = int(d_idx[0])
            delta = r[first_dissoc] - r[last_assoc]
            r[dm] = r[dm] - delta

    return r
