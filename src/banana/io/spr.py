"""Surface Plasmon Resonance (SPR) data import - placeholder for future support."""

from pathlib import Path
from typing import Optional
from typing import Union

import pandas as pd


def load_spr_csv(
    file_path: Union[str, Path],
    time_col: str = "Time",
    response_col: str = "Response",
    concentration_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load SPR data from a generic CSV file.

    Many SPR instruments export CSV with time and response columns.
    This provides a simple loader for such formats.

    Parameters
    ----------
    file_path : str or Path
        Path to CSV file.
    time_col : str
        Column name for time (seconds).
    response_col : str
        Column name for response (RU or similar).
    concentration_col : str, optional
        If present, column for analyte concentration.

    Returns
    -------
    pd.DataFrame
        Standardized columns: Time (sec), Response, Concentration (if available).
    """
    df = pd.read_csv(file_path)
    out = pd.DataFrame()
    out["Time (sec)"] = df[time_col]
    out["Binding (nm)"] = df[response_col]  # Use same column name as BLI for consistency
    if concentration_col and concentration_col in df.columns:
        out["Concentration"] = df[concentration_col]
    return out


