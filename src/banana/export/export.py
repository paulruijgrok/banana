"""Export kinetic fit results to CSV and other formats."""

import logging
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from banana.fitting.fit import FitResult
from banana.io.titration import TitrationSeries

logger = logging.getLogger(__name__)


def export_results(
    global_result: FitResult,
    trace_results: Optional[List[FitResult]] = None,
    output_path: Union[str, Path] = "kinetic_results.csv",
) -> pd.DataFrame:
    """
    Export fit results to CSV. Safe if to_dict fails on any row.
    """
    rows = []
    for label, res in [("global", global_result)]:
        try:
            row = res.to_dict()
            row["type"] = label
            rows.append(row)
        except Exception as e:
            logger.warning("export_results global row failed: %s", e)
            rows.append({"type": "global", "success": False, "message": str(e)})

    if trace_results:
        for i, res in enumerate(trace_results):
            try:
                r = res.to_dict()
                r["trace"] = i
                r["type"] = "trace"
                rows.append(r)
            except Exception as e:
                logger.warning("export_results trace %s failed: %s", i, e)
                rows.append({"type": "trace", "trace": i, "success": False, "message": str(e)})

    df = pd.DataFrame(rows)
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
    except Exception as e:
        logger.error("export_results CSV write failed: %s", e)
    return df


def export_titration_data(
    titration: TitrationSeries,
    output_path: Union[str, Path] = "titration_data.csv",
) -> pd.DataFrame:
    try:
        df = titration.to_dataframe()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
    except Exception as e:
        logger.error("export_titration_data failed: %s", e)
        return pd.DataFrame()
    return df
