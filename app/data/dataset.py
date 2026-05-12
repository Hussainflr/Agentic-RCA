from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from app.data.schema import infer_dataset_readiness


AI4I_FEATURE_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(col).strip() for col in df.columns]
    return df


def dataset_metadata(df: pd.DataFrame, path: str | Path | None = None) -> dict[str, Any]:
    failure_col = "Machine failure" if "Machine failure" in df.columns else None
    readiness = infer_dataset_readiness(df).to_dict()
    return {
        "path": str(path) if path else "",
        "rows": int(len(df)),
        "columns": list(df.columns),
        "numeric_columns": list(df.select_dtypes("number").columns),
        "failure_rate": float(df[failure_col].mean()) if failure_col else None,
        "failure_count": int(df[failure_col].sum()) if failure_col else None,
        "readiness": readiness,
    }


def select_record(df: pd.DataFrame, record_id: int | None) -> dict[str, Any]:
    if df.empty:
        return {}
    if record_id is None:
        row = df.iloc[0]
    elif "UDI" in df.columns and record_id in set(df["UDI"].astype(int).tolist()):
        row = df.loc[df["UDI"].astype(int) == int(record_id)].iloc[0]
    else:
        safe_idx = max(0, min(int(record_id), len(df) - 1))
        row = df.iloc[safe_idx]
    return row.to_dict()


def feature_columns(df: pd.DataFrame) -> list[str]:
    ai4i = [col for col in AI4I_FEATURE_COLUMNS if col in df.columns]
    if ai4i:
        return ai4i
    readiness = infer_dataset_readiness(df)
    if readiness.numeric_sensor_columns:
        return readiness.numeric_sensor_columns[:8]
    return list(df.select_dtypes("number").columns[:8])
