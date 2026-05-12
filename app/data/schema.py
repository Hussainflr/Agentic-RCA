from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


ID_HINTS = ("id", "udi", "record", "asset", "machine", "product")
TIME_HINTS = ("time", "date", "timestamp")
FAILURE_HINTS = ("failure", "failed", "target", "label", "fault")
FAULT_CODE_COLUMNS = {"TWF", "HDF", "PWF", "OSF", "RNF"}


@dataclass
class DatasetReadiness:
    id_columns: list[str]
    timestamp_columns: list[str]
    numeric_sensor_columns: list[str]
    failure_label_columns: list[str]
    fault_code_columns: list[str]
    categorical_columns: list[str]
    readiness_score: float
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id_columns": self.id_columns,
            "timestamp_columns": self.timestamp_columns,
            "numeric_sensor_columns": self.numeric_sensor_columns,
            "failure_label_columns": self.failure_label_columns,
            "fault_code_columns": self.fault_code_columns,
            "categorical_columns": self.categorical_columns,
            "readiness_score": self.readiness_score,
            "notes": self.notes,
        }


def _has_hint(column: str, hints: tuple[str, ...]) -> bool:
    normalized = column.lower().replace(" ", "_")
    return any(hint in normalized for hint in hints)


def infer_dataset_readiness(df: pd.DataFrame) -> DatasetReadiness:
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = df.select_dtypes(exclude="number").columns.tolist()
    id_columns = [column for column in df.columns if _has_hint(column, ID_HINTS)]
    timestamp_columns = [column for column in df.columns if _has_hint(column, TIME_HINTS)]
    fault_code_columns = [column for column in df.columns if column in FAULT_CODE_COLUMNS or column.lower() in {"fault_code", "fault_type"}]
    failure_label_columns = [
        column for column in df.columns if _has_hint(column, FAILURE_HINTS) and column not in fault_code_columns
    ]

    excluded = set(id_columns + failure_label_columns + fault_code_columns)
    numeric_sensor_columns = [column for column in numeric_columns if column not in excluded]

    score = 0.0
    notes: list[str] = []
    if numeric_sensor_columns:
        score += 0.35
        notes.append(f"Detected {len(numeric_sensor_columns)} numeric sensor columns.")
    else:
        notes.append("No numeric sensor columns detected; anomaly detection will be limited.")
    if failure_label_columns:
        score += 0.2
        notes.append(f"Detected failure label columns: {', '.join(failure_label_columns)}.")
    else:
        notes.append("No failure label detected; RCA will rely on anomaly/manual evidence.")
    if fault_code_columns:
        score += 0.2
        notes.append(f"Detected fault-code columns: {', '.join(fault_code_columns)}.")
    else:
        notes.append("No explicit fault-code columns detected.")
    if id_columns:
        score += 0.1
        notes.append(f"Detected identifier columns: {', '.join(id_columns[:3])}.")
    if timestamp_columns:
        score += 0.1
        notes.append(f"Detected timestamp columns: {', '.join(timestamp_columns[:3])}.")
    if len(df) >= 100:
        score += 0.05
    else:
        notes.append("Dataset is small; statistical baselines may be weak.")

    return DatasetReadiness(
        id_columns=id_columns,
        timestamp_columns=timestamp_columns,
        numeric_sensor_columns=numeric_sensor_columns,
        failure_label_columns=failure_label_columns,
        fault_code_columns=fault_code_columns,
        categorical_columns=categorical_columns,
        readiness_score=round(min(score, 1.0), 2),
        notes=notes,
    )


def suggested_record_id(df: pd.DataFrame, readiness: DatasetReadiness) -> int:
    if "UDI" in df.columns and "Machine failure" in df.columns:
        failed = df[df["Machine failure"] == 1]
        if not failed.empty:
            return int(failed["UDI"].iloc[0])
    for column in readiness.failure_label_columns + readiness.fault_code_columns:
        if column in df.columns:
            failed = df[pd.to_numeric(df[column], errors="coerce").fillna(0) == 1]
            if not failed.empty:
                if "UDI" in df.columns:
                    return int(failed["UDI"].iloc[0])
                return int(failed.index[0])
    return 0

