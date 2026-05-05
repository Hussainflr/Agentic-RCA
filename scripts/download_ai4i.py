from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/raw/ai4i2020.csv"
URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv",
    "https://raw.githubusercontent.com/plotly/datasets/master/ai4i2020.csv",
]


def create_demo_fallback() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "UDI": 1,
            "Product ID": "M14860",
            "Type": "M",
            "Air temperature [K]": 298.1,
            "Process temperature [K]": 308.6,
            "Rotational speed [rpm]": 1551,
            "Torque [Nm]": 42.8,
            "Tool wear [min]": 0,
            "Machine failure": 0,
            "TWF": 0,
            "HDF": 0,
            "PWF": 0,
            "OSF": 0,
            "RNF": 0,
        },
        {
            "UDI": 79,
            "Product ID": "L47258",
            "Type": "L",
            "Air temperature [K]": 298.9,
            "Process temperature [K]": 309.1,
            "Rotational speed [rpm]": 1350,
            "Torque [Nm]": 68.2,
            "Tool wear [min]": 215,
            "Machine failure": 1,
            "TWF": 0,
            "HDF": 0,
            "PWF": 0,
            "OSF": 1,
            "RNF": 0,
        },
        {
            "UDI": 160,
            "Product ID": "H29510",
            "Type": "H",
            "Air temperature [K]": 303.4,
            "Process temperature [K]": 312.9,
            "Rotational speed [rpm]": 1412,
            "Torque [Nm]": 53.6,
            "Tool wear [min]": 74,
            "Machine failure": 1,
            "TWF": 0,
            "HDF": 1,
            "PWF": 0,
            "OSF": 0,
            "RNF": 0,
        },
    ]
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"Created demo fallback dataset at {OUT}")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for url in URLS:
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            OUT.write_bytes(response.content)
            print(f"Downloaded AI4I dataset to {OUT}")
            return
        except Exception as exc:
            print(f"Could not download from {url}: {exc}")
    create_demo_fallback()


if __name__ == "__main__":
    main()

