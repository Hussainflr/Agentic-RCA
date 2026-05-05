from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from langchain_core.tools import tool
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from app.core.hooks import hooks
from app.core.permissions import permission_manager
from app.core.safety import safety_layer
from app.data.dataset import dataset_metadata, feature_columns, load_csv, select_record
from app.data.fault_library import AI4I_FAULT_LIBRARY


def guarded_tool_call(state: dict[str, Any], tool_name: str, args: dict[str, Any], fn):
    permission_manager.assert_can_call(state.get("user_role", "viewer"), tool_name)
    hooks.before_tool_call(state, tool_name, args)
    try:
        result = fn()
        hooks.after_tool_call(state, tool_name, result)
        return result
    except Exception as exc:
        hooks.on_error(state, tool_name, exc)
        raise


def load_dataset_tool(state: dict[str, Any]) -> dict[str, Any]:
    def _run() -> dict[str, Any]:
        df = load_csv(state["dataset_path"])
        record = select_record(df, state.get("selected_record_id"))
        meta = dataset_metadata(df, state["dataset_path"])
        return {"metadata": meta, "selected_record": record}

    return guarded_tool_call(state, "dataset_loader", {"path": state["dataset_path"]}, _run)


def statistical_summary_tool(state: dict[str, Any]) -> dict[str, Any]:
    def _run() -> dict[str, Any]:
        df = load_csv(state["dataset_path"])
        cols = feature_columns(df)
        selected = state.get("selected_record") or select_record(df, state.get("selected_record_id"))
        baseline = df[cols].describe().to_dict()
        deviations = {}
        for col in cols:
            mean = float(df[col].mean())
            std = float(df[col].std(ddof=0)) or 1.0
            value = float(selected.get(col, np.nan))
            deviations[col] = {
                "value": value,
                "mean": mean,
                "z_score": float((value - mean) / std),
                "percentile": float((df[col] <= value).mean()),
            }
        return {"feature_columns": cols, "baseline": baseline, "selected_record_deviations": deviations}

    return guarded_tool_call(state, "statistical_summary", {}, _run)


def anomaly_detection_tool(state: dict[str, Any]) -> dict[str, Any]:
    def _run() -> dict[str, Any]:
        df = load_csv(state["dataset_path"])
        cols = feature_columns(df)
        clean = df[cols].dropna()
        scaler = StandardScaler()
        matrix = scaler.fit_transform(clean)
        model = IsolationForest(n_estimators=150, contamination="auto", random_state=42)
        model.fit(matrix)
        selected = state.get("selected_record") or select_record(df, state.get("selected_record_id"))
        selected_values = pd.DataFrame([{col: selected.get(col, clean[col].median()) for col in cols}])
        selected_scaled = scaler.transform(selected_values)
        raw_score = float(model.decision_function(selected_scaled)[0])
        anomaly_label = bool(model.predict(selected_scaled)[0] == -1)
        deviations = statistical_summary_tool(state)["selected_record_deviations"]
        contributing = sorted(
            [{"feature": k, "z_score": v["z_score"], "value": v["value"]} for k, v in deviations.items()],
            key=lambda item: abs(item["z_score"]),
            reverse=True,
        )[:3]
        normalized = float(max(0.0, min(1.0, 0.5 - raw_score)))
        return {
            "anomaly_label": anomaly_label,
            "anomaly_score": normalized,
            "raw_isolation_score": raw_score,
            "contributing_features": contributing,
        }

    return guarded_tool_call(state, "anomaly_detector", {}, _run)


def fault_lookup_tool(state: dict[str, Any]) -> dict[str, Any]:
    def _run() -> dict[str, Any]:
        selected = state.get("selected_record", {})
        active_codes = []
        for code, info in AI4I_FAULT_LIBRARY.items():
            if int(selected.get(code, 0) or 0) == 1:
                active_codes.append({"code": code, **info})
        if not active_codes and int(selected.get("Machine failure", 0) or 0) == 1:
            active_codes.append({"code": "UNKNOWN", "name": "Unlabeled Machine Failure", "description": "Failure label present without specific AI4I failure code.", "checks": ["Review full sensor history and maintenance log"]})
        return {"active_faults": active_codes, "library": AI4I_FAULT_LIBRARY}

    return guarded_tool_call(state, "fault_lookup", {}, _run)


def manual_rag_search_tool(state: dict[str, Any], query: str, manuals_dir: Path) -> list[dict[str, Any]]:
    def _run() -> list[dict[str, Any]]:
        safe = safety_layer.inspect_text(query)
        if not safe.allowed:
            return [{"source": "safety", "kind": "blocked", "content": "; ".join(safe.reasons), "score": 1.0}]
        docs = []
        for path in manuals_dir.glob("*.md"):
            text = path.read_text(encoding="utf-8")
            docs.append((path.name, text))
        terms = {term.lower() for term in safe.sanitized_text.replace("/", " ").split() if len(term) > 2}
        results = []
        for name, text in docs:
            lower = text.lower()
            score = sum(1 for term in terms if term in lower) / max(len(terms), 1)
            if score > 0:
                snippet = text[:1200]
                results.append({"source": name, "kind": "manual", "content": snippet, "score": float(score)})
        return sorted(results, key=lambda item: item["score"], reverse=True)[:5]

    return guarded_tool_call(state, "manual_rag_search", {"query": query}, _run)


def chart_generation_tool(state: dict[str, Any], output_dir: Path) -> dict[str, str]:
    def _run() -> dict[str, str]:
        output_dir.mkdir(parents=True, exist_ok=True)
        df = load_csv(state["dataset_path"])
        paths: dict[str, str] = {}
        if "Machine failure" in df.columns:
            fig = px.histogram(df, x="Machine failure", title="Machine Failure Distribution")
            path = output_dir / f"{state['session_id']}_failure_distribution.html"
            fig.write_html(path)
            paths["failure_distribution"] = str(path)
        cols = feature_columns(df)
        fig = px.line(df.reset_index().head(500), x="index", y=cols[:3], title="Sensor Trends")
        path = output_dir / f"{state['session_id']}_sensor_trends.html"
        fig.write_html(path)
        paths["sensor_trends"] = str(path)
        return paths

    return guarded_tool_call(state, "chart_generator", {}, _run)


def report_generation_tool(state: dict[str, Any], output_dir: Path) -> str:
    def _run() -> str:
        output_dir.mkdir(parents=True, exist_ok=True)
        report = state.get("report", {})
        path = output_dir / f"{state['session_id']}_rca_report.md"
        lines = [
            "# Root Cause Analysis Report",
            "",
            f"**Most likely root cause:** {report.get('most_likely_root_cause', 'Unknown')}",
            f"**Confidence:** {report.get('confidence_score', 0):.2f}",
            f"**Risk level:** {report.get('risk_level', 'medium')}",
            "",
            "## Dataset Evidence",
            *[f"- {item}" for item in report.get("evidence_from_dataset", [])],
            "",
            "## Manual / SOP Evidence",
            *[f"- {item}" for item in report.get("evidence_from_manuals", [])],
            "",
            "## Alternative Causes",
            *[f"- {item}" for item in report.get("alternative_causes", [])],
            "",
            "## Recommended Checks",
            *[f"- {item}" for item in report.get("recommended_checks", [])],
            "",
            "## Safety Notes",
            *[f"- {item}" for item in report.get("safety_notes", [])],
            "",
            "## Final Action Plan",
            *[f"- {item}" for item in report.get("final_action_plan", [])],
            "",
            f"_{report.get('disclaimer', safety_layer.disclaimer())}_",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")
        return str(path)

    return guarded_tool_call(state, "report_generator", {}, _run)


@tool
def dataset_loader(path: str) -> str:
    """Load a predictive maintenance CSV and return dataset metadata."""
    df = load_csv(path)
    return json.dumps(dataset_metadata(df, path))

