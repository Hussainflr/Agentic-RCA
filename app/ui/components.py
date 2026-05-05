from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from app.data.dataset import feature_columns, load_csv


def dataset_dashboard(df: pd.DataFrame) -> None:
    st.subheader("Dashboard")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", len(df.columns))
    if "Machine failure" in df.columns:
        c3.metric("Failure Rate", f"{df['Machine failure'].mean() * 100:.2f}%")
    else:
        c3.metric("Failure Rate", "N/A")

    if "Machine failure" in df.columns:
        st.plotly_chart(px.histogram(df, x="Machine failure", title="Failure Distribution"), use_container_width=True)

    cols = feature_columns(df)
    if cols:
        st.plotly_chart(
            px.line(df.reset_index().head(500), x="index", y=cols[:5], title="Sensor Trends - First 500 Rows"),
            use_container_width=True,
        )


def evidence_viewer(final_state: dict) -> None:
    st.subheader("Evidence Viewer")
    tab1, tab2, tab3, tab4 = st.tabs(["Sensor Evidence", "Manual/SOP Evidence", "Fault Evidence", "Agent Trace"])
    with tab1:
        st.json(final_state.get("sensor_summary", {}).get("selected_record_deviations", {}))
    with tab2:
        for item in final_state.get("retrieved_evidence", []):
            st.markdown(f"**{item.get('source')}** | score `{item.get('score', 0):.2f}`")
            st.write(item.get("content", "")[:900])
    with tab3:
        st.json(final_state.get("failure_classification", {}))
    with tab4:
        traces = final_state.get("traces", [])
        st.dataframe(pd.DataFrame(traces) if traces else pd.DataFrame())


def evaluation_dashboard(final_state: dict) -> None:
    st.subheader("Evaluation Dashboard")
    verification = final_state.get("verification", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Groundedness", f"{verification.get('groundedness_score', 0):.2f}")
    c2.metric("Faithfulness", f"{verification.get('faithfulness_score', 0):.2f}")
    c3.metric("Hallucination Risk", verification.get("hallucination_risk", "unknown"))
    c4.metric("Refinements", final_state.get("refinement_count", 0))
    st.write("Verifier decision:", verification.get("verifier_decision", "unknown"))
    if verification.get("missing_evidence"):
        st.warning("Missing evidence: " + "; ".join(verification["missing_evidence"]))


def report_viewer(final_state: dict, report_path: str | None = None) -> None:
    st.subheader("RCA Report")
    report = final_state.get("report", {})
    st.markdown(f"### {report.get('most_likely_root_cause', 'No report yet')}")
    st.progress(float(report.get("confidence_score", 0)))
    st.write("Risk level:", report.get("risk_level", "unknown"))
    for section, key in [
        ("Dataset Evidence", "evidence_from_dataset"),
        ("Manual/SOP Evidence", "evidence_from_manuals"),
        ("Alternative Causes", "alternative_causes"),
        ("Recommended Checks", "recommended_checks"),
        ("Safety Notes", "safety_notes"),
        ("Final Action Plan", "final_action_plan"),
    ]:
        st.markdown(f"**{section}**")
        for item in report.get(key, []):
            st.markdown(f"- {item}")
    st.info(report.get("disclaimer", "Decision-support only."))
    if report_path and Path(report_path).exists():
        st.download_button(
            "Download Markdown Report",
            data=Path(report_path).read_text(encoding="utf-8"),
            file_name=Path(report_path).name,
            mime="text/markdown",
        )


def load_dataframe_from_upload_or_path(upload, path: str | None) -> tuple[pd.DataFrame | None, str | None]:
    if upload is not None:
        temp_path = Path("data/raw") / upload.name
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_bytes(upload.getvalue())
        return load_csv(temp_path), str(temp_path)
    if path and Path(path).exists():
        return load_csv(path), path
    return None, None

