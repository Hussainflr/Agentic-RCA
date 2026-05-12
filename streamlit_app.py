from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from app.agents.workflow import run_rca_workflow
from app.core.config import settings
from app.core.safety import safety_layer
from app.core.storage import session_store
from app.data.schema import infer_dataset_readiness, suggested_record_id
from app.tools.engineering_tools import report_generation_tool
from app.ui.components import (
    dataset_dashboard,
    evaluation_dashboard,
    evidence_viewer,
    load_dataframe_from_upload_or_path,
    readiness_panel,
    report_viewer,
)


st.set_page_config(page_title="Agentic RCA Copilot", layout="wide")

st.title("Agentic Root Cause Analysis Copilot")
st.caption(
    "Industrial equipment RCA with dataset profiling, anomaly evidence, RAG evidence, LangGraph agents, verifier/refiner loop, and exportable reports."
)
st.info(safety_layer.disclaimer())

with st.sidebar:
    st.header("Configuration")
    provider = st.selectbox("Model provider", ["ollama", "openai", "anthropic", "deterministic"], index=0)
    dataset_path = st.text_input("Dataset CSV path", value=str(settings.raw_data_dir / "ai4i2020.csv"))
    upload = st.file_uploader("Or upload CSV", type=["csv"])
    user_role = st.selectbox("User role", ["maintenance_reliability_engineer", "viewer"], index=0)

    st.divider()
    st.subheader("Previous Sessions")
    sessions = session_store.list_sessions()
    selected_session = st.selectbox(
        "Reload session",
        [""] + [f"{item['updated_at']} | {item['session_id']}" for item in sessions],
    )
    if selected_session:
        sid = selected_session.split("|")[-1].strip()
        loaded = session_store.load_session(sid)
        if loaded:
            st.session_state["loaded_session"] = loaded

df, active_dataset_path = load_dataframe_from_upload_or_path(upload, dataset_path)

if df is None:
    st.warning("Provide a CSV path or upload a dataset. Run `python scripts/download_ai4i.py` to fetch AI4I.")
    st.stop()

readiness = infer_dataset_readiness(df)
suggested_id = suggested_record_id(df, readiness)

tab_setup, tab_dashboard, tab_console, tab_evidence, tab_eval, tab_sessions = st.tabs(
    ["Dataset Setup", "Dashboard", "RCA Console", "Evidence Viewer", "Evaluation", "Sessions"]
)

with tab_setup:
    st.subheader("Dataset Setup")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", len(df.columns))
    c3.metric("Numeric Sensors", len(readiness.numeric_sensor_columns))
    c4.metric("Fault Codes", len(readiness.fault_code_columns))
    readiness_panel(readiness)

    with st.expander("Detected Dataset Schema", expanded=True):
        st.json(readiness.to_dict())

    st.subheader("Data Preview")
    st.dataframe(df.head(30), use_container_width=True)

with tab_dashboard:
    dataset_dashboard(df)

with tab_console:
    st.subheader("RCA Console")
    st.write("Select a machine record, ask an RCA question, then run the multi-agent workflow.")

    selector_label = "Selected machine UDI" if "UDI" in df.columns else "Selected record index"
    record_id = st.number_input(selector_label, min_value=0, value=int(suggested_id), step=1)

    selected_preview = {}
    if "UDI" in df.columns and int(record_id) in set(df["UDI"].astype(int).tolist()):
        selected_preview = df.loc[df["UDI"].astype(int) == int(record_id)].iloc[0].to_dict()
    else:
        safe_idx = max(0, min(int(record_id), len(df) - 1))
        selected_preview = df.iloc[safe_idx].to_dict()

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Selected Record**")
        st.json({str(k): str(v) for k, v in selected_preview.items()})
    with c2:
        st.markdown("**Good RCA questions**")
        sample_question = st.radio(
            "Choose a starting point",
            [
                "Why did this equipment likely fail and what should maintenance check first?",
                "What evidence supports the most likely root cause?",
                "What alternative causes should be ruled out before repair?",
            ],
            label_visibility="collapsed",
        )

    user_query = st.text_area("Engineering question", value=sample_question, height=90)
    safety_decision = safety_layer.inspect_text(user_query)
    if not safety_decision.allowed:
        st.error("Request blocked by safety layer.")
        st.write(safety_decision.reasons)
    elif safety_decision.reasons:
        st.warning("Safety layer notice: " + "; ".join(safety_decision.reasons))
    else:
        st.success("Safety check passed.")

    run = st.button("Run Agentic RCA Workflow", type="primary", disabled=not safety_decision.allowed)

    if run:
        session_id = str(uuid.uuid4())
        with st.status("Running multi-agent RCA workflow...", expanded=True) as status:
            st.write("Interpreting selected record and sensor baselines...")
            st.write("Classifying failure and anomaly evidence...")
            st.write("Retrieving manual/SOP evidence...")
            st.write("Generating root-cause hypotheses...")
            st.write("Verifying grounding and refining if needed...")
            final_state = run_rca_workflow(
                session_id=session_id,
                dataset_path=str(active_dataset_path),
                user_query=safety_decision.sanitized_text,
                selected_record_id=int(record_id),
                provider=provider,
                user_role=user_role,
            )
            report_path = report_generation_tool(final_state, settings.exports_dir)
            session_store.save_state(final_state)
            st.session_state["final_state"] = final_state
            st.session_state["report_path"] = report_path
            status.update(label="RCA workflow completed", state="complete")

    final_state = st.session_state.get("final_state")
    if final_state:
        report = final_state.get("report", {})
        st.divider()
        st.subheader("RCA Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Root Cause", report.get("most_likely_root_cause", "Unknown"))
        c2.metric("Confidence", f"{report.get('confidence_score', 0):.2f}")
        c3.metric("Risk", report.get("risk_level", "medium"))
        st.markdown("**Recommended Checks**")
        for check in report.get("recommended_checks", []):
            st.checkbox(check, value=False)
        report_viewer(final_state, st.session_state.get("report_path"))

with tab_evidence:
    final_state = st.session_state.get("final_state")
    if final_state:
        evidence_viewer(final_state)
    else:
        st.info("Run the RCA workflow to see sensor, manual, fault-code, and agent-trace evidence.")

with tab_eval:
    final_state = st.session_state.get("final_state")
    if final_state:
        evaluation_dashboard(final_state)
    else:
        st.info("Run the RCA workflow to see groundedness, faithfulness, hallucination risk, and refinement count.")

with tab_sessions:
    st.subheader("Sessions")
    if "loaded_session" in st.session_state:
        loaded = st.session_state["loaded_session"]
        st.markdown(f"**Loaded session:** `{loaded['session_id']}`")
        st.json(loaded)
    else:
        st.dataframe(pd.DataFrame(sessions), use_container_width=True)

