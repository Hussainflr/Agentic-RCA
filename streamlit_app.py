from __future__ import annotations

import uuid

import pandas as pd
import streamlit as st

from app.agents.workflow import run_rca_workflow
from app.core.config import settings
from app.core.safety import safety_layer
from app.core.storage import session_store
from app.data.schema import infer_dataset_readiness, selected_record_preview, suggested_record_id
from app.tools.engineering_tools import report_generation_tool
from app.ui.components import (
    dataset_dashboard,
    dataset_setup_panel,
    evaluation_dashboard,
    evidence_viewer,
    inject_app_css,
    load_dataframe_from_upload_or_path,
    report_viewer,
    selected_record_panel,
    workflow_steps,
)


st.set_page_config(page_title="Agentic RCA Copilot", layout="wide")
inject_app_css()

st.title("Agentic Root Cause Analysis Copilot")
st.caption(
    "Industrial equipment RCA with sensor analytics, manual evidence, LangGraph orchestration, LangChain chains, and verifier/refiner control loops."
)

with st.sidebar:
    st.header("Workspace")
    provider = st.selectbox("Model provider", ["ollama", "openai", "anthropic", "deterministic"], index=0)
    user_role = st.selectbox("User role", ["maintenance_reliability_engineer", "viewer"], index=0)

    st.divider()
    st.header("Dataset")
    dataset_path = st.text_input("CSV path", value=str(settings.raw_data_dir / "ai4i2020.csv"))
    upload = st.file_uploader("Upload CSV", type=["csv"])

    st.divider()
    st.header("RCA Flow")
    workflow_steps()

    st.divider()
    st.header("Sessions")
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

tab_setup, tab_dashboard, tab_console, tab_evidence, tab_evaluation, tab_sessions = st.tabs(
    ["Dataset Setup", "Dashboard", "RCA Console", "Evidence", "Evaluation", "Sessions"]
)

with tab_setup:
    dataset_setup_panel(df)

with tab_dashboard:
    dataset_dashboard(df)

with tab_console:
    st.subheader("RCA Console")
    st.write("Select a machine record, ask an engineering question, and run the multi-agent RCA workflow.")

    selector_label = "Selected machine UDI" if "UDI" in df.columns else "Selected record index"
    record_id = st.number_input(selector_label, min_value=0, value=int(suggested_id), step=1)
    selected_record = selected_record_preview(df, int(record_id))

    left, right = st.columns([1, 1])
    with left:
        selected_record_panel(selected_record)
    with right:
        st.markdown("**RCA Question**")
        sample_question = st.radio(
            "Sample questions",
            [
                "Why did this equipment likely fail and what should maintenance check first?",
                "What sensor evidence supports the most likely root cause?",
                "What alternative causes should be ruled out before repair?",
            ],
            label_visibility="collapsed",
        )
        user_query = st.text_area("Engineering question", value=sample_question, height=120)

        safety_decision = safety_layer.inspect_text(user_query)
        if not safety_decision.allowed:
            st.error("Request blocked by safety layer.")
            st.write(safety_decision.reasons)
        elif safety_decision.reasons:
            st.warning("Safety notice: " + "; ".join(safety_decision.reasons))
        else:
            st.success("Safety check passed.")

        run = st.button("Run Agentic RCA Workflow", type="primary", disabled=not safety_decision.allowed)

    if run:
        session_id = str(uuid.uuid4())
        with st.status("Running RCA workflow...", expanded=True) as status:
            st.write("Interpreting dataset and selected record.")
            st.write("Classifying failure and anomaly evidence.")
            st.write("Retrieving manual/SOP context.")
            st.write("Generating RCA hypotheses.")
            st.write("Verifying grounding and refining if needed.")
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
        st.divider()
        report_viewer(final_state, st.session_state.get("report_path"))

with tab_evidence:
    final_state = st.session_state.get("final_state")
    if final_state:
        evidence_viewer(final_state)
    else:
        st.info("Run the RCA workflow to see sensor, manual/SOP, fault-code, and agent-trace evidence.")

with tab_evaluation:
    final_state = st.session_state.get("final_state")
    if final_state:
        evaluation_dashboard(final_state)
    else:
        st.info("Run the RCA workflow to see groundedness, faithfulness, hallucination risk, and refinement count.")

with tab_sessions:
    st.subheader("Session History")
    if "loaded_session" in st.session_state:
        loaded = st.session_state["loaded_session"]
        st.markdown(f"**Loaded session:** `{loaded['session_id']}`")
        st.json(loaded)
    else:
        st.dataframe(pd.DataFrame(sessions), use_container_width=True)

st.divider()
st.caption(safety_layer.disclaimer())

