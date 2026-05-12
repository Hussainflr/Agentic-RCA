from __future__ import annotations

import uuid
from pathlib import Path

import streamlit as st

from app.agents.workflow import run_rca_workflow
from app.core.config import settings
from app.core.safety import safety_layer
from app.core.storage import session_store
from app.tools.engineering_tools import report_generation_tool
from app.ui.components import (
    dataset_dashboard,
    evaluation_dashboard,
    evidence_viewer,
    load_dataframe_from_upload_or_path,
    report_viewer,
)


st.set_page_config(page_title="Agentic RCA Copilot", layout="wide")

st.title("Agentic Root Cause Analysis Copilot")
st.caption("Industrial equipment fault diagnosis with sensor analytics, RAG evidence, LangGraph agents, and verifier/refiner control loops.")

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
    st.warning("Provide a CSV path or upload a dataset. Run `python scripts/download_ai4i.py` to fetch AI4I or create a demo fallback.")
    st.stop()

dataset_dashboard(df)

st.divider()
st.subheader("RCA Console")

default_failed = None
if "Machine failure" in df.columns and "UDI" in df.columns and not df[df["Machine failure"] == 1].empty:
    default_failed = int(df[df["Machine failure"] == 1]["UDI"].iloc[0])
record_id = st.number_input("Selected machine record / UDI", min_value=0, value=int(default_failed or 0), step=1)
user_query = st.text_area(
    "Engineering question",
    value="Why did this equipment likely fail and what should maintenance check first?",
    height=90,
)
safety_decision = safety_layer.inspect_text(user_query)
if not safety_decision.allowed:
    st.error("Request blocked by safety layer.")
    st.write(safety_decision.reasons)
elif safety_decision.reasons:
    st.warning("Safety notice: " + "; ".join(safety_decision.reasons))
else:
    st.success("Safety check passed.")

run = st.button("Run Multi-Agent RCA Workflow", type="primary", disabled=not safety_decision.allowed)

if run:
    session_id = str(uuid.uuid4())
    with st.spinner("Running LangGraph RCA workflow..."):
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

if "loaded_session" in st.session_state:
    st.info("Loaded previous session summary.")
    st.json(st.session_state["loaded_session"])

final_state = st.session_state.get("final_state")
if final_state:
    report_viewer(final_state, st.session_state.get("report_path"))
    evidence_viewer(final_state)
    evaluation_dashboard(final_state)

st.divider()
st.caption(
    "Decision-support only. Not final engineering certification. Follow site safety rules, lockout/tagout, qualified inspection requirements, and OEM documentation."
)
