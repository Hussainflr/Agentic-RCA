from __future__ import annotations

from typing import Any

from app.chains.rca_chains import (
    create_action_planner_chain,
    create_data_interpreter_chain,
    create_failure_classifier_chain,
    create_knowledge_rag_chain,
    create_refiner_chain,
    create_report_writer_chain,
    create_root_cause_chain,
    create_verifier_chain,
)
from app.core.config import settings
from app.core.hooks import hooks


def _invoke_chain(state: dict[str, Any], node_name: str, chain_factory, state_key: str | None = None):
    hooks.before_agent_run(state, node_name)
    try:
        output = chain_factory().invoke(state)
        if state_key:
            state[state_key] = output
        state.setdefault("agent_outputs", {})[node_name] = output
        hooks.after_agent_run(state, node_name, output if isinstance(output, dict) else {"output": output})
    except Exception as exc:
        hooks.on_error(state, node_name, exc)
    return state


def data_interpreter_agent(state: dict[str, Any]) -> dict[str, Any]:
    state = _invoke_chain(state, "data_interpreter", create_data_interpreter_chain, "sensor_summary")
    return state


def failure_classifier_agent(state: dict[str, Any]) -> dict[str, Any]:
    state = _invoke_chain(state, "failure_classifier", create_failure_classifier_chain, "failure_classification")
    state["anomaly_result"] = state.get("failure_classification", {}).get("anomaly", {})
    return state


def knowledge_rag_agent(state: dict[str, Any]) -> dict[str, Any]:
    state = _invoke_chain(state, "knowledge_rag", create_knowledge_rag_chain)
    output = state.get("agent_outputs", {}).get("knowledge_rag", {})
    state["retrieved_evidence"] = output.get("retrieved_evidence", [])
    return state


def root_cause_reasoning_agent(state: dict[str, Any]) -> dict[str, Any]:
    state = _invoke_chain(state, "root_cause_reasoner", create_root_cause_chain)
    output = state.get("agent_outputs", {}).get("root_cause_reasoner", {})
    state["hypotheses"] = output.get("hypotheses", [])
    return state


def verifier_agent(state: dict[str, Any]) -> dict[str, Any]:
    return _invoke_chain(state, "verifier", create_verifier_chain, "verification")


def should_refine(state: dict[str, Any]) -> str:
    verification = state.get("verification", {})
    should = verification.get("verifier_decision") == "refine"
    below_limit = state.get("refinement_count", 0) < settings.max_refinements
    if should and below_limit:
        hooks.on_refinement_trigger(state, "; ".join(verification.get("missing_evidence", [])) or "Low verifier score.")
        return "refine"
    return "continue"


def refiner_agent(state: dict[str, Any]) -> dict[str, Any]:
    state = _invoke_chain(state, "refiner", create_refiner_chain)
    output = state.get("agent_outputs", {}).get("refiner", {})
    state["refinement_count"] = output.get("refinement_count", state.get("refinement_count", 0))
    return state


def action_planner_agent(state: dict[str, Any]) -> dict[str, Any]:
    return _invoke_chain(state, "action_planner", create_action_planner_chain, "action_plan")


def report_writer_agent(state: dict[str, Any]) -> dict[str, Any]:
    state = _invoke_chain(state, "report_writer", create_report_writer_chain, "report")
    hooks.on_report_generated(state)
    return state

