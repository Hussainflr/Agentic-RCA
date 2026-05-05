from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from app.agents.agent_nodes import (
    action_planner_agent,
    data_interpreter_agent,
    failure_classifier_agent,
    knowledge_rag_agent,
    refiner_agent,
    report_writer_agent,
    root_cause_reasoning_agent,
    should_refine,
    verifier_agent,
)
from app.core.config import settings
from app.core.schemas import RCAState, empty_state
from app.core.skills_registry import load_skills
from app.core.storage import session_store


def build_rca_graph():
    graph = StateGraph(RCAState)
    graph.add_node("data_interpreter", data_interpreter_agent)
    graph.add_node("failure_classifier", failure_classifier_agent)
    graph.add_node("knowledge_rag", knowledge_rag_agent)
    graph.add_node("root_cause_reasoner", root_cause_reasoning_agent)
    graph.add_node("verifier", verifier_agent)
    graph.add_node("refiner", refiner_agent)
    graph.add_node("action_planner", action_planner_agent)
    graph.add_node("report_writer", report_writer_agent)

    graph.set_entry_point("data_interpreter")
    graph.add_edge("data_interpreter", "failure_classifier")
    graph.add_edge("failure_classifier", "knowledge_rag")
    graph.add_edge("knowledge_rag", "root_cause_reasoner")
    graph.add_edge("root_cause_reasoner", "verifier")
    graph.add_conditional_edges("verifier", should_refine, {"refine": "refiner", "continue": "action_planner"})
    graph.add_edge("refiner", "root_cause_reasoner")
    graph.add_edge("action_planner", "report_writer")
    graph.add_edge("report_writer", END)
    return graph.compile()


def run_rca_workflow(
    session_id: str,
    dataset_path: str,
    user_query: str,
    selected_record_id: int | None,
    provider: str,
    user_role: str = "maintenance_reliability_engineer",
) -> dict[str, Any]:
    state = empty_state(
        session_id=session_id,
        dataset_path=dataset_path,
        user_query=user_query,
        selected_record_id=selected_record_id,
        provider=provider,
        user_role=user_role,
    )
    state["available_skills"] = load_skills(settings.skills_path)
    app = build_rca_graph()
    final_state = app.invoke(state)
    session_store.save_state(final_state)
    return final_state

