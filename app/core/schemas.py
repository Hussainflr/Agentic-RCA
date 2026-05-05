from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, TypedDict

from typing_extensions import NotRequired

from pydantic import BaseModel, Field


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EvidenceItem(BaseModel):
    source: str
    kind: str
    content: str
    score: float = 0.0


class AgentTrace(BaseModel):
    name: str
    started_at: str = Field(default_factory=utc_now)
    completed_at: str | None = None
    input_summary: str = ""
    output_summary: str = ""
    status: Literal["started", "completed", "error"] = "started"
    error: str | None = None


class VerificationResult(BaseModel):
    groundedness_score: float = 0.0
    faithfulness_score: float = 0.0
    hallucination_risk: Literal["low", "medium", "high"] = "medium"
    verifier_decision: Literal["accept", "refine", "reject"] = "refine"
    missing_evidence: list[str] = Field(default_factory=list)


class RCAReport(BaseModel):
    most_likely_root_cause: str
    confidence_score: float
    evidence_from_dataset: list[str]
    evidence_from_manuals: list[str]
    alternative_causes: list[str]
    recommended_checks: list[str]
    risk_level: Literal["low", "medium", "high", "critical"]
    safety_notes: list[str]
    final_action_plan: list[str]
    disclaimer: str


class RCAState(TypedDict):
    session_id: str
    user_role: str
    provider: str
    task_type: str
    user_query: str
    dataset_path: str
    selected_record_id: int | None
    selected_record: dict[str, Any]
    dataset_metadata: dict[str, Any]
    sensor_summary: dict[str, Any]
    anomaly_result: dict[str, Any]
    failure_classification: dict[str, Any]
    retrieved_evidence: list[dict[str, Any]]
    hypotheses: list[dict[str, Any]]
    verification: dict[str, Any]
    action_plan: dict[str, Any]
    report: dict[str, Any]
    agent_outputs: dict[str, Any]
    traces: list[dict[str, Any]]
    errors: list[str]
    refinement_count: int
    available_skills: NotRequired[list[dict[str, Any]]]


def empty_state(
    session_id: str,
    dataset_path: str,
    user_query: str,
    selected_record_id: int | None,
    provider: str,
    user_role: str = "maintenance_reliability_engineer",
) -> RCAState:
    return {
        "session_id": session_id,
        "user_role": user_role,
        "provider": provider,
        "task_type": "root_cause_analysis",
        "user_query": user_query,
        "dataset_path": dataset_path,
        "selected_record_id": selected_record_id,
        "selected_record": {},
        "dataset_metadata": {},
        "sensor_summary": {},
        "anomaly_result": {},
        "failure_classification": {},
        "retrieved_evidence": [],
        "hypotheses": [],
        "verification": {},
        "action_plan": {},
        "report": {},
        "agent_outputs": {},
        "traces": [],
        "errors": [],
        "refinement_count": 0,
        "available_skills": [],
    }
