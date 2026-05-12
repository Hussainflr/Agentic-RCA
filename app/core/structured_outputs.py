from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SensorSummaryOutput(BaseModel):
    feature_columns: list[str] = Field(default_factory=list)
    baseline: dict[str, Any] = Field(default_factory=dict)
    selected_record_deviations: dict[str, Any] = Field(default_factory=dict)


class FailureClassificationOutput(BaseModel):
    primary_failure_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    active_faults: list[dict[str, Any]] = Field(default_factory=list)
    anomaly: dict[str, Any] = Field(default_factory=dict)


class KnowledgeRetrievalOutput(BaseModel):
    retrieved_evidence: list[dict[str, Any]] = Field(default_factory=list)


class RootCauseHypothesis(BaseModel):
    rank: int
    hypothesis: str
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class RootCauseOutput(BaseModel):
    hypotheses: list[RootCauseHypothesis]


class VerificationOutput(BaseModel):
    groundedness_score: float = Field(ge=0.0, le=1.0)
    faithfulness_score: float = Field(ge=0.0, le=1.0)
    hallucination_risk: Literal["low", "medium", "high"]
    verifier_decision: Literal["accept", "refine", "reject"]
    missing_evidence: list[str] = Field(default_factory=list)


class RefinementOutput(BaseModel):
    refinement_count: int
    missing_evidence_addressed: list[str] = Field(default_factory=list)


class ActionPlanOutput(BaseModel):
    recommended_checks: list[str] = Field(default_factory=list)
    immediate_actions: list[str] = Field(default_factory=list)
    follow_up_actions: list[str] = Field(default_factory=list)
    escalation_criteria: list[str] = Field(default_factory=list)
    risk_level: Literal["low", "medium", "high", "critical"] = "medium"


class ReportOutput(BaseModel):
    most_likely_root_cause: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    evidence_from_dataset: list[str] = Field(default_factory=list)
    evidence_from_manuals: list[str] = Field(default_factory=list)
    alternative_causes: list[str] = Field(default_factory=list)
    recommended_checks: list[str] = Field(default_factory=list)
    risk_level: Literal["low", "medium", "high", "critical"] = "medium"
    safety_notes: list[str] = Field(default_factory=list)
    final_action_plan: list[str] = Field(default_factory=list)
    disclaimer: str

