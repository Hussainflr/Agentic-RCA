from __future__ import annotations

from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from app.core.config import settings
from app.core.prompts import build_agent_prompt
from app.core.providers import get_llm, invoke_text
from app.core.safety import safety_layer
from app.core.structured_outputs import (
    ActionPlanOutput,
    FailureClassificationOutput,
    KnowledgeRetrievalOutput,
    RefinementOutput,
    ReportOutput,
    RootCauseHypothesis,
    RootCauseOutput,
    SensorSummaryOutput,
    VerificationOutput,
)
from app.data.dataset import load_csv
from app.data.schema import infer_dataset_readiness
from app.tools.engineering_tools import (
    anomaly_detection_tool,
    fault_lookup_tool,
    load_dataset_tool,
    manual_rag_search_tool,
    statistical_summary_tool,
)


def _dump(model) -> dict[str, Any]:
    return model.model_dump()


def _format_deviation_evidence(deviations: dict[str, Any], limit: int = 3) -> list[str]:
    top_features = sorted(
        deviations.items(),
        key=lambda item: abs(item[1].get("z_score", 0)),
        reverse=True,
    )[:limit]
    return [
        f"{name}: value={vals.get('value'):.2f}, z={vals.get('z_score'):.2f}, percentile={vals.get('percentile'):.2f}"
        for name, vals in top_features
    ]


def create_data_interpreter_chain():
    def _run(state: dict[str, Any]) -> dict[str, Any]:
        loaded = load_dataset_tool(state)
        state["dataset_metadata"] = loaded["metadata"]
        state["selected_record"] = loaded["selected_record"]

        df = load_csv(state["dataset_path"])
        readiness = infer_dataset_readiness(df).to_dict()
        state["dataset_metadata"]["readiness"] = readiness

        summary = statistical_summary_tool(state)
        output = SensorSummaryOutput(**summary)
        return _dump(output)

    return RunnableLambda(_run).with_config(run_name="data_interpreter_chain")


def create_failure_classifier_chain():
    def _run(state: dict[str, Any]) -> dict[str, Any]:
        anomaly = anomaly_detection_tool(state)
        fault_lookup = fault_lookup_tool(state)
        active = fault_lookup["active_faults"]
        if active:
            primary = active[0]["name"]
            confidence = 0.84 if state["selected_record"].get("Machine failure", 0) == 1 else 0.62
        elif anomaly["anomaly_label"]:
            primary = "Unlabeled anomalous operating condition"
            confidence = 0.56
        else:
            primary = "No clear failure signature"
            confidence = 0.35

        output = FailureClassificationOutput(
            primary_failure_type=primary,
            confidence=confidence,
            active_faults=active,
            anomaly=anomaly,
        )
        return _dump(output)

    return RunnableLambda(_run).with_config(run_name="failure_classifier_chain")


def create_knowledge_rag_chain():
    def _run(state: dict[str, Any]) -> dict[str, Any]:
        classification = state.get("failure_classification", {})
        query = " ".join(
            [
                state.get("user_query", ""),
                classification.get("primary_failure_type", ""),
                " ".join(item.get("name", "") for item in classification.get("active_faults", [])),
                " ".join(item.get("feature", "") for item in classification.get("anomaly", {}).get("contributing_features", [])),
            ]
        )
        evidence = manual_rag_search_tool(state, query=query, manuals_dir=settings.manuals_dir)
        return _dump(KnowledgeRetrievalOutput(retrieved_evidence=evidence))

    return RunnableLambda(_run).with_config(run_name="knowledge_rag_chain")


def create_root_cause_chain():
    def _run(state: dict[str, Any]) -> dict[str, Any]:
        llm = get_llm(state.get("provider"))
        system_prompt = build_agent_prompt(
            "root_cause_reasoner",
            state,
            retrieved_context="\n\n".join(item.get("content", "")[:600] for item in state.get("retrieved_evidence", [])),
            tools=["statistical_summary", "fault_lookup", "manual_rag_search"],
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                (
                    "human",
                    "Use the structured evidence to produce ranked RCA hypotheses. "
                    "Return only evidence-supported claims.",
                ),
            ]
        )
        rendered = prompt.format()
        _ = invoke_text(llm, rendered)

        classification = state.get("failure_classification", {})
        deviations = state.get("sensor_summary", {}).get("selected_record_deviations", {})
        evidence = _format_deviation_evidence(deviations)
        active_faults = classification.get("active_faults", [])

        if active_faults:
            root = active_faults[0]["name"]
            rationale = active_faults[0]["description"]
        else:
            root = classification.get("primary_failure_type", "Unclear condition")
            rationale = "No explicit failure-code label was available; hypothesis is based on anomaly and sensor deviations."

        hypotheses = [
            RootCauseHypothesis(
                rank=1,
                hypothesis=root,
                supporting_evidence=evidence + [rationale],
                contradicting_evidence=[],
                confidence=classification.get("confidence", 0.45),
            )
        ]
        for fault in active_faults[1:]:
            hypotheses.append(
                RootCauseHypothesis(
                    rank=len(hypotheses) + 1,
                    hypothesis=fault["name"],
                    supporting_evidence=[fault["description"]],
                    contradicting_evidence=["Lower priority than primary active label."],
                    confidence=0.48,
                )
            )
        if len(hypotheses) == 1:
            hypotheses.append(
                RootCauseHypothesis(
                    rank=2,
                    hypothesis="Sensor or process drift",
                    supporting_evidence=evidence,
                    contradicting_evidence=["Requires maintenance logs or time-series confirmation."],
                    confidence=0.38,
                )
            )

        return _dump(RootCauseOutput(hypotheses=hypotheses))

    return RunnableLambda(_run).with_config(run_name="root_cause_chain")


def create_verifier_chain():
    def _run(state: dict[str, Any]) -> dict[str, Any]:
        hypotheses = state.get("hypotheses", [])
        has_dataset_evidence = bool(state.get("sensor_summary", {}).get("selected_record_deviations"))
        has_manual_evidence = bool(state.get("retrieved_evidence"))
        has_fault_evidence = bool(state.get("failure_classification", {}).get("active_faults"))
        groundedness = 0.35 + (0.25 if has_dataset_evidence else 0) + (0.2 if has_manual_evidence else 0) + (0.15 if has_fault_evidence else 0)
        faithfulness = 0.45 + (0.25 if hypotheses else 0) + (0.15 if has_dataset_evidence else 0) + (0.1 if has_fault_evidence else 0)
        groundedness = min(1.0, groundedness)
        faithfulness = min(1.0, faithfulness)
        missing = []
        if not has_manual_evidence:
            missing.append("No manual/SOP evidence retrieved.")
        if not has_fault_evidence:
            missing.append("No explicit fault-code evidence for selected record.")
        decision = "accept"
        if groundedness < settings.grounding_threshold or faithfulness < settings.faithfulness_threshold:
            decision = "refine"
        risk = "low" if groundedness > 0.8 and faithfulness > 0.8 else "medium"
        if groundedness < 0.55:
            risk = "high"
        return _dump(
            VerificationOutput(
                groundedness_score=groundedness,
                faithfulness_score=faithfulness,
                hallucination_risk=risk,
                verifier_decision=decision,
                missing_evidence=missing,
            )
        )

    return RunnableLambda(_run).with_config(run_name="verifier_chain")


def create_refiner_chain():
    def _run(state: dict[str, Any]) -> dict[str, Any]:
        state["refinement_count"] = state.get("refinement_count", 0) + 1
        missing = state.get("verification", {}).get("missing_evidence", [])
        if missing:
            query = state.get("failure_classification", {}).get("primary_failure_type", "") + " maintenance SOP troubleshooting"
            extra = manual_rag_search_tool(state, query=query, manuals_dir=settings.manuals_dir)
            seen = {item.get("source") + item.get("content", "")[:30] for item in state.get("retrieved_evidence", [])}
            for item in extra:
                key = item.get("source") + item.get("content", "")[:30]
                if key not in seen:
                    state.setdefault("retrieved_evidence", []).append(item)
        for hypothesis in state.get("hypotheses", []):
            hypothesis.setdefault("contradicting_evidence", []).append(
                "Verifier requested refinement; confidence remains conditional on available evidence."
            )
            hypothesis["confidence"] = max(0.25, float(hypothesis.get("confidence", 0.5)) - 0.03)
        return _dump(RefinementOutput(refinement_count=state["refinement_count"], missing_evidence_addressed=missing))

    return RunnableLambda(_run).with_config(run_name="refiner_chain")


def create_action_planner_chain():
    def _run(state: dict[str, Any]) -> dict[str, Any]:
        faults = state.get("failure_classification", {}).get("active_faults", [])
        checks = []
        for fault in faults:
            checks.extend(fault.get("checks", []))
        if not checks:
            checks = ["Review sensor calibration", "Inspect recent maintenance logs", "Capture additional time-series data"]
        checks = safety_layer.filter_engineering_advice(checks)
        risk_level = "high" if state.get("selected_record", {}).get("Machine failure", 0) == 1 else "medium"
        if state.get("verification", {}).get("hallucination_risk") == "high":
            risk_level = "high"
        return _dump(
            ActionPlanOutput(
                recommended_checks=checks,
                immediate_actions=[
                    "Place equipment in a controlled troubleshooting workflow if currently unsafe.",
                    "Verify condition with qualified maintenance personnel before returning to service.",
                ],
                follow_up_actions=[
                    "Trend the top contributing sensor deviations over time.",
                    "Document findings in CMMS or maintenance log.",
                ],
                escalation_criteria=[
                    "Escalate if temperature, torque, vibration, or wear indicators continue rising.",
                    "Escalate if inspection contradicts dataset-based diagnosis.",
                ],
                risk_level=risk_level,
            )
        )

    return RunnableLambda(_run).with_config(run_name="action_planner_chain")


def create_report_writer_chain():
    def _run(state: dict[str, Any]) -> dict[str, Any]:
        top = (state.get("hypotheses") or [{"hypothesis": "Unknown", "confidence": 0.0}])[0]
        manual_evidence = [
            f"{item.get('source')}: {item.get('content', '')[:180].replace(chr(10), ' ')}"
            for item in state.get("retrieved_evidence", [])[:3]
        ]
        report = ReportOutput(
            most_likely_root_cause=top.get("hypothesis", "Unknown"),
            confidence_score=float(top.get("confidence", 0.0)),
            evidence_from_dataset=top.get("supporting_evidence", []),
            evidence_from_manuals=manual_evidence,
            alternative_causes=[item.get("hypothesis", "") for item in state.get("hypotheses", [])[1:]],
            recommended_checks=state.get("action_plan", {}).get("recommended_checks", []),
            risk_level=state.get("action_plan", {}).get("risk_level", "medium"),
            safety_notes=[
                "Use this output as decision support, not final certification.",
                "Follow lockout/tagout, site safety rules, and OEM procedures.",
                "Confirm with physical inspection and maintenance history.",
            ],
            final_action_plan=state.get("action_plan", {}).get("immediate_actions", [])
            + state.get("action_plan", {}).get("follow_up_actions", []),
            disclaimer=safety_layer.disclaimer(),
        )
        return _dump(report)

    return RunnableLambda(_run).with_config(run_name="report_writer_chain")

