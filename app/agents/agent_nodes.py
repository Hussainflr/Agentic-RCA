from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.core.hooks import hooks
from app.core.prompts import build_agent_prompt
from app.core.providers import get_llm, invoke_text
from app.core.safety import safety_layer
from app.tools.engineering_tools import (
    anomaly_detection_tool,
    fault_lookup_tool,
    load_dataset_tool,
    manual_rag_search_tool,
    statistical_summary_tool,
)


def _run_agent(state: dict[str, Any], agent_name: str, fn):
    hooks.before_agent_run(state, agent_name)
    try:
        output = fn()
        state.setdefault("agent_outputs", {})[agent_name] = output
        hooks.after_agent_run(state, agent_name, output if isinstance(output, dict) else {"output": output})
    except Exception as exc:
        hooks.on_error(state, agent_name, exc)
    return state


def data_interpreter_agent(state: dict[str, Any]) -> dict[str, Any]:
    def _work() -> dict[str, Any]:
        loaded = load_dataset_tool(state)
        state["dataset_metadata"] = loaded["metadata"]
        state["selected_record"] = loaded["selected_record"]
        summary = statistical_summary_tool(state)
        state["sensor_summary"] = summary
        return summary

    return _run_agent(state, "data_interpreter", _work)


def failure_classifier_agent(state: dict[str, Any]) -> dict[str, Any]:
    def _work() -> dict[str, Any]:
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
        classification = {
            "primary_failure_type": primary,
            "confidence": confidence,
            "active_faults": active,
            "anomaly": anomaly,
        }
        state["anomaly_result"] = anomaly
        state["failure_classification"] = classification
        return classification

    return _run_agent(state, "failure_classifier", _work)


def knowledge_rag_agent(state: dict[str, Any]) -> dict[str, Any]:
    def _work() -> dict[str, Any]:
        classification = state.get("failure_classification", {})
        query = " ".join(
            [
                state.get("user_query", ""),
                classification.get("primary_failure_type", ""),
                " ".join(item.get("name", "") for item in classification.get("active_faults", [])),
            ]
        )
        evidence = manual_rag_search_tool(state, query=query, manuals_dir=settings.manuals_dir)
        state["retrieved_evidence"] = evidence
        return {"retrieved_evidence": evidence}

    return _run_agent(state, "knowledge_rag", _work)


def root_cause_reasoning_agent(state: dict[str, Any]) -> dict[str, Any]:
    def _work() -> dict[str, Any]:
        llm = get_llm(state.get("provider"))
        prompt = build_agent_prompt(
            "root_cause_reasoner",
            state,
            retrieved_context="\n\n".join(item.get("content", "")[:600] for item in state.get("retrieved_evidence", [])),
            tools=["statistical_summary", "fault_lookup", "manual_rag_search"],
        )
        _ = invoke_text(llm, prompt)
        classification = state.get("failure_classification", {})
        deviations = state.get("sensor_summary", {}).get("selected_record_deviations", {})
        top_features = sorted(
            deviations.items(),
            key=lambda item: abs(item[1].get("z_score", 0)),
            reverse=True,
        )[:3]
        evidence = [
            f"{name}: value={vals.get('value'):.2f}, z={vals.get('z_score'):.2f}, percentile={vals.get('percentile'):.2f}"
            for name, vals in top_features
        ]
        active_faults = classification.get("active_faults", [])
        if active_faults:
            root = active_faults[0]["name"]
            rationale = active_faults[0]["description"]
        else:
            root = classification.get("primary_failure_type", "Unclear condition")
            rationale = "No explicit failure-code label was available; hypothesis is based on anomaly and sensor deviations."
        hypotheses = [
            {
                "rank": 1,
                "hypothesis": root,
                "supporting_evidence": evidence + [rationale],
                "contradicting_evidence": [],
                "confidence": classification.get("confidence", 0.45),
            }
        ]
        for fault in active_faults[1:]:
            hypotheses.append(
                {
                    "rank": len(hypotheses) + 1,
                    "hypothesis": fault["name"],
                    "supporting_evidence": [fault["description"]],
                    "contradicting_evidence": ["Lower priority than primary active label."],
                    "confidence": 0.48,
                }
            )
        if len(hypotheses) == 1:
            hypotheses.append(
                {
                    "rank": 2,
                    "hypothesis": "Sensor or process drift",
                    "supporting_evidence": evidence,
                    "contradicting_evidence": ["Requires maintenance logs or time-series confirmation."],
                    "confidence": 0.38,
                }
            )
        state["hypotheses"] = hypotheses
        return {"hypotheses": hypotheses}

    return _run_agent(state, "root_cause_reasoner", _work)


def verifier_agent(state: dict[str, Any]) -> dict[str, Any]:
    def _work() -> dict[str, Any]:
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
        verification = {
            "groundedness_score": groundedness,
            "faithfulness_score": faithfulness,
            "hallucination_risk": risk,
            "verifier_decision": decision,
            "missing_evidence": missing,
        }
        state["verification"] = verification
        return verification

    return _run_agent(state, "verifier", _work)


def should_refine(state: dict[str, Any]) -> str:
    verification = state.get("verification", {})
    should = verification.get("verifier_decision") == "refine"
    below_limit = state.get("refinement_count", 0) < settings.max_refinements
    if should and below_limit:
        hooks.on_refinement_trigger(state, "; ".join(verification.get("missing_evidence", [])) or "Low verifier score.")
        return "refine"
    return "continue"


def refiner_agent(state: dict[str, Any]) -> dict[str, Any]:
    def _work() -> dict[str, Any]:
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
        return {"refinement_count": state["refinement_count"], "missing_evidence_addressed": missing}

    return _run_agent(state, "refiner", _work)


def action_planner_agent(state: dict[str, Any]) -> dict[str, Any]:
    def _work() -> dict[str, Any]:
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
        action_plan = {
            "recommended_checks": checks,
            "immediate_actions": [
                "Place equipment in a controlled troubleshooting workflow if currently unsafe.",
                "Verify condition with qualified maintenance personnel before returning to service.",
            ],
            "follow_up_actions": [
                "Trend the top contributing sensor deviations over time.",
                "Document findings in CMMS or maintenance log.",
            ],
            "escalation_criteria": [
                "Escalate if temperature, torque, vibration, or wear indicators continue rising.",
                "Escalate if inspection contradicts dataset-based diagnosis.",
            ],
            "risk_level": risk_level,
        }
        state["action_plan"] = action_plan
        return action_plan

    return _run_agent(state, "action_planner", _work)


def report_writer_agent(state: dict[str, Any]) -> dict[str, Any]:
    def _work() -> dict[str, Any]:
        top = (state.get("hypotheses") or [{"hypothesis": "Unknown", "confidence": 0.0}])[0]
        manual_evidence = [
            f"{item.get('source')}: {item.get('content', '')[:180].replace(chr(10), ' ')}"
            for item in state.get("retrieved_evidence", [])[:3]
        ]
        report = {
            "most_likely_root_cause": top.get("hypothesis", "Unknown"),
            "confidence_score": float(top.get("confidence", 0.0)),
            "evidence_from_dataset": top.get("supporting_evidence", []),
            "evidence_from_manuals": manual_evidence,
            "alternative_causes": [item.get("hypothesis", "") for item in state.get("hypotheses", [])[1:]],
            "recommended_checks": state.get("action_plan", {}).get("recommended_checks", []),
            "risk_level": state.get("action_plan", {}).get("risk_level", "medium"),
            "safety_notes": [
                "Use this output as decision support, not final certification.",
                "Follow lockout/tagout, site safety rules, and OEM procedures.",
                "Confirm with physical inspection and maintenance history.",
            ],
            "final_action_plan": state.get("action_plan", {}).get("immediate_actions", [])
            + state.get("action_plan", {}).get("follow_up_actions", []),
            "disclaimer": safety_layer.disclaimer(),
        }
        state["report"] = report
        hooks.on_report_generated(state)
        return report

    return _run_agent(state, "report_writer", _work)

