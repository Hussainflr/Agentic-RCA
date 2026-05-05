from __future__ import annotations

from typing import Any

from app.core.schemas import utc_now


class LifecycleHooks:
    """Simple lifecycle hook bus for traceable agent/tool execution."""

    def _append(self, state: dict[str, Any], event: str, payload: dict[str, Any]) -> None:
        state.setdefault("traces", []).append({"event": event, "timestamp": utc_now(), **payload})

    def before_agent_run(self, state: dict[str, Any], agent_name: str) -> None:
        self._append(state, "before_agent_run", {"agent": agent_name})

    def after_agent_run(self, state: dict[str, Any], agent_name: str, output: dict[str, Any]) -> None:
        self._append(
            state,
            "after_agent_run",
            {"agent": agent_name, "output_keys": sorted(output.keys())},
        )

    def before_tool_call(self, state: dict[str, Any], tool_name: str, args: dict[str, Any]) -> None:
        self._append(state, "before_tool_call", {"tool": tool_name, "args": args})

    def after_tool_call(self, state: dict[str, Any], tool_name: str, result: Any) -> None:
        summary = str(result)
        self._append(state, "after_tool_call", {"tool": tool_name, "result_summary": summary[:500]})

    def on_error(self, state: dict[str, Any], component: str, error: Exception | str) -> None:
        message = str(error)
        state.setdefault("errors", []).append(message)
        self._append(state, "on_error", {"component": component, "error": message})

    def on_refinement_trigger(self, state: dict[str, Any], reason: str) -> None:
        self._append(state, "on_refinement_trigger", {"reason": reason})

    def on_report_generated(self, state: dict[str, Any], report_path: str | None = None) -> None:
        self._append(state, "on_report_generated", {"report_path": report_path})


hooks = LifecycleHooks()

