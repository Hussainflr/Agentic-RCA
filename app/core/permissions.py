from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PermissionManager:
    role_tool_allowlist: dict[str, set[str]] = field(
        default_factory=lambda: {
            "maintenance_reliability_engineer": {
                "dataset_loader",
                "statistical_summary",
                "anomaly_detector",
                "fault_lookup",
                "manual_rag_search",
                "chart_generator",
                "report_generator",
            },
            "viewer": {
                "dataset_loader",
                "statistical_summary",
                "anomaly_detector",
                "fault_lookup",
                "manual_rag_search",
                "chart_generator",
                "report_generator",
            },
        }
    )

    def can_call(self, role: str, tool_name: str) -> bool:
        return tool_name in self.role_tool_allowlist.get(role, set())

    def assert_can_call(self, role: str, tool_name: str) -> None:
        if not self.can_call(role, tool_name):
            raise PermissionError(f"Role '{role}' cannot call tool '{tool_name}'.")


permission_manager = PermissionManager()
