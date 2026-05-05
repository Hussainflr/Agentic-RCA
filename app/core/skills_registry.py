from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def load_skills(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    blocks = re.split(r"\n---\n", text)
    skills: list[dict[str, Any]] = []
    for block in blocks:
        header = re.search(r"## Skill:\s*(.+)", block)
        if not header:
            continue
        skill: dict[str, Any] = {"name": header.group(1).strip()}
        for field in ["description", "inputs", "outputs", "allowed_tools", "safety_constraints", "example_usage"]:
            pattern = rf"{field}:\s*(.*?)(?=\n[a-z_]+:|\Z)"
            match = re.search(pattern, block, flags=re.S)
            if match:
                raw = match.group(1).strip()
                lines = [line.strip("- ").strip() for line in raw.splitlines() if line.strip()]
                skill[field] = lines if len(lines) > 1 else (lines[0] if lines else "")
        skills.append(skill)
    return skills


def skills_for_prompt(skills: list[dict[str, Any]]) -> str:
    chunks = []
    for skill in skills:
        chunks.append(
            f"- {skill.get('name')}: {skill.get('description')} "
            f"Allowed tools: {skill.get('allowed_tools', [])}. "
            f"Safety: {skill.get('safety_constraints', [])}."
        )
    return "\n".join(chunks)

