from __future__ import annotations

from typing import Any

from app.core.config import settings


class DeterministicLLM:
    """Fallback model that keeps the demo runnable without external services."""

    def invoke(self, prompt: str) -> str:
        return (
            "Deterministic fallback response: use the structured sensor, anomaly, fault-code, "
            "and manual evidence already computed by tools. Avoid unsupported claims."
        )


def get_llm(provider: str | None = None) -> Any:
    provider = (provider or settings.llm_provider).lower()
    try:
        if provider == "ollama":
            from langchain_ollama import ChatOllama

            return ChatOllama(model=settings.ollama_model, base_url=settings.ollama_base_url, temperature=0.1)
        if provider == "openai" and settings.allow_cloud_llm:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model=settings.openai_model, temperature=0.1)
        if provider in {"anthropic", "claude"} and settings.allow_cloud_llm:
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(model=settings.anthropic_model, temperature=0.1)
    except Exception:
        return DeterministicLLM()
    return DeterministicLLM()


def invoke_text(llm: Any, prompt: str) -> str:
    try:
        response = llm.invoke(prompt)
        return getattr(response, "content", str(response))
    except Exception:
        return DeterministicLLM().invoke(prompt)

