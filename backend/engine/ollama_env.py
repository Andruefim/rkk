"""
Единая точка для URL и имени модели Ollama.

Задаётся в корневом .env репозитория (рядом с backend/):
  RKK_OLLAMA_MODEL=gemma4:e4b
  RKK_OLLAMA_URL=http://localhost:11434/api/generate

Короткий алиас имени модели (если RKK_OLLAMA_MODEL не задан):
  OLLAMA_MODEL=llama3.2
"""
from __future__ import annotations

import os
from typing import Any

__all__ = [
    "DEFAULT_OLLAMA_GENERATE_URL",
    "DEFAULT_OLLAMA_MODEL",
    "get_ollama_generate_url",
    "get_ollama_model",
    "ollama_think_disabled_payload",
    "ollama_yield_to_system2_enabled",
    "system2_ollama_busy",
]

DEFAULT_OLLAMA_MODEL = "gemma4:e4b"
DEFAULT_OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"


def get_ollama_model() -> str:
    m = (os.environ.get("RKK_OLLAMA_MODEL") or os.environ.get("OLLAMA_MODEL") or "").strip()
    return m or DEFAULT_OLLAMA_MODEL


def get_ollama_generate_url() -> str:
    u = (os.environ.get("RKK_OLLAMA_URL") or "").strip()
    return u or DEFAULT_OLLAMA_GENERATE_URL


def ollama_think_disabled_payload() -> dict[str, Any]:
    """
    Qwen3.x в Ollama пишет рассуждения в поле `thinking`, ответ — в `response`.
    При лимите num_predict весь бюджет может уйти в thinking → response пустой, Phase3/RAG/VLM ломаются.

    Top-level `think: false` отключает отдельный thinking-канал (см. Ollama Thinking / Qwen3).
    Не класть внутрь `options`.

    RKK_OLLAMA_THINK=1 — не добавлять флаг (оставить поведение модели по умолчанию).
    """
    if os.environ.get("RKK_OLLAMA_THINK", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return {}
    return {"think": False}


def ollama_yield_to_system2_enabled() -> bool:
    """
    Если включено (по умолчанию да), фоновые вызовы Ollama (L2/L3, curriculum LLM)
    не стартуют, пока System2 держит in-flight запрос к Ollama — один GPU/CPU у Ollama.
    Выкл: RKK_OLLAMA_YIELD_TO_S2=0
    """
    return os.environ.get("RKK_OLLAMA_YIELD_TO_S2", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def system2_ollama_busy(sim: Any) -> bool:
    """True, если у симуляции активен System2 и у него незавершённый Ollama (план или recovery)."""
    s2 = getattr(sim, "_system2", None)
    if s2 is None:
        return False
    fn = getattr(s2, "ollama_busy", None)
    if not callable(fn):
        return False
    try:
        return bool(fn())
    except Exception:
        return False
