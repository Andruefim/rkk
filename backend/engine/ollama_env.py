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
