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

DEFAULT_OLLAMA_MODEL = "gemma4:e4b"
DEFAULT_OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"


def get_ollama_model() -> str:
    m = (os.environ.get("RKK_OLLAMA_MODEL") or os.environ.get("OLLAMA_MODEL") or "").strip()
    return m or DEFAULT_OLLAMA_MODEL


def get_ollama_generate_url() -> str:
    u = (os.environ.get("RKK_OLLAMA_URL") or "").strip()
    return u or DEFAULT_OLLAMA_GENERATE_URL
