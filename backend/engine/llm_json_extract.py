"""
Устойчивое извлечение JSON из ответов локальных LLM (Ollama / Gemma и др.):
markdown-фенсы, мусор до/после, хвостовые запятые, опционально think-теги.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any


def ollama_json_format_enabled() -> bool:
    """RKK_OLLAMA_JSON_FORMAT=0 — без format=json в generate (bootstrap/LLM loop).

    Phase3/VLM: ollama_json_format_teacher_vlm_payload() — отдельно, по умолчанию без format.
    """
    return os.environ.get("RKK_OLLAMA_JSON_FORMAT", "1").strip().lower() not in (
        "0",
        "false",
        "off",
        "no",
    )


def ollama_json_format_payload() -> dict[str, Any]:
    return {"format": "json"} if ollama_json_format_enabled() else {}


def ollama_json_format_teacher_vlm_payload() -> dict[str, Any]:
    """
    Phase3 + VLM: при format=json многие локальные модели (Gemma и др.) отдают
    мета-схему вместо ig_rules / slot_* — по умолчанию не шлём format.

    Включить принудительно: RKK_OLLAMA_JSON_FORMAT_TEACHER_VLM=1 при RKK_OLLAMA_JSON_FORMAT=1.
    """
    if os.environ.get("RKK_OLLAMA_JSON_FORMAT_TEACHER_VLM", "0").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return {}
    return ollama_json_format_payload()


_T_OPEN, _T_CLOSE = "<" + "think" + ">", "</" + "think" + ">"
_THINK_RE = re.compile(
    re.escape(_T_OPEN) + r"[\s\S]*?" + re.escape(_T_CLOSE) + r"|"
    r"<reasoning>[\s\S]*?</reasoning>|"
    r"<thought>[\s\S]*?</thought>|"
    r"<redacted_thinking>[\s\S]*?</redacted_thinking>",
    re.IGNORECASE,
)


def _strip_think_blocks(s: str) -> str:
    return _THINK_RE.sub("", s)


def _extract_markdown_json_blocks(raw: str) -> list[str]:
    out: list[str] = []
    for m in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", raw, re.IGNORECASE):
        inner = m.group(1).strip()
        if inner.lower().startswith("json"):
            inner = inner[4:].lstrip()
        if inner:
            out.append(inner)
    return out


def _relax_trailing_commas(s: str) -> str:
    s = re.sub(r",\s*}", "}", s)
    s = re.sub(r",\s*]", "]", s)
    return s


def _scan_decode_objects(s: str) -> dict[str, Any] | None:
    dec = json.JSONDecoder()
    i = 0
    while True:
        j = s.find("{", i)
        if j < 0:
            return None
        try:
            obj, _end = dec.raw_decode(s, j)
        except json.JSONDecodeError:
            i = j + 1
            continue
        if isinstance(obj, dict):
            return obj
        i = j + 1


def _scan_decode_arrays(s: str) -> list[Any] | None:
    dec = json.JSONDecoder()
    i = 0
    while True:
        j = s.find("[", i)
        if j < 0:
            return None
        try:
            obj, _end = dec.raw_decode(s, j)
        except json.JSONDecodeError:
            i = j + 1
            continue
        if isinstance(obj, list):
            return obj
        i = j + 1


def _try_parse_object(s: str) -> dict[str, Any] | None:
    s = s.strip()
    if not s:
        return None
    for variant in (s, _relax_trailing_commas(s)):
        try:
            v = json.loads(variant)
            if isinstance(v, dict):
                return v
        except json.JSONDecodeError:
            pass
        got = _scan_decode_objects(variant)
        if got is not None:
            return got
    return None


def _try_parse_array(s: str) -> list[Any] | None:
    s = s.strip()
    if not s:
        return None
    for variant in (s, _relax_trailing_commas(s)):
        try:
            v = json.loads(variant)
            if isinstance(v, list):
                return v
        except json.JSONDecodeError:
            pass
        got = _scan_decode_arrays(variant)
        if got is not None:
            return got
    return None


def parse_json_object_loose(raw: str) -> dict[str, Any] | None:
    """Первый валидный JSON-object из ответа модели."""
    if not raw:
        return None
    seen: set[str] = set()

    def _emit(s: str) -> str:
        return _strip_think_blocks(s).strip()

    candidates: list[str] = []
    # Сначала варианты без вырезания think: JSON иногда целиком внутри
    # <redacted_thinking>…</redacted_thinking> (Gemma 4 и др.).
    for block in _extract_markdown_json_blocks(raw):
        b = block.strip()
        if b:
            candidates.append(b)
        eb = _emit(block)
        if eb:
            candidates.append(eb)
    er = _emit(raw)
    if er:
        candidates.append(er)
    tr = raw.strip()
    if tr:
        candidates.append(tr)

    for c in candidates:
        if not c or c in seen:
            continue
        seen.add(c)
        obj = _try_parse_object(c)
        if obj is not None:
            return obj
    return None


def scan_json_objects_having_any_key(raw: str, keys: set[str]) -> dict[str, Any] | None:
    """
    Перебирает все вложенные JSON-object в тексте; первый dict с пересечением ключей с keys.
    Обходит случай, когда parse_json_object_loose схватил schema-мусор до настоящего объекта.
    """
    if not raw or not keys:
        return None
    dec = json.JSONDecoder()

    def _scan(text: str) -> dict[str, Any] | None:
        i = 0
        while True:
            j = text.find("{", i)
            if j < 0:
                return None
            try:
                obj, _end = dec.raw_decode(text, j)
            except json.JSONDecodeError:
                i = j + 1
                continue
            if isinstance(obj, dict) and keys.intersection(obj.keys()):
                return obj
            i = j + 1

    for text in (raw, _strip_think_blocks(raw)):
        got = _scan(text)
        if got is not None:
            return got
    return None


def parse_json_array_loose(raw: str) -> list[Any] | None:
    """Первый валидный JSON-array; иначе object с ключами edges|hypotheses|links."""
    if not raw:
        return None
    seen: set[str] = set()

    def _emit(s: str) -> str:
        return _strip_think_blocks(s).strip()

    candidates: list[str] = []
    for block in _extract_markdown_json_blocks(raw):
        b = block.strip()
        if b:
            candidates.append(b)
        eb = _emit(block)
        if eb:
            candidates.append(eb)
    er = _emit(raw)
    if er:
        candidates.append(er)
    tr = raw.strip()
    if tr:
        candidates.append(tr)

    for c in candidates:
        if not c or c in seen:
            continue
        seen.add(c)
        arr = _try_parse_array(c)
        if arr is not None:
            return arr
        obj = _try_parse_object(c)
        if isinstance(obj, dict):
            for key in ("edges", "hypotheses", "links", "candidates", "array"):
                v = obj.get(key)
                if isinstance(v, list):
                    return v
    return None
