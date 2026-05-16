from __future__ import annotations

import json
import os
import time
from typing import Any

from engine.system2.schema import (
    System2Proposal,
    _extract_json_object,
    proposal_from_dict,
)
from engine.system2.validate import validate_proposal


def _ollama_url() -> str:
    return os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")


def _ollama_model() -> str:
    return os.environ.get(
        "RKK_SYSTEM2_OLLAMA_MODEL",
        os.environ.get("RKK_OLLAMA_MODEL", "llama3.2"),
    ).strip()


_llm_cache: dict[str, tuple[float, System2Proposal]] = {}


def llm_teacher_enabled() -> bool:
    return os.environ.get("RKK_SYSTEM2_LLM", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _cache_ttl_s() -> float:
    try:
        return float(os.environ.get("RKK_SYSTEM2_LLM_CACHE_TTL", "45"))
    except ValueError:
        return 45.0


def _cache_key(summary: dict[str, Any]) -> str:
    q = int(os.environ.get("RKK_SYSTEM2_LLM_CACHE_BUCKET", "12"))
    q = max(1, min(240, q))
    t = int(summary.get("tick", 0) or 0) // q
    cz = round(float(summary.get("com_z") or 0.5), 2)
    ps = round(float(summary.get("posture_stability") or 0.5), 2)
    td = round(float(summary.get("target_dist") or 0.5), 2)
    return f"{t}|{cz}|{ps}|{td}"


def vlm_slots_from_sim(sim: Any | None) -> str:
    """Текст для промпта; вызывать с main thread до submit в worker (без доступа к sim из фона)."""
    return _vlm_slots_hint(sim)


def _vlm_slots_hint(sim: Any | None) -> str:
    if sim is None:
        return ""
    try:
        ve = getattr(sim, "_visual_env", None)
        if ve is None:
            return ""
        lex = getattr(ve, "_slot_lexicon", None) or {}
        bits = []
        for i in range(min(6, int(getattr(ve, "n_slots", 0) or 0))):
            e = lex.get(f"slot_{i}") or {}
            lab = str(e.get("label", "?"))[:24]
            bits.append(f"{i}:{lab}")
        return "; ".join(bits) if bits else ""
    except Exception:
        return ""


def proposal_from_llm_cache_only(obs_summary: dict[str, Any]) -> System2Proposal | None:
    """Только кэш Ollama — без сети, для главного тика."""
    if not llm_teacher_enabled():
        return None
    key = _cache_key(obs_summary)
    now = time.monotonic()
    ent = _llm_cache.get(key)
    if ent is not None and (now - ent[0]) < _cache_ttl_s():
        return ent[1]
    return None


def proposal_from_llm_network_fetch(
    obs_summary: dict[str, Any],
    vlm_slots_str: str,
) -> System2Proposal | None:
    """
    Сетевой вызов Ollama (блокирующий). Вызывать из worker thread.
    `vlm_slots_str` — заранее собранная строка с main thread.
    """
    if not llm_teacher_enabled():
        return None
    key = _cache_key(obs_summary)
    now = time.monotonic()
    ent = _llm_cache.get(key)
    if ent is not None and (now - ent[0]) < _cache_ttl_s():
        return ent[1]

    extra = f"\nVLM_slots: {vlm_slots_str}\n" if vlm_slots_str else ""
    prompt = (
        "You are System2 for an embodied humanoid (Nova). Output ONLY one JSON object, no markdown.\n"
        "Keys: macro (one of IDLE, RECOVER_POSTURE, LOCOMOTE_DELIVERY, EXPLORE), "
        "goal (optional: com_z_min, posture_stability_min, target_dist_max as floats in [0.05,0.95]), "
        "intent_deltas (optional: small floats for keys starting with intent_ only, max magnitude 0.12), "
        "rationale (one short sentence).\n"
        f"{extra}"
        "Current summary:\n"
        f"{json.dumps(obs_summary, ensure_ascii=False, indent=2)[:3200]}\n"
    )
    raw: dict[str, Any] | None = None
    try:
        import httpx

        url = f"{_ollama_url()}/api/chat"
        body = {
            "model": _ollama_model(),
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.12, "num_predict": 240},
        }
        with httpx.Client(timeout=35.0) as client:
            r = client.post(url, json=body)
            r.raise_for_status()
            data = r.json()
        msg = (data.get("message") or {}) if isinstance(data, dict) else {}
        text = str(msg.get("content", "")).strip()
        if not text:
            text = str(data.get("response", "")).strip() if isinstance(data, dict) else ""
        if text:
            try:
                raw = json.loads(text)
            except json.JSONDecodeError:
                raw = _extract_json_object(text)
    except Exception:
        try:
            import httpx

            url = f"{_ollama_url()}/api/generate"
            body = {
                "model": _ollama_model(),
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.12, "num_predict": 240},
            }
            with httpx.Client(timeout=35.0) as client:
                r = client.post(url, json=body)
                r.raise_for_status()
                data = r.json()
            text = str(data.get("response", "")).strip() if isinstance(data, dict) else ""
            if text:
                try:
                    raw = json.loads(text)
                except json.JSONDecodeError:
                    raw = _extract_json_object(text)
        except Exception:
            return None

    if isinstance(raw, dict):
        try:
            if len(json.dumps(raw, ensure_ascii=False)) > 12000:
                return None
        except (TypeError, ValueError):
            return None

    prop = validate_proposal(
        proposal_from_dict(raw if isinstance(raw, dict) else None)
    )
    if prop is not None:
        _llm_cache[key] = (now, prop)
    return prop


def proposal_from_llm(
    obs_summary: dict[str, Any],
    *,
    sim: Any | None = None,
) -> System2Proposal | None:
    """
    Кэш + синхронный сетевой вызов (для тестов / RKK_SYSTEM2_LLM_SYNC=1).
    В симуляции при включённом LLM без SYNC используйте cache_only + worker.
    """
    if not llm_teacher_enabled():
        return None
    hit = proposal_from_llm_cache_only(obs_summary)
    if hit is not None:
        return hit
    return proposal_from_llm_network_fetch(obs_summary, vlm_slots_from_sim(sim))
