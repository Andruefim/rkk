from __future__ import annotations

import json
import os
import time
from typing import Any

from engine.system2.schema import (
    System2Proposal,
    _extract_json_object,
    expected_state_key_allowlist,
    parse_recovery_llm_plan,
    proposal_from_dict,
)
from engine.system2.validate import clip_intent_deltas, validate_proposal


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
    es_hint = ", ".join(sorted(expected_state_key_allowlist())[:56])
    prompt = (
        "You are System2 for an embodied humanoid (Nova). Output ONLY one JSON object, no markdown.\n"
        "Keys: macro (one of IDLE, RECOVER_POSTURE, LOCOMOTE_DELIVERY, EXPLORE), "
        "goal (optional: com_z_min, posture_stability_min, target_dist_max as floats in [0.05,0.95]), "
        "intent_deltas (optional: small floats for keys starting with intent_ only, max magnitude 0.12), "
        "rationale (one short sentence).\n"
        "Optional intentional prior for episode success: expected_state object mapping sensor names to "
        "target floats (subset of known keys only; unknown keys ignored). "
        "max_prediction_error (optional positive float): allowed L1 prediction error across those keys; "
        "if omitted, a code fallback scales with how many keys you set. "
        "skill_id (optional short string) for curriculum / concept naming.\n"
        f"Example expected_state keys (not exhaustive): {es_hint}\n"
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
        with httpx.Client(timeout=httpx.Timeout(40.0, connect=8.0)) as client:
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
            with httpx.Client(timeout=httpx.Timeout(40.0, connect=8.0)) as client:
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


def recovery_llm_enabled() -> bool:
    """Multi-step motor recovery JSON (requires ``RKK_SYSTEM2_LLM``)."""
    if not llm_teacher_enabled():
        return False
    return os.environ.get("RKK_S2_RECOVERY_LLM", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def build_compact_recovery_state(base: Any, obs_f: dict[str, Any]) -> dict[str, float]:
    """Grounded pose vector: raw m + normalized torso/com where available + graph snapshot cues."""
    out: dict[str, float] = {}
    sim = getattr(base, "_sim", None)
    raw: dict[str, Any] = {}
    if sim is not None and callable(getattr(sim, "get_state", None)):
        try:
            r = sim.get_state()
            if isinstance(r, dict):
                raw = r
        except Exception:
            raw = {}
    norm = getattr(base, "_norm", None)
    for key in ("com_z", "torso_pitch", "torso_roll"):
        if key in raw:
            try:
                out[f"{key}_m"] = float(raw[key])
            except (TypeError, ValueError):
                pass
        if key in raw and callable(norm):
            try:
                out[f"{key}_norm"] = float(norm(key, float(raw[key])))
            except Exception:
                pass
    for key in ("foot_contact_l", "foot_contact_r", "posture_stability", "support_bias"):
        v = obs_f.get(key, obs_f.get(f"phys_{key}"))
        if v is not None:
            try:
                out[key] = float(v)
            except (TypeError, ValueError):
                pass
    return out


def recovery_steps_from_llm_network_fetch(
    compact: dict[str, Any],
    vlm_slots_str: str,
) -> tuple[list[dict[str, Any]], dict[str, float], float | None] | None:
    """
    Blocking Ollama call: JSON with ``steps`` plus optional ``expected_state``, ``max_prediction_error``.
    Returns ``(steps, expected_state, max_pe)`` or None if steps invalid.
    Worker-thread only; same host as ``proposal_from_llm_network_fetch``.
    """
    if not recovery_llm_enabled():
        return None

    extra = f"\nVLM_slots: {vlm_slots_str}\n" if vlm_slots_str else ""
    es_hint = ", ".join(sorted(expected_state_key_allowlist())[:40])
    prompt = (
        "You are System2 motor recovery planner for humanoid Nova. "
        "Output ONLY one JSON object, no markdown.\n"
        "Schema: {\"steps\": [{\"ticks\": <int 1-80>, \"intent_deltas\": "
        "{ \"intent_*\": <small float -0.12..0.12> } }, ...] }\n"
        "Optional top-level keys: expected_state (map of sensor names to target floats), "
        "max_prediction_error (positive float, L1 cap vs expected_state at episode end).\n"
        "2–6 steps. Prefer intent_stop_recover, intent_support_left/right, "
        "intent_torso_forward, intent_arm_counterbalance, intent_lean_forward. "
        "Do not invent keys outside intent_* for intent_deltas; "
        f"expected_state keys must be from the registry, e.g. {es_hint}\n"
        f"{extra}"
        "Current grounded state (m = meters, _norm = env normalization):\n"
        f"{json.dumps(compact, ensure_ascii=False, indent=2)[:2800]}\n"
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
            "options": {"temperature": 0.1, "num_predict": 320},
        }
        with httpx.Client(timeout=httpx.Timeout(40.0, connect=8.0)) as client:
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
        return None

    if not isinstance(raw, dict):
        return None
    plan = parse_recovery_llm_plan(raw)
    if plan is None:
        return None
    steps, es, mx = plan
    for st in steps:
        st["intent_deltas"] = clip_intent_deltas(st.get("intent_deltas") or {})
    return (steps, es, mx)


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
