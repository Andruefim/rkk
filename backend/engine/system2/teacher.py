from __future__ import annotations

import json
import os
import time
from typing import Any

from engine.llm_json_extract import ollama_json_format_payload
from engine.ollama_env import ollama_think_disabled_payload
from engine.slot_lexicon import (
    allowed_target_ids,
    normalize_ollama_image_b64,
    ollama_chat_url,
    validate_slot_labels,
)
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


def s2_unified_vlm_enabled() -> bool:
    if not llm_teacher_enabled():
        return False
    return os.environ.get("RKK_S2_UNIFIED_VLM", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def s2_unified_slots_enabled() -> bool:
    if not s2_unified_vlm_enabled():
        return False
    return os.environ.get("RKK_S2_UNIFIED_SLOTS", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _s2_fallen_camera_policy() -> str:
    """exo_only | ego_only | both — при fallen что слать в unified S2."""
    v = (os.environ.get("RKK_S2_FALLEN_CAMERA", "exo_only") or "exo_only").strip().lower()
    if v in ("both", "exo+ego", "dual"):
        return "both"
    if v in ("ego", "ego_only", "first_person"):
        return "ego_only"
    return "exo_only"


def enrich_summary_for_s2_llm(
    base: Any,
    obs_f: dict[str, float],
    summary: dict[str, Any],
    *,
    fallen: bool,
) -> dict[str, Any]:
    out: dict[str, Any] = dict(summary)
    out["grounding"] = build_compact_recovery_state(base, obs_f)
    out["fallen"] = bool(fallen)
    fn = getattr(base, "is_fallen", None)
    if callable(fn):
        try:
            out["env_reports_fallen"] = bool(fn())
        except Exception:
            pass
    return out


def collect_s2_planning_frames(agent: Any, *, fallen: bool) -> tuple[list[str], str]:
    """Main-thread only: base64 JPEGs for Ollama ``images`` + caption for the prompt."""
    fn = getattr(getattr(agent, "env", None), "get_frame_base64", None)
    if not callable(fn):
        return [], ""
    try:
        ego = fn("ego") or fn(None)
    except Exception:
        ego = None
    try:
        exo = fn("exo")
    except Exception:
        exo = None
    pol = _s2_fallen_camera_policy()
    if not fallen:
        imgs = [x for x in [ego] if x]
        cap = "Image 1: robot first-person camera."
        return imgs, cap
    if pol == "ego_only":
        imgs = [x for x in [ego] if x]
        cap = "Image 1: first-person camera (agent is fallen — geometry may be confusing)."
        return imgs, cap
    if pol == "both":
        seq = [ego, exo]
        imgs = [x for x in seq if x]
        parts = []
        if ego:
            parts.append("Image 1: first-person.")
        if exo:
            idx = 2 if ego else 1
            parts.append(f"Image {idx}: third-person overview of body pose and floor.")
        cap = " ".join(parts) if parts else "Camera views of the fallen humanoid."
        return imgs, cap
    imgs = [x for x in [exo or ego] if x]
    cap = (
        "Third-person overview of full body pose and ground contact (fallen). "
        "Prefer this view over proprioception alone."
    )
    return imgs, cap


def _s2_unified_max_masks() -> int:
    try:
        return max(0, min(8, int(os.environ.get("RKK_S2_UNIFIED_MAX_MASKS", "4"))))
    except ValueError:
        return 4


def _build_unified_s2_prompt(
    *,
    obs_summary: dict[str, Any],
    vlm_slots_str: str,
    image_caption: str,
    include_slots: bool,
    slot_ids: list[str],
    allowed_csv: str,
) -> str:
    extra = f"\nPrevious VLM slot hints (may be stale): {vlm_slots_str}\n" if vlm_slots_str else ""
    es_hint = ", ".join(sorted(expected_state_key_allowlist())[:56])
    slot_block = ""
    if include_slots and slot_ids:
        slot_block = (
            "\nAdditionally, label EVERY attention slot (exact keys) in JSON key \"slot_labels\":\n"
            f'  "slot_labels": {{"slot_0":{{"label":"short English","likely_phys":["phys_lknee"],'
            f'"confidence":0.7}}, ...}}\n'
            "Rules: likely_phys MUST copy EXACT strings from the allowed list only; "
            "include every slot key listed below even if unsure (low confidence).\n"
            f"Slot keys: {', '.join(slot_ids)}\n"
            f"Allowed likely_phys targets (subset): {allowed_csv}\n"
        )
    img_block = ""
    if image_caption:
        img_block = (
            "\nYou are given one or more images in order (JPEG). "
            f"{image_caption}\n"
            "After scene images, any further images are small slot-attention thumbnails "
            "(brighter = stronger attention), in slot index order.\n"
        )
    return (
        "You are System2 for an embodied humanoid (Nova). Output ONLY one JSON object, no markdown.\n"
        "Primary keys: macro (IDLE, RECOVER_POSTURE, LOCOMOTE_DELIVERY, EXPLORE), "
        "goal (optional com_z_min, posture_stability_min, target_dist_max in [0.05,0.95]), "
        "intent_deltas (optional intent_* floats, max magnitude 0.12), rationale (short).\n"
        "Optional: expected_state (sensor→float), max_prediction_error (positive float), skill_id.\n"
        f"Example expected_state keys: {es_hint}\n"
        f"{slot_block}{img_block}{extra}"
        "Numeric summary (graph + meters / norms):\n"
        f"{json.dumps(obs_summary, ensure_ascii=False, indent=2)[:3600]}\n"
    )


def _ollama_post_chat_json(
    *,
    prompt: str,
    images: list[str] | None,
    num_predict: int,
    temperature: float,
) -> dict[str, Any] | None:
    import httpx

    gen_url = f"{_ollama_url()}/api/generate"
    chat_url = ollama_chat_url(gen_url)
    body: dict[str, Any] = {
        "model": _ollama_model(),
        "messages": [],
        "stream": False,
        **ollama_think_disabled_payload(),
        "options": {"temperature": temperature, "num_predict": int(num_predict)},
        **ollama_json_format_payload(),
    }
    imgs = [normalize_ollama_image_b64(x) for x in (images or [])]
    imgs = [x for x in imgs if x]
    if imgs:
        body["messages"] = [{"role": "user", "content": prompt, "images": imgs}]
    else:
        body["messages"] = [{"role": "user", "content": prompt}]
    try:
        with httpx.Client(timeout=httpx.Timeout(90.0, connect=8.0)) as client:
            r = client.post(chat_url, json=body)
            r.raise_for_status()
            data = r.json()
        msg = (data.get("message") or {}) if isinstance(data, dict) else {}
        text = str(msg.get("content", "")).strip()
        if not text:
            text = str(data.get("response", "")).strip() if isinstance(data, dict) else ""
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return _extract_json_object(text)
    except Exception:
        return None


def unified_s2_planning_network_fetch(job: dict[str, Any]) -> tuple[System2Proposal | None, dict[str, dict[str, Any]] | None]:
    """
    Single Ollama /api/chat: System2 macro JSON + optional slot_labels (validated).
    No text cache — multimodal state changes too fast.
    """
    if not llm_teacher_enabled():
        return None, None
    summary = job.get("summary") or {}
    vlm = str(job.get("vlm_slots", "") or "")
    images: list[str] = list(job.get("images") or [])
    caption = str(job.get("image_caption", "") or "")
    include_slots = bool(job.get("include_slots"))
    slot_ids: list[str] = list(job.get("slot_ids") or [])
    allowed_csv = str(job.get("allowed_csv", "") or "")
    max_masks = _s2_unified_max_masks()
    for m in job.get("masks_b64") or []:
        if m and max_masks > 0 and len(images) < 14:
            images.append(str(m))
            max_masks -= 1

    prompt = _build_unified_s2_prompt(
        obs_summary=summary,
        vlm_slots_str=vlm,
        image_caption=caption,
        include_slots=include_slots and bool(slot_ids),
        slot_ids=slot_ids,
        allowed_csv=allowed_csv,
    )
    try:
        npred = int(os.environ.get("RKK_S2_UNIFIED_NUM_PREDICT", "420"))
    except ValueError:
        npred = 420
    npred = max(120, min(2048, npred))
    raw = _ollama_post_chat_json(
        prompt=prompt,
        images=images if images else None,
        num_predict=npred,
        temperature=0.12,
    )
    if not isinstance(raw, dict):
        return None, None

    slot_out: dict[str, dict[str, Any]] | None = None
    if include_slots and slot_ids:
        sl_raw = raw.get("slot_labels")
        if isinstance(sl_raw, dict):
            allowed = allowed_target_ids(list(job.get("variable_ids") or []))
            slot_out = validate_slot_labels(sl_raw, slot_ids, allowed)
            if not slot_out:
                slot_out = None

    prop = validate_proposal(proposal_from_dict(raw))
    return prop, slot_out


def llm_plan_worker(job: dict[str, Any]) -> tuple[System2Proposal | None, dict[str, dict[str, Any]] | None]:
    """Worker entry: text-only cache path OR unified multimodal (no proposal cache)."""
    if job.get("unified_vlm"):
        return unified_s2_planning_network_fetch(job)
    summ = job.get("summary") or {}
    vlm = str(job.get("vlm_slots", "") or "")
    return proposal_from_llm_network_fetch(summ, vlm), None


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


def s2_recovery_vision_enabled() -> bool:
    """Третье лицо в recovery LLM при падении (multimodal /api/chat)."""
    if not recovery_llm_enabled():
        return False
    return os.environ.get("RKK_S2_RECOVERY_VISION", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
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
    *,
    images_b64: list[str] | None = None,
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
    imgs = [normalize_ollama_image_b64(x) for x in (images_b64 or []) if x]
    imgs = [x for x in imgs if x]
    if imgs:
        prompt_vis = (
            prompt
            + "\nAttached image(s): third-person view of the humanoid pose, contacts, and floor.\n"
        )
        raw = _ollama_post_chat_json(
            prompt=prompt_vis,
            images=imgs,
            num_predict=380,
            temperature=0.1,
        )
    if raw is None:
        try:
            import httpx

            url = f"{_ollama_url()}/api/chat"
            body = {
                "model": _ollama_model(),
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                **ollama_think_disabled_payload(),
                "format": "json",
                "options": {"temperature": 0.1, "num_predict": 320},
            }
            with httpx.Client(timeout=httpx.Timeout(55.0, connect=8.0)) as client:
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
    from engine.system2.recovery_schedule import enrich_recovery_steps

    for st in steps:
        st["intent_deltas"] = clip_intent_deltas(st.get("intent_deltas") or {})
    steps = enrich_recovery_steps(steps)
    if not any(st.get("intent_deltas") for st in steps):
        return None
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
