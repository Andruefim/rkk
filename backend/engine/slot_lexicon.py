"""
slot_lexicon.py — Фаза 2: VLM-разметка слотов (лейблы + likely_phys).

Ollama /api/chat с полем images (кадр + миниатюры масок); при ошибке — текстовый /api/generate
по slot_values/variability (хуже, но без vision API).
"""
from __future__ import annotations

import hashlib
import json
import os
import traceback
from typing import Any

import httpx

from engine.llm_json_extract import (
    ollama_json_format_teacher_vlm_payload,
    parse_json_object_loose,
    scan_json_objects_having_any_key,
)
from engine.ollama_env import ollama_think_disabled_payload


def frame_content_hash(frame_b64: str | None) -> str:
    if not frame_b64:
        return ""
    raw = frame_b64.encode("utf-8", errors="ignore")[:8192]
    return hashlib.sha256(raw + str(len(frame_b64)).encode()).hexdigest()[:16]


def _vlm_num_predict() -> int:
    """
    Ollama options.num_predict — верхняя граница числа генерируемых токенов (не минимум).
    Слишком мало: ответ обрезается до JSON → пустой/битый parse. Раньше 600 резали для скорости;
    модели с преамбулой/thinking съедают лимит до фактического JSON.
    Задать: RKK_VLM_NUM_PREDICT (128..32768).
    """
    try:
        v = int(os.environ.get("RKK_VLM_NUM_PREDICT", "1600"))
    except ValueError:
        v = 1600
    return max(128, min(v, 32768))


def normalize_ollama_image_b64(s: str) -> str:
    """Ollama /api/chat ожидает сырой base64; префикс data:image/...;base64, ломает приём."""
    s = (s or "").strip()
    if not s:
        return ""
    if s.lower().startswith("data:"):
        comma = s.find(",")
        if comma >= 0:
            s = s[comma + 1 :].strip()
    return s


def ollama_generate_url(url: str) -> str:
    """Нормализуем к .../api/generate для текстового fallback."""
    s = url.strip().rstrip("/")
    if s.endswith("/api/generate"):
        return s
    if s.endswith("/api/chat"):
        return s.replace("/api/chat", "/api/generate")
    if s.endswith("/generate"):
        return s
    if "/api/" in s:
        return s.rsplit("/", 1)[0] + "/generate"
    return s + "/api/generate"


def ollama_chat_url(generate_or_chat_url: str) -> str:
    u = generate_or_chat_url.strip().rstrip("/")
    if u.endswith("/api/generate"):
        return u[: -len("/generate")] + "/chat"
    if u.endswith("/generate") and "/api/" in u:
        return u.rsplit("/", 1)[0] + "/chat"
    if "/api/chat" in u:
        return u
    if "/api/" not in u:
        return u + "/api/chat"
    return u + "/chat" if not u.endswith("/chat") else u


def allowed_target_ids(variable_ids: list[str]) -> set[str]:
    """Цели рёбер slot→*: всё кроме слотов (и произвольные id графа)."""
    return {str(v) for v in variable_ids if not str(v).startswith("slot_")}


def normalize_phys_target(name: str, allowed: set[str]) -> str | None:
    n = name.strip()
    if not n:
        return None
    if n in allowed:
        return n
    if n.startswith("phys_") and n in allowed:
        return n
    cand = f"phys_{n}"
    if cand in allowed:
        return cand
    return None


def validate_slot_labels(
    obj: dict[str, Any],
    slot_ids: list[str],
    allowed_targets: set[str],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    slot_set = set(slot_ids)
    for k, v in obj.items():
        ks = str(k).strip()
        if ks not in slot_set:
            continue
        if not isinstance(v, dict):
            continue
        label = str(v.get("label", "")).strip()[:160] or "?"
        try:
            conf = float(v.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        conf = max(0.0, min(1.0, conf))
        raw_lp = v.get("likely_phys") or v.get("phys") or []
        if isinstance(raw_lp, str):
            raw_lp = [raw_lp]
        likely: list[str] = []
        seen: set[str] = set()
        if isinstance(raw_lp, list):
            for p in raw_lp:
                t = normalize_phys_target(str(p).strip(), allowed_targets)
                if t and t not in seen:
                    seen.add(t)
                    likely.append(t)
        out[ks] = {"label": label, "likely_phys": likely, "confidence": conf}
    return out


def build_multimodal_prompt(slot_ids: list[str], allowed_csv: str) -> str:
    return f"""You are labeling object-centric attention slots for a robotics / humanoid view.

Image order in this request:
  1) Full camera frame (robot first-person).
  2+) Small heatmap thumbnails: each corresponds to ONE slot in order slot_0, slot_1, ... (brighter = stronger attention in that region).

Slots (use EXACT keys): {", ".join(slot_ids)}

Allowed target variable names for "likely_phys" (copy EXACTLY from this list, subset only):
{allowed_csv}

Task: For each slot, give a short English label (object or body part), 0-1 confidence, and 0-3 likely_phys from the allowed list.

Return ONLY a single JSON object, no markdown, shape:
{{"slot_0":{{"label":"...","likely_phys":["phys_lknee"],"confidence":0.7}}, ...}}

Rules:
- Include every slot key listed above (use empty likely_phys and low confidence if unsure).
- likely_phys MUST use strings exactly from the allowed list (e.g. phys_lknee, com_z).
- confidence in [0,1]."""


def build_text_fallback_prompt(
    slot_ids: list[str],
    slot_values: list[float],
    variability: list[float],
    allowed_csv: str,
) -> str:
    lines = []
    for i, sid in enumerate(slot_ids):
        sv = slot_values[i] if i < len(slot_values) else 0.0
        va = variability[i] if i < len(variability) else 0.0
        lines.append(f"  {sid}: activation={sv:.3f}, variability={va:.4f}")
    body = "\n".join(lines)
    return f"""Robot visual slots (numeric summary only — no images).

{body}

Allowed variable names for likely_phys (exact strings):
{allowed_csv}

Return ONLY a JSON object mapping each slot to label, likely_phys (0-3 items from allowed list), confidence in [0,1].
Shape: {{"slot_0":{{"label":"...","likely_phys":[],"confidence":0.5}}, ...}}
Include all slots: {", ".join(slot_ids)}."""


async def _ollama_chat_multimodal(
    chat_url: str,
    model: str,
    prompt: str,
    images_b64: list[str],
    timeout: float = 300.0,
) -> str:
    images_clean = [normalize_ollama_image_b64(x) for x in images_b64]
    images_clean = [x for x in images_clean if x]
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": images_clean}],
        "stream": False,
        **ollama_think_disabled_payload(),
        "options": {
            "temperature": 0.2,
            "num_predict": _vlm_num_predict(),
        },
        **ollama_json_format_teacher_vlm_payload(),
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(chat_url, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        msg = data.get("message") or {}
        return (msg.get("content") or data.get("response") or "").strip()


async def _ollama_generate_text(
    generate_url: str,
    model: str,
    prompt: str,
    timeout: float = 1000.0,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        **ollama_think_disabled_payload(),
        "options": {
            "temperature": 0.2,
            "num_predict": _vlm_num_predict(),
        },
        **ollama_json_format_teacher_vlm_payload(),
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(generate_url, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        return (resp.json().get("response") or "").strip()


async def run_slot_vlm_labeling(
    *,
    frame_b64: str | None,
    masks_b64: list[str],
    slot_values: list[float],
    variability: list[float],
    n_slots: int,
    variable_ids: list[str],
    llm_url: str,
    llm_model: str,
    max_mask_images: int = 4,
    text_only: bool = False,
) -> tuple[dict[str, dict[str, Any]], str, str | None]:
    """
    Возвращает (validated_labels, mode, error).
    mode: "multimodal" | "text" | "failed"
    """
    slot_ids = [f"slot_{k}" for k in range(n_slots)]
    allowed = allowed_target_ids(variable_ids)
    allowed_list = sorted(allowed)
    if allowed_list:
        allowed_csv = ", ".join(allowed_list[:80])
        if len(allowed_list) > 80:
            allowed_csv += ", ..."
    else:
        allowed_csv = (
            "(no non-slot variables — visual-only graph; use likely_phys: [] for every slot)"
        )

    gen_url = ollama_generate_url(llm_url)
    chat_url = ollama_chat_url(gen_url)

    slot_key_set = set(slot_ids)

    def _finalize(raw_text: str) -> tuple[dict[str, dict[str, Any]], str | None]:
        obj = parse_json_object_loose(raw_text)
        if not obj or not (slot_key_set & obj.keys()):
            alt = scan_json_objects_having_any_key(raw_text, slot_key_set)
            if alt:
                obj = alt
        if not obj:
            return {}, "no JSON object in model response"
        validated = validate_slot_labels(obj, slot_ids, allowed)
        if not validated:
            alt = scan_json_objects_having_any_key(raw_text, slot_key_set)
            if alt is not None and alt is not obj:
                validated = validate_slot_labels(alt, slot_ids, allowed)
            if not validated:
                return {}, "no valid slot entries after validation"
        return validated, None

    if not text_only and frame_b64:
        images = [frame_b64]
        for m in masks_b64[: max(0, max_mask_images)]:
            if m:
                images.append(m)
        prompt = build_multimodal_prompt(slot_ids, allowed_csv)
        try:
            raw = await _ollama_chat_multimodal(chat_url, llm_model, prompt, images)
            val, err = _finalize(raw)
            if err is None:
                return val, "multimodal", None
            # fall through to text
            text_err = err
        except Exception as e:
            text_err = str(e) or repr(e)
            raw = ""
        # Text fallback after multimodal failure
        try:
            tprompt = build_text_fallback_prompt(
                slot_ids, slot_values, variability, allowed_csv
            )
            raw2 = await _ollama_generate_text(gen_url, llm_model, tprompt)
            val2, err2 = _finalize(raw2)
            if err2 is None:
                return val2, "text", f"multimodal_failed:{text_err}"
            return {}, "failed", f"{text_err}; text:{err2}"
        except Exception as e2:
            print(f"[VLM] text fallback exception:\n{traceback.format_exc()}")
            return {}, "failed", f"{text_err}; text_exc:{e2!r}"

    # Explicit text-only
    try:
        tprompt = build_text_fallback_prompt(
            slot_ids, slot_values, variability, allowed_csv
        )
        raw = await _ollama_generate_text(gen_url, llm_model, tprompt)
        val, err = _finalize(raw)
        if err is None:
            return val, "text", None
        return {}, "failed", err
    except Exception as e:
        print(f"[VLM] text-only exception:\n{traceback.format_exc()}")
        return {}, "failed", repr(e)


def weak_slot_to_phys_edges(
    labels: dict[str, dict[str, Any]],
    *,
    min_confidence: float = 0.62,
    weight: float = 0.055,
    alpha: float = 0.04,
    max_edges: int = 16,
) -> list[dict[str, Any]]:
    """Очень слабые рёбра slot_i → phys для inject_text_priors (Фаза 2, опция)."""
    out: list[dict[str, Any]] = []
    for sid, meta in labels.items():
        if not str(sid).startswith("slot_"):
            continue
        try:
            conf = float(meta.get("confidence", 0))
        except (TypeError, ValueError):
            conf = 0.0
        if conf < min_confidence:
            continue
        for tgt in meta.get("likely_phys") or []:
            if len(out) >= max_edges:
                return out
            out.append(
                {
                    "from_": sid,
                    "to": str(tgt),
                    "weight": weight,
                    "alpha": alpha,
                }
            )
    return out
