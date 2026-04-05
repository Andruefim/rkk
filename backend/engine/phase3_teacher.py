"""
Фаза 3 — виртуальный учитель: LLM → правила для System1 (IG-бонус) + мягкие дельты Value Layer.

Один вызов Ollama /api/generate; разбор через llm_json_extract + поиск объекта с ig_rules/vl_overlay.
По умолчанию без format=json (RKK_OLLAMA_JSON_FORMAT_TEACHER_VLM). Дельты VL — merge_teacher_vl.
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any

import httpx

from engine.llm_json_extract import (
    ollama_json_format_teacher_vlm_payload,
    parse_json_object_loose,
    scan_json_objects_having_any_key,
)
from engine.ollama_env import ollama_think_disabled_payload
from engine.value_layer import TeacherVLOverlay


@dataclass
class TeacherIGRule:
    """Если условие по when_var выполнено, do(target_var=…) получает бонус к actual_ig (× teacher_weight)."""

    target_var: str
    when_var: str | None
    when_min: float | None
    when_max: float | None
    bonus: float


def _slot_lexicon_summary(visual_env) -> str:
    if visual_env is None:
        return ""
    lex = getattr(visual_env, "_slot_lexicon", None) or {}
    if not lex:
        return ""
    parts = []
    for k in sorted(lex.keys(), key=lambda x: int(x.split("_")[1]) if "_" in x else 0)[:12]:
        e = lex.get(k) or {}
        lab = e.get("label")
        if lab:
            parts.append(f"{k}:{lab}")
    return "; ".join(parts)[:400]


def build_phase3_digest(
    *,
    variable_ids: list[str],
    nodes: dict[str, float],
    phase_idx: int,
    phase_name: str,
    fallen: bool,
    block_rate: float,
    total_interventions: int,
    top_uncertain_vars: list[tuple[str, float]],
    slot_lexicon: str,
) -> str:
    allowed = ", ".join(variable_ids[:60])
    if len(variable_ids) > 60:
        allowed += ", …"
    node_sample = {k: round(float(v), 3) for k, v in list(nodes.items())[:35]}
    unc_lines = [f"  {v}: u={u:.3f}" for v, u in top_uncertain_vars[:5]]
    unc_body = "\n".join(unc_lines) if unc_lines else "  (n/a)"
    slot_line = f"Slot labels (if any): {slot_lexicon or 'none'}\n"
    return f"""Simulation digest:
phase={phase_idx} ({phase_name}) tick_context
fallen={fallen} vl_block_rate={block_rate:.3f} interventions={total_interventions}

Variable ids (use ONLY these exact strings):
{allowed}

Current node values (subset):
{json.dumps(node_sample, ensure_ascii=False)}

Most uncertain vars (by edge uncertainty proxy):
{unc_body}
{slot_line}"""


def build_phase3_prompt(digest: str) -> str:
    return f"""CRITICAL OUTPUT RULE: Reply with NOTHING except one JSON object. No markdown, no headings, no "Final Answer", no analysis.
The first non-whitespace character of your reply MUST be `{{`.

You are a cautious robotics curriculum advisor for a causal discovery humanoid agent.

{digest}

Return valid JSON only (one object). Shape:
{{
  "ig_rules": [
    {{
      "target_var": "<id — variable the agent intervenes on (do target)>",
      "when_var": "<id or null>",
      "when_min": <number 0-1 or null>,
      "when_max": <number 0-1 or null>,
      "bonus": <0.05-0.22 — extra training signal weight when rule matches>
    }}
  ],
  "vl_overlay": {{
    "ttl_ticks": <int 200-1200 — how long overlay applies>,
    "phi_min_delta": <small float -0.03..0.02>,
    "env_entropy_max_delta": <float -0.1..0.12 — added to entropy jump limit, positive = more permissive>,
    "h_slow_max_delta": <float -1..1>,
    "predict_lo_delta": <float -0.04..0.02 — negative widens lower prediction bound>,
    "predict_hi_delta": <float -0.02..0.04>,
    "entropy_spike_phi_low_delta": <float -0.08..0.12>,
    "entropy_spike_autonomy_delta": <float -0.08..0.12>
  }}
}}

Rules:
- 2–6 ig_rules. Prefer exploring leg/torso joints when com_z is healthy; caution when com_z is very low (fallen risk).
- target_var must be a variable the agent can do(); use exact ids from the list.
- when_var null means rule always applies to target_var (use sparingly, low bonus).
- vl_overlay: mild adjustments only; ttl_ticks reasonable for early training help."""

_PHASE3_ROOT_KEYS = frozenset({"ig_rules", "vl_overlay"})


def _phase3_num_predict() -> int:
    try:
        v = int(os.environ.get("RKK_PHASE3_NUM_PREDICT", "2400"))
    except ValueError:
        v = 2400
    return max(256, min(v, 8192))


def _phase3_http_timeout() -> float:
    try:
        v = float(os.environ.get("RKK_PHASE3_HTTP_TIMEOUT", "300"))
    except ValueError:
        v = 300.0
    return max(45.0, min(v, 3600.0))


def _phase3_parsed_root_usable(obj: dict[str, Any] | None) -> bool:
    if not isinstance(obj, dict) or not obj:
        return False
    if "@rev" in obj:
        return False
    if "ig_rules" in obj or "vl_overlay" in obj:
        return True
    if obj.get("type") == "object" and "properties" in obj:
        return False
    return False


def parse_phase3_response(
    raw_text: str,
    valid_vars: set[str],
    current_tick: int,
) -> tuple[list[TeacherIGRule], TeacherVLOverlay | None]:
    obj = parse_json_object_loose(raw_text)
    if not _phase3_parsed_root_usable(obj):
        obj = scan_json_objects_having_any_key(raw_text, set(_PHASE3_ROOT_KEYS))
    if not obj:
        return [], None

    rules_out: list[TeacherIGRule] = []
    for r in obj.get("ig_rules") or []:
        if not isinstance(r, dict):
            continue
        tv = str(r.get("target_var", "")).strip()
        if tv not in valid_vars:
            continue
        wv = r.get("when_var")
        wv = str(wv).strip() if wv not in (None, "", "null") else None
        if wv is not None and wv not in valid_vars:
            wv = None
        try:
            wm = r.get("when_min")
            wmin = float(wm) if wm is not None and str(wm).lower() != "null" else None
        except (TypeError, ValueError):
            wmin = None
        try:
            wx = r.get("when_max")
            wmax = float(wx) if wx is not None and str(wx).lower() != "null" else None
        except (TypeError, ValueError):
            wmax = None
        try:
            bonus = float(r.get("bonus", 0.1))
        except (TypeError, ValueError):
            bonus = 0.1
        bonus = max(0.02, min(0.24, bonus))
        rules_out.append(
            TeacherIGRule(
                target_var=tv,
                when_var=wv,
                when_min=wmin,
                when_max=wmax,
                bonus=bonus,
            )
        )

    vl_part = obj.get("vl_overlay")
    overlay: TeacherVLOverlay | None = None
    if isinstance(vl_part, dict):
        try:
            ttl = int(vl_part.get("ttl_ticks", 500))
        except (TypeError, ValueError):
            ttl = 500
        ttl = max(80, min(4000, ttl))

        def gf(key: str, default: float = 0.0) -> float:
            try:
                return float(vl_part.get(key, default))
            except (TypeError, ValueError):
                return default

        overlay = TeacherVLOverlay(
            expires_at_tick=current_tick + ttl,
            phi_min_delta=max(-0.05, min(0.03, gf("phi_min_delta"))),
            env_entropy_max_delta=max(-0.15, min(0.18, gf("env_entropy_max_delta"))),
            h_slow_max_delta=max(-2.0, min(2.0, gf("h_slow_max_delta"))),
            predict_lo_delta=max(-0.06, min(0.03, gf("predict_lo_delta"))),
            predict_hi_delta=max(-0.03, min(0.06, gf("predict_hi_delta"))),
            entropy_spike_phi_low_delta=max(-0.12, min(0.15, gf("entropy_spike_phi_low_delta"))),
            entropy_spike_autonomy_delta=max(-0.12, min(0.15, gf("entropy_spike_autonomy_delta"))),
        )

    return rules_out, overlay


async def fetch_phase3_teacher_bundle(
    *,
    llm_url: str,
    llm_model: str,
    digest: str,
    valid_vars: set[str],
    current_tick: int,
    timeout: float | None = None,
) -> tuple[list[TeacherIGRule], TeacherVLOverlay | None, str | None]:
    """
    Возвращает (rules, vl_overlay, error).
    """
    if timeout is None:
        timeout = _phase3_http_timeout()
    prompt = build_phase3_prompt(digest)
    payload = {
        "model": llm_model,
        "prompt": prompt,
        "stream": False,
        **ollama_think_disabled_payload(),
        "options": {
            "temperature": 0.1,
            "num_predict": _phase3_num_predict(),
        },
        **ollama_json_format_teacher_vlm_payload(),
    }
    url = llm_url.strip().rstrip("/")
    if not url.endswith("/generate"):
        if "/api/" not in url:
            url = url + "/api/generate"
        elif not url.endswith("/generate"):
            url = url.rsplit("/", 1)[0] + "/generate"

    raw = ""
    data: dict[str, Any] = {}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            for attempt in range(2):
                resp = await client.post(url, json=payload)
                if resp.status_code != 200:
                    return [], None, f"HTTP {resp.status_code}: {resp.text[:200]}"
                try:
                    data = resp.json()
                except json.JSONDecodeError as e:
                    return [], None, f"Ollama JSON body: {e}; text={resp.text[:300]!r}"
                raw = (data.get("response") or "").strip()
                if raw:
                    break
                err_o = data.get("error")
                if attempt == 0:
                    print(
                        "[Phase3] Ollama returned empty `response` "
                        f"(attempt 1). keys={list(data.keys())} error={err_o!r} "
                        f"eval_count={data.get('eval_count')!r} — retry in 3s…"
                    )
                    await asyncio.sleep(3.0)
            if not raw:
                print(
                    "[Phase3] Ollama still empty after retry. "
                    f"keys={list(data.keys())} error={data.get('error')!r} "
                    f"eval_count={data.get('eval_count')!r} "
                    f"body_preview={resp.text[:500]!r}"
                )
    except Exception as e:
        return [], None, str(e)

    rules, ov = parse_phase3_response(raw, valid_vars, current_tick)
    if not rules and ov is None:
        tail = (raw[-400:] if raw else "").replace("\n", " ")
        return [], None, f"no parseable ig_rules/vl_overlay; response_tail={tail!r}"
    return rules, ov, None


def top_uncertain_vars_from_agent(agent, k: int = 5) -> list[tuple[str, float]]:
    """По рёбрам графа: концы с наибольшей средней (1-alpha) неопределённостью."""
    import torch

    core = agent.graph._core
    if core is None or not agent.graph._node_ids:
        return []

    with torch.no_grad():
        A = core.alpha_trust_matrix().detach().float().cpu().numpy()
    nids = agent.graph._node_ids
    d = len(nids)
    if A.shape[0] != d:
        return []
    unc = 1.0 - A
    score: dict[str, float] = {}
    for i in range(d):
        s = float(unc[i].mean()) + float(unc[:, i].mean())
        score[nids[i]] = s * 0.5
    ranked = sorted(score.items(), key=lambda x: -x[1])
    return ranked[:k]
