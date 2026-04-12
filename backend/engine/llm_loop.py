"""
Этап D — LLM/VLM в петле (не только bootstrap).

Уровень 2: контрфактуальная консультация — не «дай рёбра», а
«наблюдаю X, ожидал Y — что объясняет расхождение?» → JSON с кандидатными рёбрами + объяснение.

Уровень 3: редкая перезапись гипотез о структуре (тот же контракт, что humanoid_structured, sync HTTP).
"""
from __future__ import annotations

import json
import os
from typing import Any

import httpx

from engine.llm_json_extract import ollama_json_format_payload, parse_json_object_loose
from engine.ollama_env import ollama_think_disabled_payload
from engine.rag_seeder import (
    _dedupe_cap_edges,
    _parse_json_array_from_llm_text,
    humanoid_urdf_digest,
)


def _prompt_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x) if x is not None else default
    except (TypeError, ValueError):
        return default


def _llm_loop_http_timeout() -> float:
    """Ollama generate for L2/L3; large models need generous read timeout (seconds)."""
    try:
        v = float(os.environ.get("RKK_LLM_LOOP_TIMEOUT", "900"))
    except ValueError:
        v = 900.0
    return max(60.0, min(v, 7200.0))


def _normalize_generate_url(llm_url: str) -> str:
    url = llm_url.strip().rstrip("/")
    if url.endswith("/generate"):
        return url
    if "/api/" not in url:
        return url + "/api/generate"
    return url.rsplit("/", 1)[0] + "/generate"


def build_counterfactual_prompt(ctx: dict[str, Any]) -> str:
    ids = list(ctx.get("variable_ids") or [])
    allowed = ", ".join(ids)
    if len(allowed) > 3500:
        allowed = allowed[:3500] + " …"
    triggers = ", ".join(ctx.get("triggers") or [])
    pred = json.dumps(ctx.get("cf_predicted") or {}, ensure_ascii=False)[:2800]
    obs = json.dumps(ctx.get("cf_observed") or {}, ensure_ascii=False)[:2800]
    slot_line = (ctx.get("slot_lexicon") or "")[:500]
    fall_history = (ctx.get("fall_history") or "")[:4000]
    curriculum_stage = (ctx.get("curriculum_stage") or "")[:200]
    temporal_context = (ctx.get("temporal_context") or "")[:4000]
    reward_breakdown = json.dumps(ctx.get("reward_breakdown") or {}, ensure_ascii=False)[
        :2800
    ]
    proprio_abstracts = json.dumps(ctx.get("proprio_abstracts") or {}, ensure_ascii=False)[
        :2000
    ]
    inner_voice_concepts = (ctx.get("inner_voice_concepts") or "")[:1200]
    inner_voice_verbal = (ctx.get("inner_voice_verbal") or "")[:800]
    return f"""You advise a robot that learns a causal world model (GNN) by experimenting.

The agent does NOT want a raw graph dump. It needs counterfactual reasoning.

Situation digest:
- Last action: do({ctx.get("variable", "?")}={_prompt_float(ctx.get("value", 0)):.4f})
- Prediction error (mean |pred-obs| over nodes): {_prompt_float(ctx.get("prediction_error", 0)):.5f}
- Discovery rate (fraction of GT-like edges found): {_prompt_float(ctx.get("discovery_rate", 0)):.4f}
- Value-layer block rate (cumulative): {_prompt_float(ctx.get("block_rate", 0)):.4f}
- Triggers that fired: {triggers or "none"}

After the action, model predicted (subset of node ids):
{pred}

Actually observed (same keys):
{obs}

Visual slot lexicon summary (if any): {slot_line or "none"}

Fall history and patterns:
{fall_history or "none"}

Current curriculum stage: {curriculum_stage or "none"}

Temporal hierarchy context (multi-scale state):
{temporal_context or "none"}

Reward breakdown (last tick):
{reward_breakdown or "none"}

Proprioception abstracts:
{proprio_abstracts or "none"}

Robot inner voice (fast recurrent head, not the LLM):
{inner_voice_concepts or "none"}

Last τ3 teacher verbal (if any):
{inner_voice_verbal or "none"}

Question (answer with structured JSON only, no markdown):
"We observe these values after the intervention; the forward model expected something else.
What hidden mechanisms, missing mediators, non-stationarity, or graph structure could explain the gap?
What single experiment would best disambiguate?"

Variable ids you may reference in candidate_edges (EXACT strings only):
{allowed}

Return ONLY a JSON object with this shape:
{{
  "explanation": "<2-5 sentences, plain text>",
  "candidate_edges": [
    {{"from_": "<id>", "to": "<id>", "weight_hint": <0.0-1.0 strength>, "rationale": "<short>"}}
  ],
  "next_probe_suggestion": "<one short sentence>"
}}

Rules:
- 0–12 candidate_edges; only from_/to from the allowed list; no self-loops.
- weight_hint is exploratory strength, not a physical constant.
"""


def parse_counterfactual_response(
    raw_text: str,
    valid_vars: set[str],
) -> tuple[str, list[dict[str, Any]], str]:
    """
    Returns (explanation, edges_for_inject, next_probe).
    edges_for_inject: dicts with from_, to, weight (scaled for inject_text_priors).
    """
    obj = parse_json_object_loose(raw_text)
    if not obj:
        return "", [], ""
    expl = str(obj.get("explanation", "")).strip()
    nxt = str(obj.get("next_probe_suggestion", "")).strip()
    edges_out: list[dict[str, Any]] = []
    for e in obj.get("candidate_edges") or []:
        if not isinstance(e, dict):
            continue
        fr = str(e.get("from_") or e.get("from") or "").strip()
        to = str(e.get("to") or "").strip()
        if not fr or not to or fr == to:
            continue
        if fr not in valid_vars or to not in valid_vars:
            continue
        try:
            wh = float(e.get("weight_hint", e.get("weight", 0.22)))
        except (TypeError, ValueError):
            wh = 0.22
        wh = max(0.0, min(1.0, wh))
        w_scaled = min(0.28, max(0.08, wh * 0.32))
        edges_out.append({"from_": fr, "to": to, "weight": w_scaled, "alpha": 0.05})
    return expl, edges_out, nxt


def consult_counterfactual_sync(
    llm_url: str,
    llm_model: str,
    ctx: dict[str, Any],
    valid_vars: set[str],
    timeout: float | None = None,
) -> dict[str, Any]:
    """Synchronous Ollama /api/generate call (for background thread)."""
    if timeout is None:
        timeout = _llm_loop_http_timeout()
    prompt = build_counterfactual_prompt(ctx)
    url = _normalize_generate_url(llm_url)
    payload = {
        "model": llm_model,
        "prompt": prompt,
        "stream": False,
        **ollama_think_disabled_payload(),
        "options": {
            "temperature": float(os.environ.get("RKK_LLM_LOOP_TEMP", "0.22")),
        },
        **ollama_json_format_payload(),
    }
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=payload)
    except httpx.TimeoutException:
        return {"ok": False, "error": "timed out"}
    if resp.status_code != 200:
        return {
            "ok": False,
            "error": f"HTTP {resp.status_code}: {resp.text[:240]}",
        }
    raw = (resp.json().get("response") or "").strip()
    expl, edges, nxt = parse_counterfactual_response(raw, valid_vars)
    return {
        "ok": True,
        "explanation": expl,
        "candidate_edges": edges,
        "next_probe": nxt,
        "raw_chars": len(raw),
    }


def build_structure_revision_prompt(var_names: list[str]) -> str:
    digest = humanoid_urdf_digest()
    var_json = json.dumps(var_names, ensure_ascii=False)
    return f"""You are revising causal hypotheses for a humanoid agent whose model stagnated.

Variables (use EXACT strings for from_ and to):
{var_json}

URDF / kinematic hint (names for intuition only):
{digest}

Task: Output directed edges to refresh exploration priors (not final truth).
Include self_* intention nodes if present in the variable list, linked to shoulders/cubes where plausible.

Valid JSON only: either a JSON array of edges, or {{"edges":[...]}} with the same elements.

Rules:
- 12–28 edges, diverse, no self-loops, from_/to must appear in the variable list.
- weight in [-1,1]."""


def structure_revision_sync(
    llm_url: str,
    llm_model: str,
    var_names: list[str],
    timeout: float | None = None,
) -> dict[str, Any]:
    """Уровень 3: редкая полная перезапись списка гипотез (как bootstrap, sync)."""
    if timeout is None:
        timeout = _llm_loop_http_timeout()
    url = _normalize_generate_url(llm_url)
    prompt = build_structure_revision_prompt(var_names)
    payload = {
        "model": llm_model,
        "prompt": prompt,
        "stream": False,
        **ollama_think_disabled_payload(),
        "options": {"temperature": 0.14},
        **ollama_json_format_payload(),
    }
    valid = set(var_names)
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=payload)
    except httpx.TimeoutException:
        return {"ok": False, "error": "timed out", "edges": []}
    if resp.status_code != 200:
        return {"ok": False, "error": f"HTTP {resp.status_code}", "edges": []}
    raw = (resp.json().get("response") or "").strip()
    arr = _parse_json_array_from_llm_text(raw)
    if not arr:
        return {"ok": False, "error": "no JSON array in response", "edges": []}
    pairs: list[tuple[str, str, float]] = []
    for e in arr:
        if not isinstance(e, dict):
            continue
        fr = e.get("from_") or e.get("from")
        to = e.get("to")
        if not fr or not to:
            continue
        fr, to = str(fr).strip(), str(to).strip()
        if fr not in valid or to not in valid:
            continue
        w = float(e.get("weight", 0.22))
        pairs.append((fr, to, w))
    capped = _dedupe_cap_edges(pairs, valid, max_total=36, max_per_source=6)
    edges = []
    for a, b, c in capped:
        mag = min(0.28, max(0.08, abs(c) * 0.35))
        w = mag * (1.0 if c >= 0 else -1.0)
        edges.append({"from_": a, "to": b, "weight": w, "alpha": 0.055})
    return {"ok": True, "edges": edges, "error": None}
