"""
Фаза 3 — виртуальный учитель: LLM → правила для System1 (IG-бонус) + мягкие дельты Value Layer.

Один вызов Ollama /api/generate, строгий JSON. Дельты VL накладываются с TTL тиков и клипами в value_layer.merge_teacher_vl.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from engine.value_layer import TeacherVLOverlay


@dataclass
class TeacherIGRule:
    """Если условие по when_var выполнено, do(target_var=…) получает бонус к actual_ig (× teacher_weight)."""

    target_var: str
    when_var: str | None
    when_min: float | None
    when_max: float | None
    bonus: float


def _parse_json_object(raw: str) -> dict[str, Any] | None:
    dec = json.JSONDecoder()
    i = 0
    while True:
        j = raw.find("{", i)
        if j < 0:
            return None
        try:
            obj, _end = dec.raw_decode(raw, j)
        except json.JSONDecodeError:
            i = j + 1
            continue
        if isinstance(obj, dict):
            return obj
        i = j + 1


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
    return f"""You are a cautious robotics curriculum advisor for a causal discovery humanoid agent.

{digest}

Return ONLY a JSON object with this shape (no markdown):
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


def parse_phase3_response(
    raw_text: str,
    valid_vars: set[str],
    current_tick: int,
) -> tuple[list[TeacherIGRule], TeacherVLOverlay | None]:
    obj = _parse_json_object(raw_text)
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
    timeout: float = 120.0,
) -> tuple[list[TeacherIGRule], TeacherVLOverlay | None, str | None]:
    """
    Возвращает (rules, vl_overlay, error).
    """
    prompt = build_phase3_prompt(digest)
    payload = {
        "model": llm_model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.18, "num_predict": 1600},
    }
    url = llm_url.strip().rstrip("/")
    if not url.endswith("/generate"):
        if "/api/" not in url:
            url = url + "/api/generate"
        elif not url.endswith("/generate"):
            url = url.rsplit("/", 1)[0] + "/generate"

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(url, json=payload)
            if resp.status_code != 200:
                return [], None, f"HTTP {resp.status_code}: {resp.text[:200]}"
            raw = (resp.json().get("response") or "").strip()
        except Exception as e:
            return [], None, str(e)

    rules, ov = parse_phase3_response(raw, valid_vars, current_tick)
    if not rules and ov is None:
        return [], None, "no parseable ig_rules/vl_overlay"
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
