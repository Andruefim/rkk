"""
Recovery motor schedules: LLM plans, deterministic fallback, bundle enrichment.
"""
from __future__ import annotations

import os
from typing import Any

from engine.system2.macros import macro_bundle


def recovery_fallback_enabled() -> bool:
    return os.environ.get("RKK_S2_RECOVERY_FALLBACK", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def default_recovery_fallback_steps() -> list[dict[str, Any]]:
    """Deterministic multi-step schedule when LLM returns nothing invalid."""
    raw = os.environ.get("RKK_S2_RECOVERY_FALLBACK_JSON", "").strip()
    if raw:
        try:
            import json

            data = json.loads(raw)
            if isinstance(data, list) and data:
                return [_normalize_step(s) for s in data if isinstance(s, dict)]
        except Exception:
            pass
    bundle = macro_bundle("RECOVER_POSTURE")
    res = dict(bundle.get("residuals") or {})
    return [
        {
            "ticks": 22,
            "intent_deltas": {
                "intent_stop_recover": res.get("intent_stop_recover", 0.08),
                "intent_support_left": 0.06,
                "intent_support_right": 0.06,
            },
        },
        {
            "ticks": 28,
            "intent_deltas": {
                "intent_torso_forward": 0.08,
                "intent_arm_counterbalance": res.get("intent_arm_counterbalance", 0.05),
            },
        },
        {
            "ticks": 36,
            "intent_deltas": {
                "intent_torso_forward": 0.06,
                "intent_lean_forward": res.get("intent_lean_forward", 0.05),
                "intent_stop_recover": 0.04,
            },
        },
    ]


def _normalize_step(step: dict[str, Any]) -> dict[str, Any]:
    try:
        ticks = int(max(1, min(80, int(step.get("ticks", 12)))))
    except (TypeError, ValueError):
        ticks = 12
    deltas = step.get("intent_deltas") or step.get("deltas") or {}
    clean: dict[str, float] = {}
    if isinstance(deltas, dict):
        for k, v in deltas.items():
            sk = str(k).strip()
            if not sk.startswith("intent_") and not sk.startswith("phys_intent_"):
                continue
            try:
                clean[sk] = float(v)
            except (TypeError, ValueError):
                continue
    return {"ticks": ticks, "intent_deltas": clean}


def enrich_recovery_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fill empty intent_deltas from RECOVER bundle so schedule is executable."""
    bundle = macro_bundle("RECOVER_POSTURE")
    res = dict(bundle.get("residuals") or {})
    out: list[dict[str, Any]] = []
    for i, st in enumerate(steps):
        st = _normalize_step(st)
        d = dict(st.get("intent_deltas") or {})
        if not d and res:
            d = dict(res)
        elif not d and i == 0:
            d = {"intent_stop_recover": 0.07, "intent_torso_forward": 0.06}
        out.append({"ticks": st["ticks"], "intent_deltas": d})
    return out
