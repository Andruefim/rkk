from __future__ import annotations

import os
from typing import Any

from engine.system2.schema import MACRO_IDS, GoalSpec, System2Proposal


def _clip01(x: float) -> float:
    return float(max(0.05, min(0.95, float(x))))


def clip_intent_deltas(
    deltas: dict[str, Any] | None,
    *,
    max_abs: float | None = None,
) -> dict[str, float]:
    if not deltas:
        return {}
    try:
        lim = float(os.environ.get("RKK_SYSTEM2_MAX_INTENT_DELTA", "0.12"))
    except ValueError:
        lim = 0.12
    if max_abs is not None:
        lim = min(lim, float(max_abs))
    lim = float(max(0.0, min(0.35, lim)))
    out: dict[str, float] = {}
    for k, v in deltas.items():
        sk = str(k).strip()
        if not (sk.startswith("intent_") or sk.startswith("phys_intent_")):
            continue
        try:
            dv = float(v)
        except (TypeError, ValueError):
            continue
        out[sk] = float(max(-lim, min(lim, dv)))
    return out


def validate_proposal(
    proposal: System2Proposal | None,
    *,
    rationale_max: int = 600,
    allowed_intent_keys: frozenset[str] | None = None,
) -> System2Proposal | None:
    if proposal is None:
        return None
    macro = proposal.normalized_macro()
    if macro not in MACRO_IDS:
        macro = "IDLE"
    g = proposal.goal
    goal = GoalSpec(
        com_z_min=_clip01(g.com_z_min) if g.com_z_min is not None else None,
        posture_stability_min=_clip01(g.posture_stability_min)
        if g.posture_stability_min is not None
        else None,
        target_dist_max=_clip01(g.target_dist_max)
        if g.target_dist_max is not None
        else None,
    )
    deltas = clip_intent_deltas(proposal.intent_deltas)
    if allowed_intent_keys is not None:
        deltas = {
            k: v for k, v in deltas.items() if k in allowed_intent_keys
        }
    rat = str(proposal.rationale or "")[: int(max(32, rationale_max))]
    return System2Proposal(macro=macro, goal=goal, intent_deltas=deltas, rationale=rat)
