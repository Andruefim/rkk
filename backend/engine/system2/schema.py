from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


MACRO_IDS = frozenset({"IDLE", "RECOVER_POSTURE", "LOCOMOTE_DELIVERY", "EXPLORE"})


@dataclass
class GoalSpec:
    """Абстрактные пороги в нормализованном [0,1] пространстве графа (как в observe)."""

    com_z_min: float | None = None
    posture_stability_min: float | None = None
    target_dist_max: float | None = None


@dataclass
class System2Proposal:
    macro: str
    goal: GoalSpec = field(default_factory=GoalSpec)
    intent_deltas: dict[str, float] = field(default_factory=dict)
    rationale: str = ""

    def normalized_macro(self) -> str:
        m = str(self.macro or "").strip().upper()
        return m if m in MACRO_IDS else "IDLE"


def _extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    t = text.strip()
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def proposal_from_dict(raw: dict[str, Any] | None) -> System2Proposal | None:
    if not raw:
        return None
    macro = str(raw.get("macro", "IDLE")).strip().upper()
    if macro not in MACRO_IDS:
        macro = "IDLE"
    g = raw.get("goal") or {}
    goal = GoalSpec(
        com_z_min=_opt_float(g.get("com_z_min")),
        posture_stability_min=_opt_float(g.get("posture_stability_min")),
        target_dist_max=_opt_float(g.get("target_dist_max")),
    )
    deltas = raw.get("intent_deltas") or raw.get("intents") or {}
    intent_deltas: dict[str, float] = {}
    if isinstance(deltas, dict):
        for k, v in deltas.items():
            sk = str(k).strip()
            if not sk.startswith("intent_") and not sk.startswith("phys_intent_"):
                continue
            try:
                intent_deltas[sk] = float(v)
            except (TypeError, ValueError):
                continue
    rationale = str(raw.get("rationale", raw.get("explanation", "")))[:500]
    return System2Proposal(
        macro=macro, goal=goal, intent_deltas=intent_deltas, rationale=rationale
    )


def _opt_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None
