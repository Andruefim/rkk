from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
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
    expected_state: dict[str, float] = field(default_factory=dict)
    max_prediction_error: float | None = None
    skill_id: str | None = None

    def normalized_macro(self) -> str:
        m = str(self.macro or "").strip().upper()
        return m if m in MACRO_IDS else "IDLE"


@dataclass
class EpisodeSuccessSpec:
    """Intentional prior for episode end: expected observe snapshot + optional PE cap (LLM or adaptive)."""

    expected_state: dict[str, float] = field(default_factory=dict)
    max_prediction_error: float | None = None
    skill_id: str | None = None

    @classmethod
    def from_proposal(cls, p: System2Proposal | None) -> EpisodeSuccessSpec:
        if p is None:
            return cls()
        return cls(
            expected_state=dict(p.expected_state),
            max_prediction_error=p.max_prediction_error,
            skill_id=p.skill_id,
        )


@lru_cache(maxsize=1)
def expected_state_key_allowlist() -> frozenset[str]:
    """Широкий реестр ключей для intentional expected_state (VAR_NAMES + intents + phys_*)."""
    from engine.features.humanoid.constants import MOTOR_INTENT_VARS, VAR_NAMES

    s: set[str] = set(VAR_NAMES) | set(MOTOR_INTENT_VARS)
    for v in VAR_NAMES:
        s.add(f"phys_{v}")
    for inv in MOTOR_INTENT_VARS:
        s.add(f"phys_{inv}")
        if inv.startswith("intent_"):
            suf = inv[len("intent_") :]
            s.add(f"phys_intent_{suf}")
    return frozenset(s)


def filter_expected_state_raw(raw: dict[str, Any] | None) -> dict[str, float]:
    """Оставить только ключи из ``expected_state_key_allowlist()`` с числовыми значениями."""
    if not raw or not isinstance(raw, dict):
        return {}
    allow = expected_state_key_allowlist()
    out: dict[str, float] = {}
    for k, v in raw.items():
        sk = str(k).strip()
        if sk not in allow:
            continue
        try:
            out[sk] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def merge_episode_success_specs(
    primary: EpisodeSuccessSpec,
    secondary: EpisodeSuccessSpec | None,
) -> EpisodeSuccessSpec:
    """Curriculum/secondary заполняет только пустые поля; primary (LLM) перекрывает ключи."""
    if secondary is None:
        return primary
    if not secondary.expected_state:
        return EpisodeSuccessSpec(
            expected_state=dict(primary.expected_state),
            max_prediction_error=primary.max_prediction_error
            if primary.max_prediction_error is not None
            else secondary.max_prediction_error,
            skill_id=primary.skill_id or secondary.skill_id,
        )
    if not primary.expected_state:
        return EpisodeSuccessSpec(
            expected_state=dict(secondary.expected_state),
            max_prediction_error=primary.max_prediction_error
            if primary.max_prediction_error is not None
            else secondary.max_prediction_error,
            skill_id=primary.skill_id or secondary.skill_id,
        )
    merged = dict(secondary.expected_state)
    merged.update(primary.expected_state)
    return EpisodeSuccessSpec(
        expected_state=merged,
        max_prediction_error=primary.max_prediction_error
        if primary.max_prediction_error is not None
        else secondary.max_prediction_error,
        skill_id=primary.skill_id or secondary.skill_id,
    )


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


def _opt_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
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
    es_raw = raw.get("expected_state")
    es = filter_expected_state_raw(es_raw if isinstance(es_raw, dict) else None)
    mx = _opt_float(raw.get("max_prediction_error"))
    sid = raw.get("skill_id")
    skill_id = str(sid).strip()[:120] if sid is not None and str(sid).strip() else None
    return System2Proposal(
        macro=macro,
        goal=goal,
        intent_deltas=intent_deltas,
        rationale=rationale,
        expected_state=es,
        max_prediction_error=mx,
        skill_id=skill_id,
    )


def parse_recovery_motor_steps(
    raw: dict[str, Any] | None,
    *,
    max_steps: int = 10,
    max_ticks_per_step: int = 96,
) -> list[dict[str, Any]] | None:
    """
    Validate LLM recovery plan: ``{"steps": [{"ticks": int, "intent_deltas": {...}}, ...]}``.
    Returns list of dicts with int ticks and dict intent_deltas, or None if invalid.
    """
    if not raw or not isinstance(raw, dict):
        return None
    steps = raw.get("steps")
    if not isinstance(steps, list) or not steps:
        return None
    out: list[dict[str, Any]] = []
    for step in steps[: int(max(1, max_steps))]:
        if not isinstance(step, dict):
            return None
        try:
            nt = int(step.get("ticks", 0))
        except (TypeError, ValueError):
            return None
        nt = max(1, min(int(max_ticks_per_step), nt))
        deltas = step.get("intent_deltas") or step.get("deltas") or {}
        if not isinstance(deltas, dict):
            return None
        clean: dict[str, float] = {}
        for k, v in deltas.items():
            sk = str(k).strip()
            if not sk.startswith("intent_") and not sk.startswith("phys_intent_"):
                continue
            try:
                clean[sk] = float(v)
            except (TypeError, ValueError):
                continue
        out.append({"ticks": nt, "intent_deltas": clean})
    return out or None


def parse_recovery_llm_plan(
    raw: dict[str, Any] | None,
    *,
    max_steps: int = 10,
    max_ticks_per_step: int = 96,
) -> tuple[list[dict[str, Any]], dict[str, float], float | None] | None:
    """
    Полный recovery JSON: ``steps`` + опционально ``expected_state``, ``max_prediction_error``.
    Возвращает (steps, expected_state, max_pe) или None если steps невалидны.
    """
    steps = parse_recovery_motor_steps(
        raw, max_steps=max_steps, max_ticks_per_step=max_ticks_per_step
    )
    if not raw or not isinstance(raw, dict) or steps is None:
        return None
    es = filter_expected_state_raw(raw.get("expected_state"))
    mx = _opt_float(raw.get("max_prediction_error"))
    return (steps, es, mx)
