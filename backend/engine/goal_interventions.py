"""Declarative predicate → motor do() interventions for WM imagination.

Physical consequences of goal predicates — not verb/keyword tables.
"""
from __future__ import annotations

import math

from engine.task_goal import GoalPredicate, TaskGoal

_APPROACH: dict[str, float] = {
    "intent_stride": 0.62,
    "intent_torso_forward": 0.55,
}
_REACH_VAL = 0.58
_GRASP_VAL = 0.45
_DISPLACE: dict[str, float] = {
    "intent_stride": 0.48,
    "intent_torso_forward": 0.55,
    "intent_lean_forward": 0.52,
}


def _normalize_xy(v: tuple[float, float]) -> tuple[float, float]:
    x, y = float(v[0]), float(v[1])
    n = math.hypot(x, y)
    if n < 1e-9:
        return 1.0, 0.0
    return x / n, y / n


def _reach_side_from_geometry(
    agent_xy: tuple[float, float] | None,
    target_xy: tuple[float, float] | None,
    agent_forward: tuple[float, float] | None,
) -> str | None:
    """Return ``left``, ``right``, or ``None`` (both arms)."""
    if agent_xy is None or target_xy is None:
        return None
    dx = float(target_xy[0]) - float(agent_xy[0])
    dy = float(target_xy[1]) - float(agent_xy[1])
    if math.hypot(dx, dy) < 1e-6:
        return None
    if agent_forward is not None:
        fx, fy = _normalize_xy(agent_forward)
        cross = fx * dy - fy * dx
        if abs(cross) < 0.05:
            return None
        return "left" if cross > 0 else "right"
    return None


def interventions_for_predicate(
    pred: GoalPredicate,
    *,
    agent_xy: tuple[float, float] | None = None,
    target_xy: tuple[float, float] | None = None,
    agent_forward: tuple[float, float] | None = None,
) -> dict[str, float]:
    """Map one predicate kind to intent_* do() values."""
    if pred.kind == "reduce_distance":
        return dict(_APPROACH)
    if pred.kind == "contact":
        side = _reach_side_from_geometry(agent_xy, target_xy, agent_forward)
        if side == "left":
            return {"intent_reach_left": _REACH_VAL, "intent_grasp": _GRASP_VAL}
        if side == "right":
            return {"intent_reach_right": _REACH_VAL, "intent_grasp": _GRASP_VAL}
        return {
            "intent_reach_left": _REACH_VAL,
            "intent_reach_right": _REACH_VAL,
            "intent_grasp": _GRASP_VAL,
        }
    if pred.kind == "displace":
        return dict(_DISPLACE)
    if pred.kind == "state_key" and pred.key and str(pred.key).startswith("intent_"):
        return {str(pred.key): float(pred.target_value)}
    return {}


def interventions_for_goal(
    goal: TaskGoal | None,
    *,
    agent_xy: tuple[float, float] | None = None,
    target_xy: tuple[float, float] | None = None,
    agent_forward: tuple[float, float] | None = None,
) -> dict[str, float]:
    """Merge interventions from all predicates (later preds override earlier keys)."""
    if goal is None or not goal.predicates:
        return {}
    motor: dict[str, float] = {}
    for pred in goal.predicates:
        motor.update(
            interventions_for_predicate(
                pred,
                agent_xy=agent_xy,
                target_xy=target_xy,
                agent_forward=agent_forward,
            )
        )
    return motor
