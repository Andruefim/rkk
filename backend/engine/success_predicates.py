"""
Goal-based success verification over observable predicates.

``evaluate_goal`` measures truth in observations (distance, contact, displacement,
state keys). PE-based verification is used only when the world model is trusted
and only over goal-relevant keys.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from engine.task_goal import GoalPredicate, TaskGoal

# Re-export legacy PE helpers for task_binding backward compatibility.
from engine.system2.success_predicates import (  # noqa: F401
    evaluate_macro_success,
    homeostatic_veto,
    prediction_error_total,
    resolve_max_prediction_error,
)


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _obs_f(obs: Mapping[str, Any], key: str, default: float = 0.5) -> float:
    v = obs.get(key, obs.get(f"phys_{key}", default))
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _resolve_ctx(ctx: Any) -> dict[str, Any]:
    """
    Normalize evaluation context.

    Expected keys (all optional):
      agent_xy: (x, y) — agent position in meters
      target_xy: (x, y) — target object position
      distance_m: float — precomputed agent↔target distance (overrides xy)
      contact: float in [0, 1] — contact signal for target
      displacement_m: float — XY displacement of manipulated object
      baseline_xy: (x, y) — object position at task start (for displace)
    """
    if callable(ctx):
        raw = ctx()
    elif isinstance(ctx, Mapping):
        raw = dict(ctx)
    else:
        raw = {}
    return raw


def _distance_m(ctx: dict[str, Any]) -> float | None:
    if "distance_m" in ctx:
        try:
            return float(ctx["distance_m"])
        except (TypeError, ValueError):
            return None
    agent = ctx.get("agent_xy")
    target = ctx.get("target_xy")
    if agent is None or target is None:
        return None
    try:
        ax, ay = float(agent[0]), float(agent[1])
        tx, ty = float(target[0]), float(target[1])
    except (TypeError, ValueError, IndexError):
        return None
    return float(((ax - tx) ** 2 + (ay - ty) ** 2) ** 0.5)


def _displacement_m(ctx: dict[str, Any]) -> float | None:
    if "displacement_m" in ctx:
        try:
            return float(ctx["displacement_m"])
        except (TypeError, ValueError):
            return None
    baseline = ctx.get("baseline_xy")
    target = ctx.get("target_xy")
    if baseline is None or target is None:
        return None
    try:
        bx, by = float(baseline[0]), float(baseline[1])
        tx, ty = float(target[0]), float(target[1])
    except (TypeError, ValueError, IndexError):
        return None
    return float(((tx - bx) ** 2 + (ty - by) ** 2) ** 0.5)


def _predicate_satisfaction(
    pred: GoalPredicate,
    obs: Mapping[str, Any],
    ctx: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """Return (satisfaction 0..1, detail dict)."""
    kind = str(pred.kind)
    detail: dict[str, Any] = {"kind": kind}

    if kind == "reduce_distance":
        dist = _distance_m(ctx)
        if dist is None:
            if "task_target_dist_m" in obs:
                dist = _obs_f(obs, "task_target_dist_m", default=2.0)
                detail["source"] = "obs_task_target_dist"
            else:
                dist = _obs_f(obs, "target_dist", default=2.0)
                detail["source"] = "obs_target_dist"
        else:
            detail["source"] = "ctx_distance_m"
        detail["distance_m"] = round(float(dist), 4)
        detail["target_m"] = float(pred.target_value)
        sat = 1.0 if dist <= float(pred.target_value) + float(pred.tolerance) else max(
            0.0, 1.0 - (dist - float(pred.target_value)) / max(float(pred.tolerance), 0.05)
        )
        return float(sat), detail

    if kind == "contact":
        contact = ctx.get("contact")
        if contact is None:
            contact = max(
                _obs_f(obs, "task_contact", 0.0),
                _obs_f(obs, "contact_signal", 0.0),
                _obs_f(obs, "grasp_contact", 0.0),
            )
            detail["source"] = "obs_contact"
        else:
            detail["source"] = "ctx_contact"
        try:
            cv = float(contact)
        except (TypeError, ValueError):
            cv = 0.0
        detail["contact"] = round(cv, 4)
        thr = float(pred.target_value) - float(pred.tolerance)
        sat = 1.0 if cv >= thr else max(0.0, cv / max(thr, 0.05))
        return float(sat), detail

    if kind == "displace":
        disp = _displacement_m(ctx)
        detail["displacement_m"] = round(float(disp), 4) if disp is not None else None
        detail["target_m"] = float(pred.target_value)
        if disp is None:
            return 0.0, {**detail, "reason": "no_displacement_ctx"}
        sat = 1.0 if disp >= float(pred.target_value) else max(
            0.0, disp / max(float(pred.target_value), 0.01)
        )
        return float(sat), detail

    if kind == "state_key":
        key = str(pred.key or "")
        av = _obs_f(obs, key, default=float("nan"))
        detail["key"] = key
        detail["obs"] = round(av, 4) if av == av else None
        detail["target"] = float(pred.target_value)
        if av != av:
            return 0.0, {**detail, "reason": "missing_key"}
        err = abs(float(av) - float(pred.target_value))
        detail["error"] = round(err, 4)
        sat = 1.0 if err <= float(pred.tolerance) else max(
            0.0, 1.0 - err / max(float(pred.tolerance) * 2.0, 0.05)
        )
        return float(sat), detail

    return 0.0, {**detail, "reason": "unknown_kind"}


def evaluate_goal(
    goal: TaskGoal,
    obs: Mapping[str, Any],
    ctx: Any = None,
) -> tuple[bool, float, dict[str, Any]]:
    """
  Verify a TaskGoal against current observations.

    ``ctx`` supplies scene geometry not always present in obs (see ``_resolve_ctx``).
    Integration code should pass agent_xy/target_xy/contact/displacement_m.

    Returns (satisfied, score, detail) where score is weighted-mean satisfaction in [0, 1].
    """
    ctx_d = _resolve_ctx(ctx)
    preds = list(goal.predicates)
    if not preds:
        return False, 0.0, {"reason": "no_predicates"}

    total_w = 0.0
    weighted = 0.0
    per_pred: list[dict[str, Any]] = []
    for p in preds:
        sat, pd = _predicate_satisfaction(p, obs, ctx_d)
        w = max(0.01, float(p.weight))
        total_w += w
        weighted += w * sat
        per_pred.append({**pd, "satisfaction": round(sat, 4), "weight": w})

    score = weighted / total_w if total_w > 0 else 0.0
    try:
        thr = float(os.environ.get("RKK_GOAL_SAT_THRESHOLD", "0.85"))
    except ValueError:
        thr = 0.85
    satisfied = score >= thr

    return bool(satisfied), float(score), {
        "score": round(score, 4),
        "threshold": thr,
        "predicates": per_pred,
        "wm_trusted": bool(goal.wm_trusted),
    }


def expected_state_keys_for_goal(goal: TaskGoal | None) -> list[str]:
    """Keys for narrowed PE verification when wm_trusted."""
    from engine.task_observation import task_observation_keys_for_goal

    return task_observation_keys_for_goal(goal)
