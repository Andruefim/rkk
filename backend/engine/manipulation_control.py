"""Geometry-aware manipulation intents — closed-loop reach toward resolved target."""
from __future__ import annotations

import math

from engine.goal_interventions import interventions_for_predicate
from engine.task_goal import GoalPredicate
from engine.task_observation import reach_start_m

_REACH_VAL = 0.58
_GRASP_VAL = 0.45


def _proximity_gain(dist: float, start_m: float) -> float:
    start = max(0.15, float(start_m))
    if dist >= start:
        return 0.0
    return float(max(0.35, min(1.0, 1.0 - dist / start)))


def manipulation_intents_from_bearing_range(
    bearing: float,
    range_m: float,
    *,
    reach_start: float | None = None,
    fallen: bool = False,
) -> dict[str, float]:
    """Reach/grasp from ego bearing + metric range (vision path)."""
    if fallen:
        return {}
    start = float(reach_start if reach_start is not None else reach_start_m())
    dist = float(range_m)
    if not math.isfinite(dist) or dist > start:
        return {}
    gain = _proximity_gain(dist, start)
    if gain <= 0.0:
        return {}

    b = float(max(-1.0, min(1.0, bearing)))
    reach_key = "intent_reach_right" if b >= 0.0 else "intent_reach_left"
    # Intents are centered at 0.5 (neutral). Scale ABOVE neutral by proximity —
    # multiplying absolute _REACH_VAL by gain previously produced <0.5 (retract)
    # for mid-band ranges (live: phys≈0.45 → reach≈0.22, arms never extend).
    reach_amt = float(max(0.56, min(0.94, 0.50 + 0.44 * gain)))
    grasp_amt = float(max(0.50, min(0.94, 0.50 + (_GRASP_VAL - 0.20) * gain)))
    return {
        reach_key: reach_amt,
        "intent_grasp": grasp_amt,
        "intent_head_yaw": float(max(0.0, min(1.0, 0.5 + 0.22 * b))),
        "vision_bearing": b,
        "vision_range_m": float(dist),
    }


def manipulation_intents(
    agent_xy: tuple[float, float],
    agent_forward: tuple[float, float],
    target_xy: tuple[float, float],
    dist: float,
    *,
    reach_start: float | None = None,
    fallen: bool = False,
) -> dict[str, float]:
    """
    Emit reach/grasp intents scaled by proximity to target.

    Uses cross-product arm side selection from goal_interventions (geometry, not verbs).
    """
    if fallen:
        return {}
    start = float(reach_start if reach_start is not None else reach_start_m())
    if float(dist) > start:
        return {}

    gain = _proximity_gain(float(dist), start)
    if gain <= 0.0:
        return {}

    pred = GoalPredicate(kind="contact", target_value=1.0, tolerance=0.5)
    base = interventions_for_predicate(
        pred,
        agent_xy=agent_xy,
        target_xy=target_xy,
        agent_forward=agent_forward,
    )
    if not base:
        return {}

    out: dict[str, float] = {}
    for k, v in base.items():
        if k.startswith("intent_reach_"):
            out[k] = float(max(0.0, min(0.94, float(v) * gain)))
        elif k == "intent_grasp":
            out[k] = float(max(0.0, min(0.94, _GRASP_VAL * gain)))
        else:
            out[k] = float(v)

    # Head bias toward target (continuous, not keyword routing).
    dx = float(target_xy[0]) - float(agent_xy[0])
    dy = float(target_xy[1]) - float(agent_xy[1])
    if math.hypot(dx, dy) > 1e-6:
        fx, fy = float(agent_forward[0]), float(agent_forward[1])
        n = math.hypot(fx, fy) + 1e-9
        fx, fy = fx / n, fy / n
        cross = fx * dy - fy * dx
        out["intent_head_yaw"] = float(max(0.0, min(1.0, 0.5 + 0.22 * math.tanh(cross * 3.0))))

    return out
