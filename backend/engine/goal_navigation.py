"""Closed-loop navigation to a point — P-controller, no object-type heuristics."""
from __future__ import annotations

import math

_STRIDE_MIN = 0.5
_STRIDE_MAX = 0.68
_TURN_STRIDE = 0.48
_TURN_COUPLING = 0.72
_HEADING_TURN_RAD = 0.12


def _normalize_xy(v: tuple[float, float]) -> tuple[float, float]:
    x, y = float(v[0]), float(v[1])
    n = math.hypot(x, y)
    if n < 1e-9:
        return 1.0, 0.0
    return x / n, y / n


def navigation_intents(
    agent_xy: tuple[float, float],
    agent_forward: tuple[float, float],
    target_xy: tuple[float, float],
    stop_distance: float,
    *,
    fallen: bool = False,
) -> dict[str, float]:
    """
    Per-tick motor intents steering agent toward ``target_xy``.

  Reuses the same turn / stride intents as ``GroundedLanguageController._motor_patch_for_tag("turn")``
    and locomote stride scaling — balance-critical fields are registered via the
    ``navigation`` arbiter source (not ``human_task`` bodysplit).
    """
    if fallen:
        return {}

    ax, ay = float(agent_xy[0]), float(agent_xy[1])
    tx, ty = float(target_xy[0]), float(target_xy[1])
    dx, dy = tx - ax, ty - ay
    dist = math.hypot(dx, dy)
    stop = max(0.05, float(stop_distance))

    if dist <= stop:
        return {}

    fx, fy = _normalize_xy(agent_forward)
    tcx, tcy = dx / dist, dy / dist

    cross = fx * tcy - fy * tcx
    dot = max(-1.0, min(1.0, fx * tcx + fy * tcy))
    heading_err = math.atan2(cross, dot)

    out: dict[str, float] = {}

    if abs(heading_err) > _HEADING_TURN_RAD:
        # Mirror turn tag: gait_coupling > 0.5 turns left, < 0.5 turns right.
        if heading_err > 0.0:
            out["intent_gait_coupling"] = _TURN_COUPLING
            out["intent_support_left"] = 0.62
            out["intent_support_right"] = 0.38
        else:
            out["intent_gait_coupling"] = 1.0 - _TURN_COUPLING
            out["intent_support_left"] = 0.38
            out["intent_support_right"] = 0.62
        out["intent_stride"] = _TURN_STRIDE
    else:
        # P-controller on stride: full speed far away, fades to min at stop_distance.
        span = max(dist - stop, 0.05)
        gain = min(1.0, span / max(dist, 1e-6))
        stride = _STRIDE_MIN + gain * (_STRIDE_MAX - _STRIDE_MIN)
        out["intent_stride"] = float(stride)
        out["intent_torso_forward"] = 0.55

    return out
