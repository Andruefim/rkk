"""Closed-loop navigation to a point — P-controller, no object-type heuristics."""
from __future__ import annotations

import math
import os

_STRIDE_MIN = 0.52
_STRIDE_MAX = 0.68
# Must exceed CPG walk_gate (~0.54) so legs advance while turning.
_TURN_STRIDE = 0.56
_TURN_COUPLING = 0.72
_HEADING_TURN_RAD = 0.12


def _ef(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return float(default)


def task_nav_pause_posture() -> float:
    return _ef("RKK_TASK_NAV_PAUSE_POSTURE", 0.32)


def task_nav_full_posture() -> float:
    return _ef("RKK_TASK_NAV_FULL_POSTURE", 0.55)


def posture_stride_scale(posture_stability: float) -> float:
    """
    Scale navigation stride by stance quality during human-task approach.
    Returns 0 when posture is too low to walk safely (brief stabilize pause).
    """
    ps = float(posture_stability)
    pause = task_nav_pause_posture()
    full = max(pause + 0.05, task_nav_full_posture())
    if ps >= full:
        return 1.0
    if ps <= pause:
        return 0.0
    if ps < 0.40:
        # Marginal band: short cautious steps.
        t = (ps - pause) / max(0.40 - pause, 1e-6)
        return float(0.30 + 0.25 * t)
    t = (ps - 0.40) / max(full - 0.40, 1e-6)
    return float(0.55 + 0.45 * t)


def apply_posture_to_navigation(
    intents: dict[str, float],
    posture_stability: float,
) -> tuple[dict[str, float], bool]:
    """
    Modulate navigation intents for task-aware balance.
    Returns (intents, nav_active). nav_active=False → brief stabilize pause.
    """
    if not intents:
        return {}, False
    scale = posture_stride_scale(posture_stability)
    if scale <= 0.0:
        return {}, False

    out = dict(intents)
    if "intent_stride" in out:
        base = float(out["intent_stride"])
        scaled = 0.5 + (base - 0.5) * scale
        if scale < 0.40:
            min_stride = 0.5
        elif scale < 0.70:
            min_stride = _TURN_STRIDE
        else:
            min_stride = _STRIDE_MIN
        out["intent_stride"] = float(max(min_stride, min(scaled, _STRIDE_MAX)))
    if "intent_torso_forward" in out:
        cap = 0.52 + 0.06 * scale
        out["intent_torso_forward"] = float(min(out["intent_torso_forward"], cap))
    return out, True


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
    posture_stability: float | None = None,
    prev_agent_xy: tuple[float, float] | None = None,
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

    closing = 0.0
    if prev_agent_xy is not None:
        px, py = float(prev_agent_xy[0]), float(prev_agent_xy[1])
        vx, vy = ax - px, ay - py
        closing = vx * tcx + vy * tcy
    drifting_away = closing < -0.002

    out: dict[str, float] = {}
    out["task_heading_err"] = float(max(-1.0, min(1.0, heading_err / math.pi)))
    out["task_closing_vel"] = float(max(-1.0, min(1.0, closing * 20.0)))

    turn_thr = _HEADING_TURN_RAD * (0.55 if drifting_away else 1.0)
    if abs(heading_err) > turn_thr or drifting_away:
        # Mirror turn tag: gait_coupling > 0.5 turns left, < 0.5 turns right.
        if heading_err > 0.0 or (drifting_away and abs(heading_err) < 1e-6 and cross >= 0.0):
            out["intent_gait_coupling"] = _TURN_COUPLING
            out["intent_support_left"] = 0.62
            out["intent_support_right"] = 0.38
        else:
            out["intent_gait_coupling"] = 1.0 - _TURN_COUPLING
            out["intent_support_left"] = 0.38
            out["intent_support_right"] = 0.62
        # Blend forward stride when still far — turn-in-place stalls approach otherwise.
        span = max(dist - stop, 0.05)
        gain = min(1.0, span / max(dist, 1e-6))
        fwd_stride = _STRIDE_MIN + gain * (_STRIDE_MAX - _STRIDE_MIN)
        stride = float(max(_TURN_STRIDE, 0.55 * _TURN_STRIDE + 0.45 * fwd_stride))
        if drifting_away:
            stride = min(stride, _TURN_STRIDE)
        out["intent_stride"] = stride
        out["intent_torso_forward"] = 0.54
    else:
        # P-controller on stride: full speed far away, fades to min at stop_distance.
        span = max(dist - stop, 0.05)
        gain = min(1.0, span / max(dist, 1e-6))
        stride = _STRIDE_MIN + gain * (_STRIDE_MAX - _STRIDE_MIN)
        out["intent_stride"] = float(stride)
        out["intent_torso_forward"] = 0.55

    if posture_stability is not None:
        scaled, active = apply_posture_to_navigation(out, float(posture_stability))
        if not active:
            return {}
        out = scaled
        out["task_nav_active"] = 1.0
    else:
        out["task_nav_active"] = 1.0

    return out
