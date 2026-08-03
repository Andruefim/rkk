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
    # Pause stride below this — brief post-resolve wobble must not walk.
    return _ef("RKK_TASK_NAV_PAUSE_POSTURE", 0.45)


def task_nav_full_posture() -> float:
    return _ef("RKK_TASK_NAV_FULL_POSTURE", 0.62)


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
    t = (ps - pause) / max(full - pause, 1e-6)
    # Cautious ramp: never jump to full stride from the pause floor.
    return float(0.35 + 0.65 * t)


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


def _sigmoid(x: float) -> float:
    x = float(max(-10.0, min(10.0, x)))
    return float(1.0 / (1.0 + math.exp(-x)))


def _blend_turn_forward(
    *,
    heading_err: float,
    turn_thr: float,
    dist: float,
    stop: float,
    force_turn: bool = False,
) -> dict[str, float]:
    """
    Continuous sigmoidal blend of turn vs forward locomotion.

    Replaces sharp ``if abs(heading_err) > turn_thr`` step discontinuities.
    """
    span = max(dist - stop, 0.05)
    gain = min(1.0, span / max(dist, 1e-6))
    # Near stop, raw gain→0 collapses stride below CPG walk/locomote thresholds
    # (~0.54 / 0.58) and approach plateaus ~0.12m short (live: stuck at 0.67).
    close_gain = _ef("RKK_NAV_CLOSE_GAIN_FLOOR", 0.50)
    if dist > stop:
        gain = max(float(gain), float(max(0.0, min(1.0, close_gain))))
    fwd_stride = _STRIDE_MIN + gain * (_STRIDE_MAX - _STRIDE_MIN)
    close_stride = _ef("RKK_NAV_CLOSE_STRIDE_FLOOR", 0.60)
    if dist > stop:
        fwd_stride = max(float(fwd_stride), float(close_stride))
        fwd_stride = min(float(fwd_stride), float(_STRIDE_MAX))

    abs_h = abs(float(heading_err))
    thr = float(turn_thr)
    w_turn = _sigmoid((abs_h - thr) * 8.0)
    if force_turn:
        w_turn = max(w_turn, 0.55)

    if heading_err > 0.0 or (force_turn and abs(heading_err) < 1e-9):
        target_coupling = _TURN_COUPLING
        target_sup_l, target_sup_r = 0.62, 0.38
    else:
        target_coupling = 1.0 - _TURN_COUPLING
        target_sup_l, target_sup_r = 0.38, 0.62

    turn_stride = max(_TURN_STRIDE, 0.55 * _TURN_STRIDE + 0.45 * fwd_stride)
    if force_turn:
        turn_stride = min(turn_stride, _TURN_STRIDE)

    out: dict[str, float] = {
        "intent_gait_coupling": float((1.0 - w_turn) * 0.5 + w_turn * target_coupling),
        "intent_support_left": float((1.0 - w_turn) * 0.5 + w_turn * target_sup_l),
        "intent_support_right": float((1.0 - w_turn) * 0.5 + w_turn * target_sup_r),
        "intent_stride": float((1.0 - w_turn) * fwd_stride + w_turn * turn_stride),
        "intent_torso_forward": float((1.0 - w_turn) * 0.55 + w_turn * 0.54),
    }

    # Large heading: softly prefer turn stride, but keep legs above walk_gate.
    w_inplace = _sigmoid((abs_h - 1.05) * 8.0)
    turn_floor = float(_TURN_STRIDE)
    out["intent_stride"] = float(
        (1.0 - w_inplace) * out["intent_stride"]
        + w_inplace * max(turn_floor * 0.92, min(out["intent_stride"], turn_floor))
    )
    out["intent_torso_forward"] = float(
        (1.0 - w_inplace) * out["intent_torso_forward"] + w_inplace * 0.53
    )
    # Final close-range floor after turn blending — soft turn weights previously
    # pulled stride back under the CPG locomote threshold near stop.
    if dist > stop and abs_h <= float(turn_thr) * 1.5:
        out["intent_stride"] = float(
            max(float(out["intent_stride"]), float(close_stride))
        )
    return out


def navigation_intents_from_ego_xy(
    x_fwd: float,
    y_right: float,
    stop_distance: float,
    *,
    fallen: bool = False,
    posture_stability: float | None = None,
    bearing_turn_thr: float | None = None,
) -> dict[str, float]:
    """
    Navigate toward egocentric target (x_fwd, y_right) in meters.
    +x = forward, +y = right of the agent.
    """
    from engine.object_working_memory import bearing_range_from_ego

    bearing, range_m = bearing_range_from_ego(float(x_fwd), float(y_right))
    out = navigation_intents_from_bearing_range(
        bearing,
        range_m,
        stop_distance,
        fallen=fallen,
        posture_stability=posture_stability,
        bearing_turn_thr=bearing_turn_thr,
    )
    if out:
        out["task_target_x"] = float(x_fwd)
        out["task_target_y"] = float(y_right)
    return out


def navigation_intents_from_bearing_range(
    bearing: float,
    range_m: float,
    stop_distance: float,
    *,
    fallen: bool = False,
    posture_stability: float | None = None,
    bearing_turn_thr: float | None = None,
) -> dict[str, float]:
    """
    Ego-frame navigation from vision bearing + metric range_m.
    bearing in [-1, 1] (left…right); range_m in meters.
    """
    dist = float(range_m)
    if not math.isfinite(dist) or dist <= 0.05:
        return {}
    stop = max(0.05, float(stop_distance))
    if dist <= stop:
        return {}

    b = float(max(-1.0, min(1.0, bearing)))
    heading_err = b * math.pi * 0.5
    turn_thr = float(bearing_turn_thr) if bearing_turn_thr is not None else _HEADING_TURN_RAD

    out: dict[str, float] = {
        "task_heading_err": float(max(-1.0, min(1.0, heading_err / math.pi))),
        "task_closing_vel": 0.0,
        "vision_bearing": b,
        "vision_range_m": float(dist),
    }
    out.update(
        _blend_turn_forward(
            heading_err=heading_err,
            turn_thr=turn_thr,
            dist=dist,
            stop=stop,
        )
    )

    if fallen:
        # Crawl-mode while recovering: keep closing on the bound target instead
        # of freezing locomotion (empty intents let S2 recovery drift away).
        crawl = {
            "intent_gait_coupling": 0.64,
            "intent_stride": 0.55,
            "intent_torso_forward": 0.56,
            "intent_stop_recover": 0.60,
            "intent_lean_forward": 0.54,
            "task_nav_active": 1.0,
            "task_heading_err": out["task_heading_err"],
            "task_closing_vel": 0.0,
            "vision_bearing": b,
            "vision_range_m": float(dist),
        }
        for k, v in out.items():
            if str(k).startswith("intent_") and k not in crawl:
                crawl[k] = float(v)
        # Soften upright-only intents.
        if "intent_stride" in crawl:
            crawl["intent_stride"] = float(min(0.58, max(0.54, crawl["intent_stride"])))
        return crawl

    if posture_stability is not None:
        scaled, active = apply_posture_to_navigation(out, float(posture_stability))
        if not active:
            # Low posture during approach: keep a crawl floor so we do not stall.
            crawl_ps = max(float(posture_stability), task_nav_pause_posture() + 0.02)
            scaled, active = apply_posture_to_navigation(out, crawl_ps)
            if not active:
                return {}
        out = scaled
        out["task_nav_active"] = 1.0
    else:
        out["task_nav_active"] = 1.0
    return out


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

    out: dict[str, float] = {
        "task_heading_err": float(max(-1.0, min(1.0, heading_err / math.pi))),
        "task_closing_vel": float(max(-1.0, min(1.0, closing * 20.0))),
    }
    turn_thr = _HEADING_TURN_RAD * (0.55 if drifting_away else 1.0)
    # When drifting with near-zero heading, bias left/right from cross sign.
    force_heading = float(heading_err)
    if drifting_away and abs(force_heading) < 1e-6:
        force_heading = 1e-6 if cross >= 0.0 else -1e-6
    out.update(
        _blend_turn_forward(
            heading_err=force_heading,
            turn_thr=turn_thr,
            dist=dist,
            stop=stop,
            force_turn=drifting_away,
        )
    )

    if posture_stability is not None:
        scaled, active = apply_posture_to_navigation(out, float(posture_stability))
        if not active:
            return {}
        out = scaled
        out["task_nav_active"] = 1.0
    else:
        out["task_nav_active"] = 1.0

    return out
