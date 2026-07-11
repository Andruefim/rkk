"""Physical manipulation verification — displacement only, no intent shortcuts."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any

from engine.object_resolver import ResolvedObject


def default_min_displacement_m() -> float:
    try:
        return float(os.environ.get("RKK_MANIP_MIN_DISP", "0.12"))
    except ValueError:
        return 0.12


@dataclass
class ManipulationEpisode:
    baseline_xy: tuple[float, float]
    target_ref: str
    target_body_id: int | None = None
    min_displacement_m: float = field(default_factory=default_min_displacement_m)
    requested_direction: tuple[float, float] | None = None

    @classmethod
    def begin(
        cls,
        resolved: ResolvedObject,
        *,
        baseline_xy: tuple[float, float] | None = None,
        requested_direction: tuple[float, float] | None = None,
        min_displacement_m: float | None = None,
    ) -> ManipulationEpisode:
        base = baseline_xy if baseline_xy is not None else (
            float(resolved.position[0]), float(resolved.position[1])
        )
        return cls(
            baseline_xy=(float(base[0]), float(base[1])),
            target_ref=str(resolved.ref),
            target_body_id=resolved.body_id,
            min_displacement_m=(
                float(min_displacement_m)
                if min_displacement_m is not None
                else default_min_displacement_m()
            ),
            requested_direction=requested_direction,
        )


def _normalize_xy(v: tuple[float, float]) -> tuple[float, float]:
    x, y = float(v[0]), float(v[1])
    n = math.hypot(x, y)
    if n < 1e-9:
        return 1.0, 0.0
    return x / n, y / n


def verify_manipulation(
    episode: ManipulationEpisode,
    current_xy: tuple[float, float],
    *,
    intent_signals: dict[str, float] | None = None,
    pe_success: bool | None = None,
) -> dict[str, Any]:
    """
    Verify manipulation success from target body XY displacement only.

    ``intent_signals`` and ``pe_success`` are recorded in diagnostics but never
    grant success.
    """
    cx, cy = float(current_xy[0]), float(current_xy[1])
    bx, by = float(episode.baseline_xy[0]), float(episode.baseline_xy[1])
    dx, dy = cx - bx, cy - by
    displacement = math.hypot(dx, dy)
    min_disp = float(episode.min_displacement_m)

    forward_proj: float | None = None
    direction_ok = True
    if episode.requested_direction is not None:
        fx, fy = _normalize_xy(episode.requested_direction)
        forward_proj = float(dx * fx + dy * fy)
        direction_ok = forward_proj > 0.02

    moved_enough = displacement >= min_disp
    success = bool(moved_enough and direction_ok)

    intent_high = False
    if intent_signals:
        grasp = float(intent_signals.get("intent_grasp", 0.0))
        reach = max(
            float(intent_signals.get("intent_reach_left", 0.0)),
            float(intent_signals.get("intent_reach_right", 0.0)),
        )
        intent_high = grasp > 0.7 or reach > 0.7

    return {
        "success": success,
        "target_ref": episode.target_ref,
        "target_body_id": episode.target_body_id,
        "baseline_xy": [bx, by],
        "current_xy": [cx, cy],
        "displacement_m": round(displacement, 5),
        "min_displacement_m": round(min_disp, 5),
        "moved_enough": moved_enough,
        "direction_ok": direction_ok,
        "forward_projection_m": (
            round(forward_proj, 5) if forward_proj is not None else None
        ),
        "requested_direction": (
            [float(episode.requested_direction[0]), float(episode.requested_direction[1])]
            if episode.requested_direction is not None
            else None
        ),
        "intent_high": intent_high,
        "pe_success_flag": bool(pe_success) if pe_success is not None else None,
        "intent_could_not_succeed": bool(intent_high and not success),
        "reason": "displacement_ok" if success else (
            "insufficient_displacement" if not moved_enough else "wrong_direction"
        ),
    }
