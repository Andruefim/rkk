"""Robot-transferable visual target contract (no privileged registry fields)."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any


def task_resolve_mode() -> str:
    """
    Control-path object resolve mode.

    - vision: camera slots + depth (AGI / robot-transferable)
    - oracle: privileged PyBullet registry (ablation / legacy tests)
    """
    raw = os.environ.get("RKK_TASK_RESOLVE", "oracle").strip().lower()
    if raw in ("vision", "visual", "camera", "slot"):
        return "vision"
    return "oracle"


def vision_resolve_enabled() -> bool:
    return task_resolve_mode() == "vision"


def sim_oracle_bind_enabled() -> bool:
    """
    Non-production sim crutch: bind from privileged oracle XY when vision is
    uncertain (no peaked slot). Explicitly NOT real perception — for parallel
    testing of approach / FSM / escalation after honest 3B kill of ontology
    fallback. Default off; set RKK_SIM_ORACLE_BIND=1 in sim .env.
    """
    raw = os.environ.get("RKK_SIM_ORACLE_BIND", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def vision_active_percept_enabled() -> bool:
    """Look-around retries before giving up / escalating (5A)."""
    raw = os.environ.get("RKK_VISION_ACTIVE_PERCEPT", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def vision_active_percept_max_tries() -> int:
    try:
        return max(1, min(6, int(os.environ.get("RKK_VISION_ACTIVE_PERCEPT_TRIES", "3"))))
    except ValueError:
        return 3


@dataclass
class VisualTarget:
    """Ego-camera target. Must not require body_id / world XY for control."""

    slot_id: str
    u: float
    v: float
    label: str
    confidence: float
    bearing: float
    range_m: float | None = None
    range_var: float | None = None
    range_conf: float | None = None
    bbox: tuple[float, float, float, float] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    # SlotAttention embedding for cosine re-ID across slot permutations.
    latent: list[float] | None = None

    @property
    def ref(self) -> str:
        return f"vision:{self.slot_id}"

    def is_ready(self, *, require_range: bool = True) -> bool:
        if self.confidence < 1e-6:
            return False
        if not (0.0 <= float(self.u) <= 1.0 and 0.0 <= float(self.v) <= 1.0):
            return False
        if require_range:
            r = self.range_m
            if r is None or not (0.05 < float(r) < 50.0):
                return False
            try:
                from engine.vision_depth import depth_max_control_m

                if float(r) >= depth_max_control_m() * 0.98:
                    return False
            except Exception:
                pass
            if self.range_conf is not None and float(self.range_conf) < 0.15:
                return False
        return True

    def with_range(
        self,
        range_m: float | None,
        *,
        range_var: float | None = None,
        range_conf: float | None = None,
    ) -> "VisualTarget":
        return VisualTarget(
            slot_id=self.slot_id,
            u=self.u,
            v=self.v,
            label=self.label,
            confidence=self.confidence,
            bearing=self.bearing,
            range_m=None if range_m is None else float(range_m),
            range_var=range_var,
            range_conf=range_conf,
            bbox=self.bbox,
            diagnostics=dict(self.diagnostics),
            latent=list(self.latent) if self.latent else None,
        )

    def with_uv(self, u: float, v: float, *, bearing: float | None = None) -> "VisualTarget":
        b = float(bearing) if bearing is not None else bearing_from_u(float(u))
        return VisualTarget(
            slot_id=self.slot_id,
            u=float(u),
            v=float(v),
            label=self.label,
            confidence=self.confidence,
            bearing=b,
            range_m=self.range_m,
            range_var=self.range_var,
            range_conf=self.range_conf,
            bbox=self.bbox,
            diagnostics=dict(self.diagnostics),
            latent=list(self.latent) if self.latent else None,
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def bearing_from_u(u: float, *, fov_h_rad: float | None = None) -> float:
    """
    Horizontal bearing from normalized image u in [-1, 1] (left…right).
    Positive = target right of center → turn right (sign convention for nav).
    """
    import math

    uu = max(0.0, min(1.0, float(u)))
    # Normalized offset: -1 left, +1 right
    offset = (uu - 0.5) * 2.0
    if fov_h_rad is None:
        return float(max(-1.0, min(1.0, offset)))
    return float(max(-1.0, min(1.0, offset * (float(fov_h_rad) / math.pi))))


def visual_target_ready(target: VisualTarget | None, *, require_range: bool = True) -> bool:
    return target is not None and target.is_ready(require_range=require_range)
