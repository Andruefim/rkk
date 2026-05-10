"""
Phase B: modality precision scalars, weighted PE aggregation, calibration protocol placeholders.

Flags:
  RKK_PRECISION_GROUPS=1       — weighted Σ π_g · PE_g² in aggregate_pe_weighted()
  RKK_PRECISION_CALIB_WINDOW=50 — rolling window for calibration acceptance / noise injection logs
  RKK_PRECISION_CALIB_*        — hooks for offline Gaussian noise sweeps (documented placeholders)

Phase C₂ (full temporal routing, after B + E in the plan): set **both**
  RKK_PRECISION_GROUPS=1
  RKK_TEMPORAL_PRECISION_ROUTING=1
so ``hierarchical_active_inference`` down-weights forward/planning PE when vision π is low
(see routing_weights_for_hai()).

Online adaptation hook: precision_decay_visual(multiplier) lowers π_vision after injected anomalies.
"""
from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field

import numpy as np


def precision_groups_enabled() -> bool:
    return os.environ.get("RKK_PRECISION_GROUPS", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def precision_calib_window() -> int:
    try:
        return max(8, min(4096, int(os.environ.get("RKK_PRECISION_CALIB_WINDOW", "50"))))
    except ValueError:
        return 50


def temporal_precision_routing_enabled() -> bool:
    return precision_groups_enabled() and os.environ.get(
        "RKK_TEMPORAL_PRECISION_ROUTING", "0"
    ).strip().lower() in ("1", "true", "yes", "on")


def modality_group_for_var(name: str) -> str:
    """
    Single canonical routing key for precision / efference / HAI (Phase B, C₂).

    Must match how ``weighted_squared_error_sum`` and ``default_precision_vector`` index weights.
    """
    s0 = str(name).strip()
    s = s0.lower()
    if s in ("vision", "proprio", "motor_intent", "sandbox", "vestibular", "other"):
        return s
    if s.startswith("slot_") or "vision" in s or s.startswith("pixel"):
        return "vision"
    if s.startswith("vestibular") or "floor_friction" in s:
        return "vestibular"
    if s.startswith("intent_") or s.startswith("phys_intent"):
        return "motor_intent"
    if any(
        s.startswith(p)
        for p in (
            "cube",
            "ball_",
            "lever_",
            "target_dist",
            "stack_",
            "stability_score",
        )
    ):
        return "sandbox"
    if s.startswith("proprio") or any(
        x in s
        for x in (
            "com_",
            "torso_",
            "foot_",
            "posture",
            "joint",
            "hip",
            "knee",
            "ankle",
        )
    ):
        return "proprio"
    if any(
        s.startswith(p)
        for p in (
            "spine_",
            "neck_",
            "lhip",
            "rhip",
            "lknee",
            "rknee",
            "lankle",
            "rankle",
            "lshoulder",
            "rshoulder",
            "lelbow",
            "relbow",
            "gait_",
            "support_",
            "motor_drive",
        )
    ):
        return "proprio"
    return "other"


@dataclass
class PrecisionGroupState:
    """Scalar precision π_g > 0 per modality (higher = trust channel more)."""

    proprio: float = 1.0
    vision: float = 1.0
    vestibular: float = 1.0
    motor_intent: float = 1.0
    sandbox: float = 1.0
    other: float = 1.0
    calib_steps: int = 0
    noise_events: deque[tuple[float, str]] = field(
        default_factory=lambda: deque(maxlen=precision_calib_window())
    )

    def copy(self) -> "PrecisionGroupState":
        return PrecisionGroupState(
            proprio=self.proprio,
            vision=self.vision,
            vestibular=self.vestibular,
            motor_intent=self.motor_intent,
            sandbox=self.sandbox,
            other=self.other,
            calib_steps=self.calib_steps,
            noise_events=deque(self.noise_events, maxlen=self.noise_events.maxlen or 50),
        )

    def weight_for_group(self, group: str) -> float:
        g = (group or "other").lower()
        v = getattr(self, g, None)
        if isinstance(v, (int, float)):
            return float(max(1e-6, float(v)))
        return float(max(1e-6, self.other))

    def record_calibration_placeholder(self, sigma: float, channel: str) -> None:
        """Offline calibration artifact hook — stores (σ, channel) for reproducibility."""
        self.calib_steps += 1
        self.noise_events.append((float(sigma), str(channel)[:40]))

    def decay_vision(self, factor: float = 0.5) -> None:
        """Lower vision precision after detected anomaly (upper × bounded)."""
        self.vision = float(max(0.05, min(4.0, self.vision * float(factor))))


_GLOBAL_PRECISION = PrecisionGroupState()


def get_precision_state() -> PrecisionGroupState:
    return _GLOBAL_PRECISION


def aggregate_pe_weighted(pe_by_group: dict[str, float]) -> float:
    """
    Σ_g π_g · PE_g² when RKK_PRECISION_GROUPS=1; else unweighted sum of squares.
    """
    st = get_precision_state()
    if not precision_groups_enabled():
        return float(sum(float(v) ** 2 for v in pe_by_group.values()))
    total = 0.0
    for raw_key, pe in pe_by_group.items():
        mg = modality_group_for_var(str(raw_key))
        w = st.weight_for_group(mg)
        total += w * float(pe) ** 2
    return float(total)


def routing_weights_for_hai() -> tuple[float, float]:
    """
    Returns (w_planning, w_proprio) multipliers for hierarchical PE when temporal routing is on.
    High vision uncertainty → down-weight forward/planning PE vs vertical proprio channels.
    """
    if not temporal_precision_routing_enabled():
        return 1.0, 1.0
    st = get_precision_state()
    pv = st.weight_for_group("vision")
    pp = st.weight_for_group("proprio")
    # If vision π is low, scale planning channel down; proprio slightly up (bounded)
    w_plan = float(np.clip(pv / max(pp, 1e-6), 0.35, 1.0))
    w_prop = float(np.clip(1.15 - 0.35 * w_plan, 0.85, 1.25))
    return w_plan, w_prop
