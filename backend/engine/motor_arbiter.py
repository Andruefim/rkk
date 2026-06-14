"""
Precision-weighted motor intent arbitration across cognitive layers.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

INTENT_FIELDS = (
    "intent_stride",
    "intent_gait_coupling",
    "intent_torso_forward",
    "intent_support_left",
    "intent_support_right",
    "intent_arm_counterbalance",
    "intent_lean_forward",
    "intent_stop_recover",
    "intent_reach_right",
    "intent_reach_left",
    "intent_grasp",
)

DEFAULT_SOURCE_PRECISION: dict[str, float] = {
    "ns_bridge": 0.92,
    "gait": 0.85,
    "hai": 0.58,
    "curriculum": 0.32,
    "genome": 0.42,
    "skill": 0.48,
    "cpg": 0.40,
    "residual": 0.50,
}


def motor_arbiter_enabled() -> bool:
    return os.environ.get("RKK_MOTOR_ARBITER", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _ef(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


@dataclass
class MotorIntent:
    source: str
    precision: float
    stride: float | None = None
    coupling: float | None = None
    support_left: float | None = None
    support_right: float | None = None
    torso_forward: float | None = None
    arm_counterbalance: float | None = None
    lean_forward: float | None = None
    stop_recover: float | None = None
    reach_right: float | None = None
    reach_left: float | None = None
    grasp: float | None = None
    extra: dict[str, float] = field(default_factory=dict)

    def as_field_map(self) -> dict[str, float]:
        out: dict[str, float] = {}
        mapping = {
            "intent_stride": self.stride,
            "intent_gait_coupling": self.coupling,
            "intent_support_left": self.support_left,
            "intent_support_right": self.support_right,
            "intent_torso_forward": self.torso_forward,
            "intent_arm_counterbalance": self.arm_counterbalance,
            "intent_lean_forward": self.lean_forward,
            "intent_stop_recover": self.stop_recover,
            "intent_reach_right": self.reach_right,
            "intent_reach_left": self.reach_left,
            "intent_grasp": self.grasp,
        }
        for k, v in mapping.items():
            if v is not None:
                out[k] = float(v)
        for k, v in self.extra.items():
            if k.startswith("intent_"):
                out[k] = float(v)
        return out

    @classmethod
    def from_dict(
        cls,
        source: str,
        values: dict[str, float],
        *,
        precision: float | None = None,
    ) -> MotorIntent:
        p = precision if precision is not None else DEFAULT_SOURCE_PRECISION.get(source, 0.5)
        extra: dict[str, float] = {}
        kw: dict[str, Any] = {"source": source, "precision": float(p)}
        field_map = {
            "intent_stride": "stride",
            "intent_gait_coupling": "coupling",
            "intent_support_left": "support_left",
            "intent_support_right": "support_right",
            "intent_torso_forward": "torso_forward",
            "intent_arm_counterbalance": "arm_counterbalance",
            "intent_lean_forward": "lean_forward",
            "intent_stop_recover": "stop_recover",
            "intent_reach_right": "reach_right",
            "intent_reach_left": "reach_left",
            "intent_grasp": "grasp",
        }
        for ik, attr in field_map.items():
            if ik in values:
                kw[attr] = float(values[ik])
        for k, v in values.items():
            if k not in field_map:
                extra[k] = float(v)
        if extra:
            kw["extra"] = extra
        return cls(**kw)


def arbitrate(
    intents: list[MotorIntent],
    *,
    macro: str = "",
    current: dict[str, float] | None = None,
) -> tuple[dict[str, float], int]:
    """
    Precision-weighted merge per field; hard clamps applied last for LOCOMOTE.
    Returns (merged intents, conflict count).
    """
    current = dict(current or {})
    buckets: dict[str, list[tuple[float, float]]] = {}
    for mi in intents:
        prec = float(np.clip(mi.precision, 0.05, 1.0))
        for k, v in mi.as_field_map().items():
            buckets.setdefault(k, []).append((float(v), prec))

    merged: dict[str, float] = {}
    conflicts = 0
    for k in INTENT_FIELDS:
        pairs = buckets.get(k, [])
        if not pairs:
            if k in current:
                merged[k] = float(current[k])
            continue
        vals = [p[0] for p in pairs]
        if max(vals) - min(vals) > 0.12:
            conflicts += 1
        wsum = sum(p for _, p in pairs)
        merged[k] = float(sum(v * p for v, p in pairs) / max(wsum, 1e-9))

    macro_u = str(macro or "").strip().upper()
    locomote = macro_u in ("LOCOMOTE_DELIVERY", "EXPLORE")
    if locomote:
        c_max = _ef("RKK_NS_LOCOMOTE_COUPLING", 0.78)
        c_min = _ef("RKK_NS_LOCOMOTE_COUPLING_MIN", 0.72)
        if "intent_gait_coupling" in merged or "intent_gait_coupling" in current:
            merged["intent_gait_coupling"] = float(
                np.clip(merged.get("intent_gait_coupling", current.get("intent_gait_coupling", 0.5)), c_min, c_max)
            )
        stride_floor = _ef("RKK_NS_LOCOMOTE_STRIDE", 0.64)
        if "intent_stride" in merged or "intent_stride" in current:
            merged["intent_stride"] = float(
                np.clip(max(merged.get("intent_stride", 0.5), stride_floor), 0.05, 0.95)
            )
        torso_floor = _ef("RKK_NS_LOCOMOTE_TORSO", 0.58)
        if "intent_torso_forward" in merged or "intent_torso_forward" in current:
            merged["intent_torso_forward"] = float(
                np.clip(max(merged.get("intent_torso_forward", 0.5), torso_floor), 0.05, 0.95)
            )

    for k, v in merged.items():
        merged[k] = float(np.clip(v, 0.05, 0.95))
    return merged, conflicts


def get_support_leg_signal(motor_state: dict) -> str:
    """Intent-based support leg (observability fix for NS/HAI)."""
    sl = float(motor_state.get("intent_support_left", 0.5))
    sr = float(motor_state.get("intent_support_right", 0.5))
    diff = sl - sr
    if diff > 0.15:
        return "left"
    if diff < -0.15:
        return "right"
    return "balanced"


class MotorArbiter:
    def __init__(self) -> None:
        self._intents: list[MotorIntent] = []
        self._last_diag: dict[str, Any] = {}

    def begin_tick(self) -> None:
        self._intents.clear()

    def register(self, intent: MotorIntent | dict[str, Any]) -> None:
        if isinstance(intent, dict):
            src = str(intent.get("source", "unknown"))
            prec = float(intent.get("precision", DEFAULT_SOURCE_PRECISION.get(src, 0.5)))
            vals = {k: float(v) for k, v in intent.items() if k.startswith("intent_")}
            if vals:
                self._intents.append(MotorIntent.from_dict(src, vals, precision=prec))
            return
        self._intents.append(intent)

    def register_from_dict(
        self,
        source: str,
        values: dict[str, float],
        *,
        precision: float | None = None,
    ) -> None:
        if not values:
            return
        self.register(MotorIntent.from_dict(source, values, precision=precision))

    def finalize(self, sim: Any) -> dict[str, float]:
        if not motor_arbiter_enabled():
            return {}
        agent = getattr(sim, "agent", None)
        if agent is None:
            return {}
        base = getattr(agent.env, "base_env", None) or agent.env
        ms = getattr(base, "_motor_state", None)
        if not isinstance(ms, dict):
            return {}

        macro = ""
        s2 = getattr(sim, "_system2_last", None) or {}
        if isinstance(s2, dict):
            macro = str(s2.get("macro") or "")
        if not macro:
            ic = getattr(sim, "_intention_state", None)
            macro = str(getattr(ic, "macro_hint", "") or "")

        merged, conflicts = arbitrate(self._intents, macro=macro, current=dict(ms))
        if not merged:
            self._last_diag = {
                "intents_registered": len(self._intents),
                "coupling_final": round(float(ms.get("intent_gait_coupling", 0.5)), 4),
                "arbiter_conflicts": 0,
            }
            return {}

        for k, v in merged.items():
            ms[k] = v
            if k in agent.graph.nodes:
                agent.graph.nodes[k] = float(v)
            pk = f"phys_{k}"
            if pk in agent.graph.nodes:
                agent.graph.nodes[pk] = float(v)

        motor_state_obj = getattr(sim, "_motor_state", None)
        if motor_state_obj is not None:
            motor_state_obj.intents.update(merged)
            motor_state_obj.support_leg = get_support_leg_signal(ms)

        self._last_diag = {
            "intents_registered": len(self._intents),
            "coupling_final": round(float(merged.get("intent_gait_coupling", ms.get("intent_gait_coupling", 0.5))), 4),
            "stride_final": round(float(merged.get("intent_stride", ms.get("intent_stride", 0.5))), 4),
            "arbiter_conflicts": int(conflicts),
            "sources": [mi.source for mi in self._intents],
        }
        return merged

    def snapshot(self) -> dict[str, Any]:
        return dict(self._last_diag)

    def warn_direct_write(self, source: str, keys: list[str]) -> None:
        if os.environ.get("RKK_MOTOR_ARBITER_STRICT", "0").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            logger.warning("[motor_arbiter] direct write from %s keys=%s", source, keys)
