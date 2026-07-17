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

# Upper-body / manipulation intents safe for human_task executive override.
TASK_SAFE_INTENT_FIELDS: frozenset[str] = frozenset(
    {
        "intent_reach_right",
        "intent_reach_left",
        "intent_grasp",
        "intent_wave",
        "intent_look_at",
    }
)

# Locomotion / balance intents — reflex/gait must retain authority during human tasks.
BALANCE_CRITICAL_INTENT_FIELDS: frozenset[str] = frozenset(
    {
        "intent_stride",
        "intent_gait_coupling",
        "intent_torso_forward",
        "intent_support_left",
        "intent_support_right",
        "intent_arm_counterbalance",
        "intent_lean_forward",
        "intent_stop_recover",
    }
)

_BALANCE_CRITICAL_TASK_PRECISION_SCALE = 0.3
_REACH_TORSO_CLAMP_THRESHOLD = 0.6
_REACH_TORSO_CLAMP_MARGIN = 0.08


def is_task_safe_intent_field(field: str) -> bool:
    f = str(field or "")
    if f in TASK_SAFE_INTENT_FIELDS:
        return True
    return f.startswith("intent_head_")


def is_balance_critical_intent_field(field: str) -> bool:
    return str(field or "") in BALANCE_CRITICAL_INTENT_FIELDS


def task_motor_bodysplit_enabled() -> bool:
    return os.environ.get("RKK_TASK_MOTOR_BODYSPLIT", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def task_motor_hold_ticks() -> int:
    try:
        return max(0, int(os.environ.get("RKK_TASK_MOTOR_HOLD_TICKS", "60")))
    except ValueError:
        return 60


def filter_human_task_targets(targets: dict[str, float]) -> dict[str, float]:
    """Keep task-safe intents; omit balance-critical fields for human_task registration."""
    targets = clamp_torso_during_reach(targets)
    if not task_motor_bodysplit_enabled():
        return dict(targets)
    out: dict[str, float] = {}
    for k, v in targets.items():
        ck = str(k)
        if not ck.startswith("intent_"):
            continue
        if is_balance_critical_intent_field(ck):
            continue
        out[ck] = float(v)
    return out


def human_task_field_precision(base_precision: float, field: str) -> float:
    if (
        task_motor_bodysplit_enabled()
        and is_balance_critical_intent_field(field)
    ):
        return float(base_precision) * _BALANCE_CRITICAL_TASK_PRECISION_SCALE
    return float(base_precision)


def clamp_torso_during_reach(targets: dict[str, float]) -> dict[str, float]:
    out = dict(targets)
    reach = max(
        float(out.get("intent_reach_left", 0.5)),
        float(out.get("intent_reach_right", 0.5)),
    )
    if reach <= _REACH_TORSO_CLAMP_THRESHOLD:
        return out
    lo = 0.5 - _REACH_TORSO_CLAMP_MARGIN
    hi = 0.5 + _REACH_TORSO_CLAMP_MARGIN
    for k in ("intent_torso_forward", "intent_lean_forward"):
        if k in out:
            out[k] = float(np.clip(out[k], lo, hi))
    return out

DEFAULT_SOURCE_PRECISION: dict[str, float] = {
    "human_task": 0.90,
    "s2_wm": 0.90,
    "intention_cortex": 0.72,
    "grounded_language": 0.68,
    "navigation": 0.68,
    "manipulation": 0.85,
    "ns_bridge": 0.92,
    "gait": 0.85,
    "hai": 0.58,
    "curriculum": 0.32,
    "genome": 0.42,
    "skill": 0.48,
    "cpg": 0.40,
    "residual": 0.50,
    "reflex": 0.35,
}

# Higher tier wins when human task is active (executive priority ladder).
SOURCE_TIER: dict[str, int] = {
    "human_task": 50,
    "s2_wm": 50,
    "intention_cortex": 40,
    "curriculum": 35,
    "skill": 32,
    "grounded_language": 30,
    "navigation": 30,
    "manipulation": 30,
    "ns_bridge": 28,
    "cpg": 20,
    "gait": 20,
    "hai": 20,
    "residual": 20,
    "genome": 10,
    "reflex": 10,
}


def source_tier(source: str) -> int:
    return int(SOURCE_TIER.get(str(source or ""), 25))


def tier_precision_multiplier(
    tier: int,
    *,
    human_task_active: bool,
    field: str = "",
) -> float:
    if not human_task_active:
        return 1.0
    if field and is_balance_critical_intent_field(field) and tier < 50:
        return 1.0
    if tier >= 50:
        return 3.5
    if tier >= 40:
        return 2.5
    if tier >= 30:
        return 1.4
    if tier >= 20:
        return 0.10
    return 0.06


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
    human_task_active: bool = False,
) -> tuple[dict[str, float], int]:
    """
    Precision-weighted merge per field; hard clamps applied last for LOCOMOTE.
    When human_task_active, tier ladder boosts task/S2/intention over CPG/reflex.
    Returns (merged intents, conflict count).
    """
    current = dict(current or {})
    buckets: dict[str, list[tuple[float, float]]] = {}
    for mi in intents:
        tier = source_tier(mi.source)
        for k, v in mi.as_field_map().items():
            prec = float(np.clip(mi.precision, 0.05, 1.0))
            prec *= tier_precision_multiplier(
                tier, human_task_active=human_task_active, field=k
            )
            if mi.source == "human_task":
                prec *= human_task_field_precision(1.0, k)
            buckets.setdefault(k, []).append((float(v), prec))

    nav_balance: dict[str, float] = {}
    if human_task_active:
        for mi in intents:
            if str(mi.source) != "navigation":
                continue
            for k, v in mi.as_field_map().items():
                if is_balance_critical_intent_field(k):
                    nav_balance[k] = float(v)

    merged: dict[str, float] = {}
    conflicts = 0
    for k in INTENT_FIELDS:
        if human_task_active and k in nav_balance:
            merged[k] = float(nav_balance[k])
            pairs = buckets.get(k, [])
            if pairs:
                vals = [p[0] for p in pairs]
                if max(vals) - min(vals) > 0.12:
                    conflicts += 1
            continue
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
    if locomote and not human_task_active:
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
        self._human_task_active = False

    def set_human_task_active(self, active: bool) -> None:
        self._human_task_active = bool(active)

    def human_task_active(self) -> bool:
        return self._human_task_active

    def should_suppress_substrate(self) -> bool:
        """When human task is active, defer direct locomotion/CPG substrate injections."""
        return self._human_task_active and motor_arbiter_enabled()

    def should_suppress_stabilization(self) -> bool:
        """Recovery/reflex stabilization always runs during human tasks."""
        return False

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

    def early_finalize(
        self,
        sim: Any,
        sources: frozenset[str],
    ) -> dict[str, float]:
        """Merge selected executive sources into graph/motor_state before CPG (same tick)."""
        if not motor_arbiter_enabled() or not sources:
            return {}
        filtered = [mi for mi in self._intents if str(mi.source) in sources]
        if not filtered:
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

        merged, _ = arbitrate(
            filtered,
            macro=macro,
            current=dict(ms),
            human_task_active=self._human_task_active,
        )
        if not merged:
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
        return merged

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

        merged, conflicts = arbitrate(
            self._intents,
            macro=macro,
            current=dict(ms),
            human_task_active=self._human_task_active,
        )
        if not merged:
            self._last_diag = {
                "intents_registered": len(self._intents),
                "coupling_final": round(float(ms.get("intent_gait_coupling", 0.5)), 4),
                "arbiter_conflicts": 0,
                "human_task_active": self._human_task_active,
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
            "human_task_active": self._human_task_active,
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
