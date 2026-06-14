"""
Fuzzy predicates over embodied state — vector-to-symbol grounding (Layer 3 ascent).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Shared thresholds — planner NOT-gate, Turn, and fallback must agree.
PATH_BLOCKED_FORWARD_MAX = 0.35  # StepForward forbidden above this
PATH_BLOCKED_TURN_MIN = 0.35  # Turn / replan when above this
PATH_BLOCKED_ARM_RAW = 0.65  # latch blocked when raw exceeds this
PATH_BLOCKED_CLEAR_RAW = 0.35  # release blocked when raw falls below this
ACTUAL_DELTA_STAGNANT = 0.01
VISUAL_DEPTH_BLOCKED = 0.3
CONTACT_STRESS_BLOCKED = 0.72


@dataclass
class PathBlockedHysteresis:
    """Sticky PathBlocked — arm >0.65, clear <0.35, hold in between."""

    _blocked: bool = False
    _level: float = 0.0

    def update(self, raw: float) -> float:
        raw = float(np.clip(raw, 0.0, 1.0))
        if self._blocked:
            if raw < PATH_BLOCKED_CLEAR_RAW:
                self._blocked = False
                self._level = raw
            else:
                self._level = max(self._level, raw, PATH_BLOCKED_TURN_MIN + 0.02)
        elif raw > PATH_BLOCKED_ARM_RAW:
            self._blocked = True
            self._level = raw
        else:
            self._level = min(raw, PATH_BLOCKED_FORWARD_MAX)
        return float(np.clip(self._level, 0.0, 1.0))

    def reset(self) -> None:
        self._blocked = False
        self._level = 0.0


def lukasiewicz_and(a: float, b: float) -> float:
    return float(max(0.0, a + b - 1.0))


def lukasiewicz_or(a: float, b: float) -> float:
    return float(min(1.0, a + b))


def lukasiewicz_implies(a: float, b: float) -> float:
    return float(min(1.0, 1.0 - a + b))


@dataclass
class Fact:
    predicate: str
    args: tuple[str, ...] = ()
    confidence: float = 1.0

    def key(self) -> str:
        if not self.args:
            return self.predicate
        return f"{self.predicate}({','.join(self.args)})"


@dataclass
class ProbabilisticState:
    facts: list[Fact] = field(default_factory=list)

    def add(self, predicate: str, confidence: float, *args: str) -> None:
        self.facts.append(
            Fact(
                predicate=predicate,
                args=args,
                confidence=float(np.clip(confidence, 0.0, 1.0)),
            )
        )

    def best(self, predicate: str) -> float:
        vals = [f.confidence for f in self.facts if f.predicate == predicate]
        return max(vals) if vals else 0.0

    def to_dict(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for f in self.facts:
            out[f.key()] = round(f.confidence, 4)
        return out


def _step_forward_skill_active(context: dict[str, Any] | None) -> bool:
    skill = str((context or {}).get("active_skill") or "")
    return skill.startswith("step_forward")


def _ignore_pe_for_path_blocked(context: dict[str, Any] | None) -> bool:
    """Turn / stance — zero forward delta is expected, not an obstacle."""
    ctx = context or {}
    skill = str(ctx.get("active_skill") or "")
    if skill in ("hold_stance", "stabilize_stance", "stand_up") or skill.startswith("stand"):
        return True
    plan_head = str(ctx.get("ns_plan_head") or ctx.get("symbolic_plan_head") or "")
    if plan_head == "Turn":
        return True
    return False


def _obstacle_physically_confirmed(
    merged: dict[str, float],
    context: dict[str, Any] | None,
) -> bool:
    ctx = context or {}
    depth = float(
        merged.get(
            "visual_depth",
            ctx.get("visual_depth", merged.get("phys_visual_depth", 1.0)),
        )
    )
    if depth < VISUAL_DEPTH_BLOCKED:
        return True
    stress = float(
        ctx.get(
            "contact_stress",
            merged.get("intero_stress", merged.get("phys_intero_stress", 0.0)),
        )
    )
    return stress >= CONTACT_STRESS_BLOCKED


def _pe_frustration_to_blocked(pe: float) -> float:
    """Steep soft-threshold: chronic forward PE jumps above dead-zone quickly."""
    if pe >= -0.35:
        return 0.0
    # Logistic centered ~−0.72; pe≈−0.75 → ~0.65, pe≈−0.91 → ~0.99
    z = -25.0 * (pe + 0.72)
    return float(np.clip(1.0 / (1.0 + np.exp(-z)), 0.0, 1.0))


def compute_path_blocked_raw(
    merged: dict[str, float],
    context: dict[str, Any] | None = None,
) -> float:
    """
    Context-aware raw PathBlocked before hysteresis.
    PE → obstacle only under step_forward_* with stagnant actual_delta AND physical cue.
    """
    ctx = context or {}
    if _ignore_pe_for_path_blocked(ctx):
        pe_blocked = 0.0
    else:
        actual_delta = abs(
            float(ctx.get("actual_delta", merged.get("actual_delta", 1.0)))
        )
        stagnant = actual_delta < ACTUAL_DELTA_STAGNANT
        pe_blocked = 0.0
        if _step_forward_skill_active(ctx) and stagnant:
            if _obstacle_physically_confirmed(merged, ctx):
                pe = float(
                    merged.get(
                        "hai_pe_fwd_ema",
                        ctx.get("hai_pe_fwd_ema", merged.get("pe_fwd_ema", 0.0)),
                    )
                )
                pe_blocked = _pe_frustration_to_blocked(pe)
                if pe < -0.75:
                    pe_blocked = max(pe_blocked, 0.65)

    depth = float(
        merged.get(
            "visual_depth",
            ctx.get("visual_depth", merged.get("phys_visual_depth", 1.0)),
        )
    )
    actual_delta = abs(float(ctx.get("actual_delta", merged.get("actual_delta", 1.0))))
    vis_blocked = 0.0
    if depth < VISUAL_DEPTH_BLOCKED and actual_delta < ACTUAL_DELTA_STAGNANT:
        vis_blocked = float(np.clip((VISUAL_DEPTH_BLOCKED - depth) / VISUAL_DEPTH_BLOCKED, 0.0, 1.0))

    return float(max(pe_blocked, vis_blocked))


def compute_path_blocked_confidence(
    merged: dict[str, float],
    context: dict[str, Any] | None = None,
    *,
    hysteresis: PathBlockedHysteresis | None = None,
) -> float:
    raw = compute_path_blocked_raw(merged, context)
    if hysteresis is not None:
        return hysteresis.update(raw)
    return raw


def path_forward_blocked(confidence: float) -> bool:
    return float(confidence) > PATH_BLOCKED_FORWARD_MAX


def path_turn_recommended(confidence: float) -> bool:
    return float(confidence) > PATH_BLOCKED_TURN_MIN


def _f(obs: dict[str, float], *keys: str, default: float = 0.5) -> float:
    for k in keys:
        if k in obs:
            try:
                return float(obs[k])
            except (TypeError, ValueError):
                continue
    return default


def ground_humanoid_state(
    obs: dict[str, float],
    graph_nodes: dict[str, float] | None = None,
    *,
    context: dict[str, Any] | None = None,
) -> ProbabilisticState:
    """Perception: continuous obs + graph → fuzzy facts."""
    nodes = graph_nodes or {}
    merged = {**nodes, **obs}
    ps = float(_f(merged, "posture_stability", "phys_posture_stability"))
    cz = float(_f(merged, "com_z", "phys_com_z"))
    stride = float(_f(merged, "intent_stride", "phys_intent_stride"))
    td = float(_f(merged, "target_dist", "phys_target_dist", default=0.5))
    foot_l = float(_f(merged, "foot_contact_l", "phys_foot_contact_l"))
    foot_r = float(_f(merged, "foot_contact_r", "phys_foot_contact_r"))
    fallen = cz < 0.28 or ps < 0.22

    st = ProbabilisticState()
    st.add("IsStanding", 1.0 - float(np.clip((0.28 - cz) / 0.2, 0, 1)) if cz < 0.28 else ps)
    st.add("IsFallen", float(fallen))
    st.add("IsStable", float(np.clip((ps - 0.45) / 0.4, 0, 1)))
    st.add("BothFeetContact", float(np.clip(min(foot_l, foot_r) * 1.2, 0, 1)))
    st.add("StrideHigh", float(np.clip((stride - 0.52) / 0.28, 0, 1)))
    st.add("TargetNear", float(np.clip(1.0 - td / 0.55, 0, 1)))
    st.add("GoalActive", float(np.clip(_f(merged, "self_goal_active"), 0, 1)))
    dist_h = _f(merged, "distance_to_human", "phys_distance_to_human", default=1.0)
    st.add("HumanNear", float(np.clip(1.0 - dist_h, 0, 1)))
    hyst = (context or {}).get("_path_blocked_hysteresis")
    st.add(
        "PathBlocked",
        compute_path_blocked_confidence(merged, context, hysteresis=hyst),
    )

    for k, v in merged.items():
        sk = str(k)
        if sk.startswith("slot_"):
            act = abs(float(v) - 0.5) * 2.0
            if act > 0.15:
                st.add("SlotActive", float(np.clip(act, 0, 1)), sk)
        if sk.startswith("concept_"):
            if float(v) > 0.55:
                st.add("ConceptActive", float(v), sk)

    return st


def embodied_var_id(var_id: str) -> bool:
    """Filter non-motor abstract vars from goal generation."""
    if not var_id:
        return False
    if var_id.startswith("concept_") or var_id.startswith("slot_"):
        return False
    if var_id in ("target_dist", "posture_stability", "com_z", "posture"):
        return True
    if var_id.startswith("intent_") or var_id.startswith("phys_intent_"):
        return True
    if var_id.startswith("l1_") or var_id.startswith("gait_phase"):
        return False
    if var_id in ("gait_phase_l", "gait_phase_r"):
        return False
    return var_id.startswith("intent")
