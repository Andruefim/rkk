"""
Layer 4: Symbolic Cognitive Engine — safety axioms, veto, unified predicate checks.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from engine.neuro_symbolic.knowledge_graph import human_proximity_threshold
from engine.neuro_symbolic.predicates import ProbabilisticState, lukasiewicz_implies
from engine.symbolic_verifier import (
    PHYSICS_CONSTRAINTS,
    normalized_to_physical_dict,
    verify_physical_state,
)


def symbolic_engine_enabled() -> bool:
    if not os.environ.get("RKK_SYMBOLIC_ENGINE", "").strip():
        return os.environ.get("RKK_NEURO_SYMBOLIC", "1").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
    return os.environ.get("RKK_SYMBOLIC_ENGINE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


@dataclass
class SafetyAxiom:
    name: str
    description: str
    min_satisfaction: float = 0.99


@dataclass
class VetoResult:
    allowed: bool
    penalty: float = 0.0
    violations: list[str] = field(default_factory=list)
    hard_veto: bool = False


@dataclass
class SymbolicHypothesis:
    predicate: str
    confidence: float
    suggested_macro: str = "IDLE"
    suggested_action: str = ""
    rationale: str = ""


class SymbolicCognitiveEngine:
    """
    System 2 symbolic supervisor: physics constraints + fuzzy safety axioms.
  Hard veto → infinite penalty for planning; soft → downrank score.
    """

    def __init__(self) -> None:
        self._axioms: list[SafetyAxiom] = [
            SafetyAxiom("com_above_floor", "CoM must stay above floor", 0.99),
            SafetyAxiom("joint_limits", "Joint angles within URDF limits", 0.95),
            SafetyAxiom("no_fall_while_locomoting", "No high stride while fallen", 0.90),
            SafetyAxiom(
                "human_proximity",
                "Maintain safe distance from human (distance_to_human)",
                0.99,
            ),
        ]
        self._human_distance_stub: float = 1.0
        self._human_distance_live: float | None = None
        self._last_veto = VetoResult(allowed=True)
        self._veto_count = 0

    def check_physical(
        self,
        state_norm: dict[str, float],
        env: Any,
    ) -> VetoResult:
        s_phys = normalized_to_physical_dict(state_norm, env)
        ok, failed = verify_physical_state(s_phys)
        if ok:
            return VetoResult(allowed=True)
        return VetoResult(
            allowed=False,
            penalty=1e6,
            violations=failed,
            hard_veto=True,
        )

    def check_fuzzy_safety(self, state: ProbabilisticState) -> VetoResult:
        violations: list[str] = []
        fallen = state.best("IsFallen")
        stride_high = state.best("StrideHigh")
        # ∀ fallen → ¬stride_high  (Lukasiewicz)
        safe_locomote = lukasiewicz_implies(fallen, 1.0 - stride_high)
        if safe_locomote < 0.90 and stride_high > 0.55 and fallen > 0.35:
            violations.append("no_fall_while_locomoting")

        standing = state.best("IsStanding")
        if standing < 0.25 and stride_high > 0.6:
            violations.append("stride_while_not_standing")

        if violations:
            self._veto_count += 1
            return VetoResult(
                allowed=False,
                penalty=50.0,
                violations=violations,
                hard_veto=True,
            )
        return VetoResult(allowed=True)

    def set_distance_to_human(self, distance_norm: float, *, live: bool = True) -> None:
        """0=contact, 1=far. Live sensor feed or API injection."""
        v = float(np.clip(distance_norm, 0.0, 1.0))
        self._human_distance_stub = v
        if live:
            self._human_distance_live = v

    def check_human_proximity(
        self,
        obs: dict[str, float] | None = None,
        *,
        fuzzy_state: ProbabilisticState | None = None,
    ) -> VetoResult:
        dist = self._human_distance_live
        if dist is None:
            dist = self._human_distance_stub
        if obs is not None:
            if "distance_to_human" in obs:
                dist = float(obs["distance_to_human"])
            elif "phys_distance_to_human" in obs:
                dist = float(obs["phys_distance_to_human"])
        thr = human_proximity_threshold()
        if dist < thr:
            self._veto_count += 1
            return VetoResult(
                allowed=False,
                penalty=1e6,
                violations=[f"human_proximity:{dist:.3f}<{thr:.3f}"],
                hard_veto=True,
            )
        return VetoResult(allowed=True)

    def veto_prediction(
        self,
        state_norm: dict[str, float],
        env: Any,
        *,
        fuzzy_state: ProbabilisticState | None = None,
    ) -> VetoResult:
        prox = self.check_human_proximity(state_norm, fuzzy_state=fuzzy_state)
        if not prox.allowed:
            self._last_veto = prox
            return prox
        phys = self.check_physical(state_norm, env)
        if not phys.allowed:
            self._last_veto = phys
            return phys
        if fuzzy_state is not None:
            fuzz = self.check_fuzzy_safety(fuzzy_state)
            if not fuzz.allowed:
                self._last_veto = fuzz
                return fuzz
        self._last_veto = VetoResult(allowed=True)
        return self._last_veto

    def generate_hypotheses(self, state: ProbabilisticState) -> list[SymbolicHypothesis]:
        """
        System 2 symbolic reasoning: propose world hypotheses and goal revisions
        (not only veto). Used by S2 controller for deliberation substrate.
        """
        hyps: list[SymbolicHypothesis] = []
        fallen = state.best("IsFallen")
        stable = state.best("IsStable")
        blocked = state.best("PathBlocked")
        goal = state.best("GoalActive")

        if fallen > 0.55:
            hyps.append(
                SymbolicHypothesis(
                    predicate="IsFallen",
                    confidence=fallen,
                    suggested_macro="RECOVER_POSTURE",
                    suggested_action="RecoverPosture",
                    rationale="posture_collapse_requires_recovery",
                )
            )
        if blocked > 0.55 and stable > 0.5:
            hyps.append(
                SymbolicHypothesis(
                    predicate="PathBlocked",
                    confidence=blocked,
                    suggested_macro="LOCOMOTE_DELIVERY",
                    suggested_action="Turn",
                    rationale="obstacle_requires_reorientation",
                )
            )
        if goal > 0.6 and stable > 0.55 and blocked < 0.45:
            hyps.append(
                SymbolicHypothesis(
                    predicate="GoalActive",
                    confidence=goal,
                    suggested_macro="LOCOMOTE_DELIVERY",
                    suggested_action="ApproachTarget",
                    rationale="active_goal_with_clear_path",
                )
            )
        if stable > 0.7 and goal < 0.35:
            hyps.append(
                SymbolicHypothesis(
                    predicate="IsStable",
                    confidence=stable,
                    suggested_macro="EXPLORE",
                    suggested_action="StepForward",
                    rationale="stable_idle_explore_frontier",
                )
            )
        return hyps

    def suggest_goal_revision(
        self,
        state: ProbabilisticState,
        current_macro: str,
    ) -> dict[str, Any] | None:
        """Return revised macro + action if symbolic hypotheses disagree with current plan."""
        hyps = self.generate_hypotheses(state)
        if not hyps:
            return None
        best = max(hyps, key=lambda h: h.confidence)
        cur = str(current_macro or "IDLE").upper()
        if best.suggested_macro == cur:
            return None
        if best.confidence < 0.52:
            return None
        return {
            "macro": best.suggested_macro,
            "action": best.suggested_action,
            "confidence": round(best.confidence, 4),
            "rationale": best.rationale,
        }

    def veto_action(
        self,
        variable: str,
        value: float,
        obs: dict[str, float],
        env: Any,
    ) -> VetoResult:
        """Pre-intervention check for agent do()."""
        if not symbolic_engine_enabled():
            return VetoResult(allowed=True)
        sim_state = dict(obs)
        sim_state[str(variable)] = float(value)
        return self.veto_prediction(sim_state, env)

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": symbolic_engine_enabled(),
            "n_axioms": len(self._axioms),
            "axioms": [a.name for a in self._axioms],
            "n_physics_constraints": len(PHYSICS_CONSTRAINTS),
            "veto_count": self._veto_count,
            "last_veto": {
                "allowed": self._last_veto.allowed,
                "violations": self._last_veto.violations,
                "hard": self._last_veto.hard_veto,
            },
        }
