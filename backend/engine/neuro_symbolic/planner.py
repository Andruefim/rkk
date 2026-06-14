"""
STRIPS-style task planner (PDDL-lite) for humanoid primitives.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from engine.neuro_symbolic.predicates import (
    PATH_BLOCKED_FORWARD_MAX,
    PATH_BLOCKED_TURN_MIN,
    ProbabilisticState,
)


@dataclass
class SymbolicAction:
    name: str
    preconditions: dict[str, float]  # predicate -> min confidence
    add_effects: list[str] = field(default_factory=list)
    remove_effects: list[str] = field(default_factory=list)
    motor_priors: dict[str, float] = field(default_factory=dict)


HUMANOID_ACTIONS: list[SymbolicAction] = [
    SymbolicAction(
        name="RecoverPosture",
        preconditions={"IsFallen": 0.5},
        add_effects=["IsStanding", "IsStable"],
        remove_effects=["IsFallen"],
        motor_priors={
            "intent_stop_recover": 0.68,
            "intent_torso_forward": 0.62,
            "intent_stride": 0.46,
        },
    ),
    SymbolicAction(
        name="HoldStance",
        preconditions={"IsStanding": 0.55},
        add_effects=["IsStable"],
        motor_priors={
            "intent_support_left": 0.55,
            "intent_support_right": 0.55,
            "intent_stride": 0.50,
        },
    ),
    SymbolicAction(
        name="Turn",
        preconditions={"PathBlocked": PATH_BLOCKED_TURN_MIN, "IsStable": 0.45},
        remove_effects=["PathBlocked"],
        motor_priors={
            "intent_stride": 0.48,
            "intent_look_at": 0.72,
            "intent_gait_coupling": 0.72,
            "intent_arm_counterbalance": 0.54,
        },
    ),
    SymbolicAction(
        name="StepForward",
        preconditions={
            "IsStable": 0.55,
            "BothFeetContact": 0.45,
            "NOT PathBlocked": 0.65,
        },
        add_effects=["StrideHigh"],
        motor_priors={
            "intent_stride": 0.64,
            "intent_torso_forward": 0.58,
            "intent_gait_coupling": 0.78,
            "intent_arm_counterbalance": 0.56,
        },
    ),
    SymbolicAction(
        name="ApproachTarget",
        preconditions={
            "IsStable": 0.5,
            "GoalActive": 0.6,
            "NOT PathBlocked": 0.65,
        },
        add_effects=["TargetNear"],
        motor_priors={
            "intent_stride": 0.68,
            "intent_torso_forward": 0.62,
            "self_goal_active": 0.88,
        },
    ),
]


def _satisfies(state: ProbabilisticState, pre: dict[str, float]) -> bool:
    for pred, thr in pre.items():
        if pred.startswith("NOT "):
            blocked_pred = pred[4:].strip()
            if state.best(blocked_pred) > (1.0 - thr):
                return False
        elif state.best(pred) < thr:
            return False
    return True


def _apply_effects(state: ProbabilisticState, action: SymbolicAction) -> ProbabilisticState:
    new = ProbabilisticState(facts=list(state.facts))
    for pred in action.remove_effects:
        new.facts = [f for f in new.facts if f.predicate != pred]
    for pred in action.add_effects:
        new.add(pred, 0.85)
    return new


def plan_to_goal(
    state: ProbabilisticState,
    goal_predicates: dict[str, float],
    *,
    actions: list[SymbolicAction] | None = None,
    max_depth: int = 4,
) -> list[SymbolicAction]:
    """Greedy BFS over symbolic actions toward goal facts."""
    actions = actions or HUMANOID_ACTIONS
    if state.best("PathBlocked") > PATH_BLOCKED_TURN_MIN:
        max_depth = max(max_depth, 5)

    def goal_met(s: ProbabilisticState) -> bool:
        return all(s.best(p) >= thr for p, thr in goal_predicates.items())

    if goal_met(state):
        return []

    frontier: list[tuple[ProbabilisticState, list[SymbolicAction]]] = [(state, [])]
    seen: set[str] = set()

    for _ in range(max_depth):
        next_frontier: list[tuple[ProbabilisticState, list[SymbolicAction]]] = []
        for st, path in frontier:
            key = "|".join(sorted(st.to_dict().keys()))
            if key in seen:
                continue
            seen.add(key)
            for act in actions:
                if not _satisfies(st, act.preconditions):
                    continue
                st2 = _apply_effects(st, act)
                path2 = path + [act]
                if goal_met(st2):
                    return path2
                next_frontier.append((st2, path2))
        frontier = next_frontier
        if not frontier:
            break
    return []


def macro_to_goal(macro: str) -> dict[str, float]:
    m = str(macro or "").strip().upper()
    if m == "RECOVER_POSTURE":
        return {"IsStanding": 0.7, "IsStable": 0.6}
    if m == "LOCOMOTE_DELIVERY":
        return {"StrideHigh": 0.55, "TargetNear": 0.45}
    if m == "EXPLORE":
        return {"SlotActive": 0.4}
    return {"IsStable": 0.6}
