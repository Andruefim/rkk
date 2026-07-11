"""Grounded task goal contract.

A human command is grounded into a TaskGoal: a small set of predicates over
*observable* world state. No verb taxonomies — predicate selection is done by
embedding similarity against predicate descriptions plus world-model
imagination, and success is measured directly in observations.

This module is intentionally dependency-free (dataclasses only) so that
language grounding, planning, execution and verification can all share it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Predicate kinds (open set; consumers must ignore kinds they don't know):
#   "reduce_distance"  agent-to-target distance -> target_value (meters)
#   "contact"          physical contact with target_ref (target_value >= 0.5)
#   "displace"         target object XY displacement >= target_value (meters)
#   "state_key"        observation `key` should reach target_value
@dataclass
class GoalPredicate:
    kind: str
    target_ref: str | None = None
    key: str | None = None
    target_value: float = 0.0
    tolerance: float = 0.1
    weight: float = 1.0

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "target_ref": self.target_ref,
            "key": self.key,
            "target_value": float(self.target_value),
            "tolerance": float(self.tolerance),
            "weight": float(self.weight),
        }


@dataclass
class TaskGoal:
    text: str
    target_ref: str | None = None
    predicates: list[GoalPredicate] = field(default_factory=list)
    # Confidence of the language->goal grounding in [0, 1].
    confidence: float = 0.0
    # True when world-model imagination for this goal is trusted enough to be
    # used for PE-based verification; otherwise verify predicates directly.
    wm_trusted: bool = False
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "target_ref": self.target_ref,
            "predicates": [p.to_dict() for p in self.predicates],
            "confidence": float(self.confidence),
            "wm_trusted": bool(self.wm_trusted),
            "diagnostics": dict(self.diagnostics),
        }
