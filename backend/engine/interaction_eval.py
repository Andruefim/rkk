"""
Object interaction eval harness (Sprint 8.1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class InteractionEvalResult:
    passed: bool
    scores: dict[str, float] = field(default_factory=dict)
    gate: str = "interaction_mastery"


class InteractionEval:
    PASS_CRITERIA = {
        "approach_success_rate": 0.7,
        "grasp_success_rate": 0.5,
        "place_success_rate": 0.4,
    }

    def __init__(self) -> None:
        self._approach: list[int] = []
        self._grasp: list[int] = []
        self._place: list[int] = []
        self._last: InteractionEvalResult | None = None

    def record_attempt(self, *, kind: str, success: bool) -> None:
        bucket = {
            "approach": self._approach,
            "grasp": self._grasp,
            "place": self._place,
        }.get(kind)
        if bucket is not None:
            bucket.append(1 if success else 0)
            if len(bucket) > 64:
                del bucket[0]

    def evaluate(self) -> InteractionEvalResult:
        def rate(buf: list[int]) -> float:
            return float(sum(buf)) / max(1, len(buf))

        scores = {
            "approach_success_rate": rate(self._approach),
            "grasp_success_rate": rate(self._grasp),
            "place_success_rate": rate(self._place),
        }
        passed = all(
            scores[k] >= self.PASS_CRITERIA[k]
            for k in self.PASS_CRITERIA
            if len({"approach": self._approach, "grasp": self._grasp, "place": self._place}[k.split("_")[0]]) >= 4
        )
        self._last = InteractionEvalResult(passed=passed, scores=scores)
        return self._last

    def snapshot(self) -> dict[str, Any]:
        r = self._last or self.evaluate()
        return {
            "passed": r.passed,
            "scores": {k: round(float(v), 4) for k, v in r.scores.items()},
            "n_approach": len(self._approach),
            "n_grasp": len(self._grasp),
            "n_place": len(self._place),
        }
