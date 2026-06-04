"""
Track H1: symbolic_control stub — boolean rules + action_select (no role_type from B0).
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from engine.genome.spectral import SYMBOLIC_CONTROL_VARIABLE_IDS


def symbolic_control_enabled() -> bool:
    return os.environ.get("RKK_H_SYMBOLIC_ENABLED", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


class EnvironmentSymbolic:
    """Rule-engine style stub for constraint repair and symbolic grounding eval."""

    PRESET = "symbolic_control"

    def __init__(self, device: Any = None, n_rules: int = 4):
        _ = device
        self.preset = self.PRESET
        self.n_rules = max(2, int(n_rules))
        self.variable_ids = list(SYMBOLIC_CONTROL_VARIABLE_IDS)
        self.n_interventions = 0
        self._rules = np.zeros(self.n_rules, dtype=np.float64)
        self._action_select = 0.0
        self._ticks = 0
        self._bailouts = 0
        self._constraint_overrides = 0
        self._satisfied_ticks = 0
        self.reset()

    def reset(self) -> dict[str, float]:
        self._rules[:] = 0.0
        self._action_select = 0.0
        return self.observe()

    def observe(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for i in range(self.n_rules):
            out[f"rule_{i}"] = float(self._rules[i])
        out["action_select"] = float(np.clip(self._action_select, 0.0, 1.0))
        return out

    def _constraints_ok(self) -> bool:
        # rule_0 => rule_1, rule_2 XOR rule_3 style soft constraints
        if self._rules[0] > 0.5 and self._rules[1] < 0.5:
            return False
        if self.n_rules >= 4 and self._rules[2] > 0.5 and self._rules[3] > 0.5:
            return False
        return True

    def intervene(self, variable: str, value: float) -> dict[str, float]:
        if variable not in self.variable_ids:
            return self.observe()
        self.n_interventions += 1
        if variable.startswith("rule_"):
            idx = int(variable.split("_", 1)[1])
            if 0 <= idx < self.n_rules:
                self._rules[idx] = 1.0 if value > 0.5 else 0.0
        elif variable == "action_select":
            self._action_select = float(np.clip(value, 0.0, 1.0))
            if not self._constraints_ok():
                self._bailouts += 1
                self._constraint_repair()

        if self._constraints_ok():
            self._satisfied_ticks += 1

        self._ticks += 1
        return self.observe()

    def _constraint_repair(self) -> None:
        """Emergency override: flip violating rules toward feasible set."""
        self._constraint_overrides += 1
        if self._rules[0] > 0.5:
            self._rules[1] = 1.0
        if self.n_rules >= 4 and self._rules[2] > 0.5 and self._rules[3] > 0.5:
            self._rules[3] = 0.0

    def autonomy_metrics(self) -> dict[str, float]:
        ticks = max(1, self._ticks)
        return {
            "rule_engine_bailout_frac": float(self._bailouts / ticks),
            "constraint_violation_override": float(
                1.0 if self._constraint_overrides > 0 and not self._constraints_ok() else 0.0
            ),
            "constraints_satisfied": float(self._satisfied_ticks / ticks),
        }

    def step_random(self, rng: np.random.Generator | None = None) -> dict[str, float]:
        gen = rng or np.random.default_rng()
        var = gen.choice(self.variable_ids)
        val = float(gen.uniform(0.0, 1.0))
        return self.intervene(var, val)
