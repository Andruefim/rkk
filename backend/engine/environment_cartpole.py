"""
Track B4: cartpole stub for cross-topology spectral / skeleton transfer eval.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from engine.genome.spectral import CARTPOLE_VARIABLE_IDS


class EnvironmentCartpole:
    """Minimal cartpole balance stub (no PyBullet)."""

    PRESET = "cartpole"

    def __init__(self, device: Any = None):
        _ = device
        self.preset = self.PRESET
        self.variable_ids = list(CARTPOLE_VARIABLE_IDS)
        self.n_interventions = 0
        self._cart_pos = 0.0
        self._cart_vel = 0.0
        self._pole_angle = 0.05
        self._pole_vel = 0.0
        self._pole_angular_vel = 0.0
        self._action_force = 0.0
        self._replan_overrides = 0
        self._balance_emergency = 0
        self._ticks = 0
        self.reset()

    def reset(self) -> dict[str, float]:
        self._cart_pos = 0.0
        self._cart_vel = 0.0
        self._pole_angle = float(np.random.uniform(-0.08, 0.08))
        self._pole_vel = 0.0
        self._pole_angular_vel = 0.0
        self._action_force = 0.0
        return self.observe()

    def observe(self) -> dict[str, float]:
        upright = float(np.clip(1.0 - abs(self._pole_angle) / np.pi, 0.0, 1.0))
        balance = float(np.clip(upright - abs(self._cart_vel) * 0.2, 0.0, 1.0))
        return {
            "cart_pos": float(np.clip(0.5 + self._cart_pos * 0.1, 0.0, 1.0)),
            "cart_vel": float(np.clip(0.5 + self._cart_vel * 0.05, 0.0, 1.0)),
            "pole_angle": float(np.clip(0.5 + self._pole_angle / np.pi, 0.0, 1.0)),
            "pole_vel": float(np.clip(0.5 + self._pole_vel * 0.05, 0.0, 1.0)),
            "pole_angular_vel": float(np.clip(0.5 + self._pole_angular_vel * 0.05, 0.0, 1.0)),
            "action_force": float(np.clip(0.5 + self._action_force * 0.1, 0.0, 1.0)),
            "upright": upright,
            "balance_stability": balance,
        }

    @property
    def variables(self) -> dict[str, float]:
        return self.observe()

    def gt_edges(self) -> list[dict]:
        return []

    def intervene(self, variable: str, value: float) -> dict[str, float]:
        if variable not in self.variable_ids:
            return self.observe()
        self.n_interventions += 1
        if variable == "action_force":
            self._action_force = float(value) * 2.0 - 1.0
        elif variable == "cart_pos":
            self._cart_pos = float(value) * 10.0 - 5.0
        elif variable == "pole_angle":
            self._pole_angle = (float(value) - 0.5) * np.pi

        # Simple dynamics
        self._pole_angular_vel += self._action_force * 0.08 - self._pole_angle * 0.04
        self._pole_vel += self._pole_angular_vel * 0.02
        self._pole_angle += self._pole_vel
        self._cart_vel += self._action_force * 0.03
        self._cart_pos += self._cart_vel * 0.02

        obs = self.observe()
        if obs["balance_stability"] < 0.35:
            self._balance_emergency += 1
            self._replan_overrides += 1
            self._pole_angle *= 0.5
            self._pole_vel *= 0.3
            obs = self.observe()

        self._ticks += 1
        return obs

    def is_fallen(self) -> bool:
        return abs(self._pole_angle) > 0.45

    def autonomy_metrics(self) -> dict[str, float]:
        ticks = max(1, self._ticks)
        return {
            "replan_script_override_frac": float(self._replan_overrides / ticks),
            "balance_emergency_override": float(self._balance_emergency / ticks),
            "upright": float(self.observe().get("upright", 0.0)),
        }

    def step_random(self, rng: np.random.Generator | None = None) -> dict[str, float]:
        gen = rng or np.random.default_rng()
        return self.intervene("action_force", float(gen.uniform(0.0, 1.0)))
