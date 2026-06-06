"""
Track H1: grid_nav stub — 2D navigation without PyBullet (pos/goal/action_dir).
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from engine.genome.spectral import GRID_NAV_VARIABLE_IDS


def grid_nav_enabled() -> bool:
    return os.environ.get("RKK_H_GRID_NAV_ENABLED", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


class EnvironmentGridNav:
    """Minimal grid world for skeleton transfer and autonomy contract probes."""

    PRESET = "grid_nav"

    def __init__(self, device: Any = None, grid_size: int = 5):
        _ = device
        self.preset = self.PRESET
        self.variable_ids = list(GRID_NAV_VARIABLE_IDS)
        self.grid_size = max(3, int(grid_size))
        self.n_interventions = 0
        self._pos = np.array([0.0, 0.0], dtype=np.float64)
        self._goal = np.array([float(self.grid_size - 1), float(self.grid_size - 1)])
        self._action_dir = 0.0
        self._stuck_steps = 0
        self._pathfinder_overrides = 0
        self._stuck_overrides = 0
        self._ticks = 0
        self._goal_reached_count = 0
        self.reset()

    def reset(self) -> dict[str, float]:
        self._pos[:] = 0.0
        self._goal[:] = float(self.grid_size - 1)
        self._action_dir = 0.0
        self._stuck_steps = 0
        return self.observe()

    def observe(self) -> dict[str, float]:
        gs = float(self.grid_size - 1)
        ticks = max(1, self._ticks)
        return {
            "pos_x": float(np.clip(self._pos[0] / gs, 0.0, 1.0)),
            "pos_y": float(np.clip(self._pos[1] / gs, 0.0, 1.0)),
            "goal_x": float(np.clip(self._goal[0] / gs, 0.0, 1.0)),
            "goal_y": float(np.clip(self._goal[1] / gs, 0.0, 1.0)),
            "action_dir": float(np.clip(self._action_dir / 3.0, 0.0, 1.0)),
            "goal_reached": float(self._goal_reached_count / ticks),
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
        prev_pos = self._pos.copy()
        if variable == "action_dir":
            self._action_dir = float(value)
            step = int(np.clip(round(value * 3.0), 0, 3))
            delta = [(1, 0), (0, 1), (-1, 0), (0, -1)][step % 4]
            self._pos[0] = np.clip(self._pos[0] + delta[0], 0, self.grid_size - 1)
            self._pos[1] = np.clip(self._pos[1] + delta[1], 0, self.grid_size - 1)
        elif variable == "pos_x":
            self._pos[0] = float(value) * (self.grid_size - 1)
        elif variable == "pos_y":
            self._pos[1] = float(value) * (self.grid_size - 1)
        elif variable == "goal_x":
            self._goal[0] = float(value) * (self.grid_size - 1)
        elif variable == "goal_y":
            self._goal[1] = float(value) * (self.grid_size - 1)

        if np.allclose(self._pos, prev_pos):
            self._stuck_steps += 1
        else:
            self._stuck_steps = 0

        if self._stuck_steps >= 4:
            self._pathfinder_override()
            self._stuck_overrides += 1
            self._stuck_steps = 0

        if np.allclose(self._pos, self._goal, atol=0.25):
            self._goal_reached_count += 1
            self._goal = np.array(
                [
                    float(np.random.randint(0, self.grid_size)),
                    float(np.random.randint(0, self.grid_size)),
                ]
            )

        self._ticks += 1
        return self.observe()

    def _pathfinder_override(self) -> None:
        """Stuck recovery: nudge toward goal (pathfinder script override probe)."""
        self._pathfinder_overrides += 1
        for ax in (0, 1):
            if self._pos[ax] < self._goal[ax]:
                self._pos[ax] = min(self._pos[ax] + 1, self.grid_size - 1)
            elif self._pos[ax] > self._goal[ax]:
                self._pos[ax] = max(self._pos[ax] - 1, 0)

    def autonomy_metrics(self) -> dict[str, float]:
        ticks = max(1, self._ticks)
        return {
            "pathfinder_override_frac": float(self._pathfinder_overrides / ticks),
            "stuck_override_active": float(1.0 if self._stuck_steps >= 3 else 0.0),
            "goal_reached": float(self._goal_reached_count / ticks),
        }

    def step_random(self, rng: np.random.Generator | None = None) -> dict[str, float]:
        gen = rng or np.random.default_rng()
        return self.intervene("action_dir", float(gen.uniform(0.0, 1.0)))
