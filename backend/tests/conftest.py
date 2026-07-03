"""
Pytest отключён по умолчанию, чтобы не мешать ручному прогону run.py / UI.

  RKK_RUN_TESTS=1 pytest
  RKK_RUN_TESTS=1 pytest tests/test_graph_perf.py -q
"""
from __future__ import annotations

import os

import pytest


def _tests_enabled() -> bool:
    return os.environ.get("RKK_RUN_TESTS", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def pytest_collection_modifyitems(config, items) -> None:
    if _tests_enabled():
        return
    skip = pytest.mark.skip(
        reason="pytest off (set RKK_RUN_TESTS=1 to run backend tests)"
    )
    for item in items:
        item.add_marker(skip)


# --- AGI human-command loop integration stubs (no PyBullet) ---

import numpy as np
import torch

from engine.causal_graph import CausalGraph
from engine.features.simulation.mixin_grounded_language import SimulationGroundedLanguageMixin
from engine.intention_cortex import IntentionCortex
from engine.motor_arbiter import MotorArbiter
from engine.system2.controller import System2Controller


class AgiLoopEnv:
    """Minimal env.observe() source for task-binding integration tests."""

    def __init__(self, obs: dict[str, float] | None = None) -> None:
        self._obs = dict(obs or _default_humanoid_obs())
        self._motor_state = {k: float(v) for k, v in self._obs.items() if k.startswith("intent_")}
        self.preset = "humanoid"

    def observe(self) -> dict[str, float]:
        return dict(self._obs)

    def apply_motor_intent_residuals(self, residuals: dict[str, float]) -> None:
        for k, dv in residuals.items():
            self._motor_state[k] = float(np.clip(self._motor_state.get(k, 0.5) + dv, 0.05, 0.95))
            if k in self._obs:
                self._obs[k] = self._motor_state[k]


class AgiLoopAgent:
    def __init__(self, obs: dict[str, float] | None = None) -> None:
        self.env = AgiLoopEnv(obs)
        self.graph = CausalGraph(device=torch.device("cpu"))
        ids = list(self.env._obs.keys()) + ["self_goal_active", "sensory_audio_semantic_0"]
        self.graph.rebind_variables(ids, {k: float(self.env._obs.get(k, 0.5)) for k in ids})


def _default_humanoid_obs() -> dict[str, float]:
    obs: dict[str, float] = {
        "target_dist": 0.55,
        "posture_stability": 0.72,
        "com_z": 0.52,
        "intent_stride": 0.42,
        "intent_torso_forward": 0.48,
        "intero_energy": 0.85,
        "intero_stress": 0.12,
        "self_goal_active": 0.0,
    }
    for i in range(6):
        obs[f"slot_{i}"] = 0.5
    return obs


class AgiLoopSim(SimulationGroundedLanguageMixin):
    """Mock sim: grounded language + task binding without PyBullet."""

    def __init__(self, obs: dict[str, float] | None = None, *, tick: int = 100) -> None:
        self.device = torch.device("cpu")
        self.tick = int(tick)
        self.current_world = "humanoid"
        self.agent = AgiLoopAgent(obs)
        self._obs = dict(self.agent.env._obs)
        self._system2 = System2Controller()
        self._intention_cortex = IntentionCortex()
        self._motor_arbiter = MotorArbiter()
        self._grounded_lang_ready = False
        self._task_binding = None
        self._intention_state = None
        self._verbal = None

    def _graph_vec_cached(self) -> dict[str, float]:
        return dict(self._obs)

    def set_obs(self, obs: dict[str, float]) -> None:
        self._obs = dict(obs)
        self.agent.env._obs = dict(obs)


@pytest.fixture
def agi_loop_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enable AGI command-loop subsystems for integration tests."""
    monkeypatch.setenv("RKK_GROUNDED_LANG", "1")
    monkeypatch.setenv("RKK_TASK_BINDING", "1")
    monkeypatch.setenv("RKK_INTENTION_CORTEX", "1")
    monkeypatch.setenv("RKK_MOTOR_ARBITER", "1")
    monkeypatch.setenv("RKK_TASK_MIN_TICKS", "1")
    monkeypatch.setenv("RKK_TASK_HOME0_GRACE", "0")


@pytest.fixture
def agi_loop_sim(agi_loop_env: None) -> AgiLoopSim:
    return AgiLoopSim()
