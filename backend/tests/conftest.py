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
        self._obs.setdefault("com_x", 0.0)
        self._obs.setdefault("com_y", 0.0)
        self._motor_state = {k: float(v) for k, v in self._obs.items() if k.startswith("intent_")}
        self.preset = "humanoid"
        self._intero_state = {
            "intero_energy": float(self._obs.get("intero_energy", 0.85)),
            "intero_stress": float(self._obs.get("intero_stress", 0.12)),
        }
        self._scene_extras: dict = {}
        self._manip_chair = [1.0, 0.0, 0.4]
        self._manip_body_id = 9001

    @property
    def base_env(self) -> AgiLoopEnv:
        return self

    def observe(self) -> dict[str, float]:
        out = dict(self._obs)
        out["intero_energy"] = float(self._intero_state["intero_energy"])
        out["intero_stress"] = float(self._intero_state["intero_stress"])
        return out

    def get_state(self) -> dict[str, float]:
        return {
            "com_x": float(self._obs.get("com_x", 0.0)),
            "com_y": float(self._obs.get("com_y", 0.0)),
            "com_z": float(self._obs.get("com_z", 0.5)),
            "torso_yaw": float(self._obs.get("torso_yaw", 0.0)),
        }

    def get_sandbox_scene_extras(self) -> dict:
        return dict(self._scene_extras)

    def set_scene_extras(self, extras: dict) -> None:
        self._scene_extras = dict(extras)

    def resolve_manipulation_target(
        self,
        query: str,
        *,
        agent_forward: tuple[float, float] | None = None,
        embed_fn=None,
    ):
        from engine.object_resolver import resolve_manipulation_target

        xy = (float(self._obs.get("com_x", 0.0)), float(self._obs.get("com_y", 0.0)))
        return resolve_manipulation_target(
            query,
            self._scene_extras,
            agent_xy=xy,
            agent_forward=agent_forward,
            embed_fn=embed_fn,
        )

    def get_manipulation_target_pose(self, ref: str) -> dict | None:
        reg = self._scene_extras.get("registry") or []
        if isinstance(reg, dict):
            reg = list(reg.values())
        for row in reg:
            if str(row.get("ref")) == str(ref):
                if int(row.get("body_id", -1)) == int(self._manip_body_id):
                    return {
                        "ref": ref,
                        "body_id": self._manip_body_id,
                        "x": float(self._manip_chair[0]),
                        "y": float(self._manip_chair[1]),
                        "z": float(self._manip_chair[2]),
                    }
                return {
                    "ref": ref,
                    "body_id": row.get("body_id"),
                    "x": float(row.get("x", self._manip_chair[0])),
                    "y": float(row.get("y", self._manip_chair[1])),
                    "z": float(row.get("z", self._manip_chair[2])),
                }
        if str(ref) == "manip_chair_front":
            return {
                "ref": ref,
                "body_id": self._manip_body_id,
                "x": float(self._manip_chair[0]),
                "y": float(self._manip_chair[1]),
                "z": float(self._manip_chair[2]),
            }
        return None

    def apply_manipulation_push(
        self,
        body_id: int,
        direction_xy: tuple[float, float],
        force_n: float | None = None,
    ) -> dict:
        if int(body_id) != int(self._manip_body_id):
            return {"applied": False, "reason": "unknown_body"}
        dx, dy = float(direction_xy[0]), float(direction_xy[1])
        n = float(np.hypot(dx, dy))
        if n < 1e-6:
            return {"applied": False, "reason": "zero_direction"}
        scale = 0.08 * float(force_n or 38.0) / 38.0
        self._manip_chair[0] += (dx / n) * scale
        self._manip_chair[1] += (dy / n) * scale
        return {"applied": True, "body_id": int(body_id)}

    def manip_approach_m(self) -> float:
        from engine.features.humanoid.environment import EnvironmentHumanoid

        return EnvironmentHumanoid.manip_approach_m()

    def manip_reach_min_ticks(self) -> int:
        from engine.features.humanoid.environment import EnvironmentHumanoid

        return EnvironmentHumanoid.manip_reach_min_ticks()

    def manip_push_every(self) -> int:
        from engine.features.humanoid.environment import EnvironmentHumanoid

        return EnvironmentHumanoid.manip_push_every()

    def apply_task_outcome_affect(self, success: bool) -> dict[str, float]:
        from engine.features.humanoid.environment import EnvironmentHumanoid

        stub = EnvironmentHumanoid.__new__(EnvironmentHumanoid)
        stub._intero_state = dict(self._intero_state)
        out = stub.apply_task_outcome_affect(success)
        self._intero_state = dict(stub._intero_state)
        self._obs["intero_energy"] = float(self._intero_state["intero_energy"])
        self._obs["intero_stress"] = float(self._intero_state["intero_stress"])
        return out

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
        self._task_tree_ctrl = None
        self._task_tree_kind = ""
        self._manip_episode = None
        self._manip_resolved = None
        self._manip_diag: dict = {}
        self._task_tree_reported = False
        self._task_tree_affect_done = False
        self._task_tree_cleared_pending_ack = False
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
    monkeypatch.setenv("RKK_TASK_TREE", "1")
    monkeypatch.setenv("RKK_INTENTION_CORTEX", "1")
    monkeypatch.setenv("RKK_MOTOR_ARBITER", "1")
    monkeypatch.setenv("RKK_TASK_MIN_TICKS", "1")
    monkeypatch.setenv("RKK_TASK_HOME0_GRACE", "0")
    monkeypatch.setenv("RKK_MANIP_APPROACH_M", "0.9")
    monkeypatch.setenv("RKK_MANIP_REACH_MIN_TICKS", "2")
    monkeypatch.setenv("RKK_MANIP_PUSH_EVERY", "1")
    monkeypatch.setenv("RKK_MANIP_MIN_DISP", "0.12")


@pytest.fixture
def agi_loop_sim(agi_loop_env: None) -> AgiLoopSim:
    return AgiLoopSim()
