"""distance_to_human wiring: vision/physics feed → NS motor sync."""
from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from engine.neuro_symbolic.engine import SymbolicCognitiveEngine
from engine.neuro_symbolic.motor_sync import (
    feed_distance_to_human,
    resolve_distance_to_human,
    sync_ns_motor_every_tick,
)


class _Graph:
    def __init__(self) -> None:
        self.nodes: dict[str, float] = {}

    def snapshot_vec_dict(self) -> dict[str, float]:
        return dict(self.nodes)


def _sim_with_com(com_x: float, com_y: float, com_z: float) -> SimpleNamespace:
    return SimpleNamespace(
        current_world="humanoid",
        _visual_mode=False,
        _visual_env=None,
        _distance_to_human=None,
        _distance_to_human_override=None,
        _tick_phys_state=lambda: {"com_x": com_x, "com_y": com_y, "com_z": com_z},
        agent=SimpleNamespace(graph=_Graph(), env=SimpleNamespace(base_env=None)),
        _ns_engine=SymbolicCognitiveEngine(),
        _ns_bridge=None,
    )


def test_resolve_distance_from_exo_camera() -> None:
    sim_near = _sim_with_com(2.1, -2.1, 1.55)
    sim_far = _sim_with_com(0.0, 0.0, 0.75)
    near = resolve_distance_to_human(sim_near)
    far = resolve_distance_to_human(sim_far)
    assert near is not None and far is not None
    assert near < far


def test_feed_updates_ns_engine_live_distance() -> None:
    sim = _sim_with_com(0.0, 0.0, 0.75)
    dist = feed_distance_to_human(sim)
    assert dist is not None
    assert sim._ns_engine._human_distance_live == pytest.approx(dist)


def test_sync_writes_distance_to_graph_nodes() -> None:
    sim = _sim_with_com(0.0, 0.0, 0.75)
    sync_ns_motor_every_tick(sim)
    assert "distance_to_human" in sim.agent.graph.nodes
    assert sim.agent.graph.nodes["distance_to_human"] < 1.0


def test_human_slot_label_proximity() -> None:
    slots = np.array([0.1, 0.9, 0.2], dtype=np.float32)
    vis_env = SimpleNamespace(
        n_slots=3,
        _last_slots=mock.Mock(detach=lambda: mock.Mock(cpu=lambda: mock.Mock(numpy=lambda: slots))),
        _slot_lexicon={"slot_1": {"label": "[HUMAN] observer"}},
    )
    sim = _sim_with_com(0.0, 0.0, 0.75)
    sim._visual_mode = True
    sim._visual_env = vis_env
    sim._tick_phys_state = lambda: None
    dist = resolve_distance_to_human(sim)
    assert dist is not None
    assert dist < 0.25
