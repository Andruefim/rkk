"""Discovery rate without GT edges (humanoid)."""
from __future__ import annotations

from unittest import mock

import numpy as np

from engine.agent import RKKAgent


class _EnvNoGt:
    preset = "humanoid"
    variable_ids = ["posture_stability"]

    def gt_edges(self):
        return []

    def observe(self):
        return {"posture_stability": 0.8}


def test_discovery_rate_positive_without_gt() -> None:
    with mock.patch.object(RKKAgent, "__init__", lambda self, *a, **k: None):
        agent = RKKAgent.__new__(RKKAgent)
        agent.env = _EnvNoGt()
        agent._disc_rate_tick = -1
        agent._disc_rate_val = 0.0
        agent._total_interventions = 300
        agent.graph = mock.Mock()
        agent.graph.train_losses = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75]
        agent._w_meta = None

    dr = agent._discovery_rate_for_tick(100)
    assert dr > 0.0

    agent._disc_rate_tick = -1
    agent._total_interventions = 0
    agent.graph.train_losses = []
    dr2 = agent._discovery_rate_for_tick(101)
    assert dr2 >= 0.02
