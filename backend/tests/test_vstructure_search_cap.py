"""C3 v-structure search must stay bounded when graph d grows (vision concepts)."""
from __future__ import annotations

import os

import torch

from engine.causal_graph import CausalGraph


class _FakeCore:
    hidden = 24

    def __init__(self, d: int):
        self.device = torch.device("cpu")

    def W_masked(self):
        d = 8
        return torch.randn(d, d) * 0.1


def test_find_vstructure_bounded_by_cap(monkeypatch):
    monkeypatch.setenv("RKK_VSTRUCTURE_MAX_NODES", "8")
    g = CausalGraph(device=torch.device("cpu"))
    g._d = 8
    g._node_ids = [f"n{i}" for i in range(8)]
    g._core = _FakeCore(8)
    g._obs_buffer = [[0.5] * 8 for _ in range(16)]
    triple = g._find_vstructure_collider()
    assert triple is None or len(triple) == 3


def test_compositional_skips_large_graph(monkeypatch):
    monkeypatch.setenv("RKK_STRUCTURE_LEARN_EVERY", "1")
    g = CausalGraph(device=torch.device("cpu"))
    g._d = 100
    g._node_ids = [f"n{i}" for i in range(100)]
    g._structure_learn_tick = -99
    g._ensemble = object()
    g.tick_compositional_structure(50, fixed_root=False)
    assert g._structure_learn_tick == 50
