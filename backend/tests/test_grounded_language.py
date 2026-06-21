"""Grounded Language: adapters, vector store, graph wiring."""
from __future__ import annotations

import os
from unittest import mock

import numpy as np
import torch

from engine.causal_gnn import LanguageAdapterIn, LanguageAdapterOut
from engine.grounded_language import (
    GroundedLanguageController,
    SemanticVectorStore,
    intent_speak_node_ids,
    sensory_node_ids,
)


def test_language_adapters_roundtrip_shape() -> None:
    adapter_in = LanguageAdapterIn(64, 8)
    adapter_out = LanguageAdapterOut(8, 64)
    emb = torch.randn(3, 64)
    ch = adapter_in(emb)
    assert ch.shape == (3, 8)
    assert float(ch.min()) >= 0.0 and float(ch.max()) <= 1.0
    recon = adapter_out(ch)
    assert recon.shape == (3, 64)


def test_semantic_vector_store_nearest() -> None:
    store = SemanticVectorStore()
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    store.add("alpha", "a", a)
    store.add("beta", "b", b)
    hits = store.nearest(a, top_k=1)
    assert hits[0][0] == "alpha"


def test_grounded_lang_ingest_without_ollama() -> None:
    class _G:
        nodes: dict[str, float] = {}

        def set_node(self, k: str, v: float) -> None:
            self.nodes[k] = float(v)

        def set_edge(self, a: str, b: str, w: float, *, alpha: float = 0.1) -> None:
            pass

    gl = GroundedLanguageController(device=torch.device("cpu"))
    fake = np.random.randn(64).astype(np.float32)
    fake /= np.linalg.norm(fake) + 1e-9
    gl.embedder.embed = lambda _t: fake  # type: ignore[method-assign]
    gl.store.add("test", "recover", fake)
    graph = _G()
    graph.nodes = {"intent_stop_recover": 0.5, "intent_stride": 0.5}
    out = gl.ingest_command(graph, "Встань")
    assert out["ok"]
    for nid in sensory_node_ids(8):
        assert nid in graph.nodes


def test_ensure_graph_nodes_creates_channels() -> None:
    class _G:
        nodes: dict[str, float] = {}

        def set_node(self, k: str, v: float) -> None:
            self.nodes[k] = float(v)

        def set_edge(self, *a, **k) -> None:
            pass

    gl = GroundedLanguageController(device=torch.device("cpu"))
    g = _G()
    g.nodes = {"intent_stop_recover": 0.5}
    gl.ensure_graph_nodes(g)
    assert sensory_node_ids(8)[0] in g.nodes
    assert intent_speak_node_ids(8)[0] in g.nodes


def test_alignment_step_runs() -> None:
    class _G:
        nodes: dict[str, float] = {}

        def set_node(self, k: str, v: float) -> None:
            self.nodes[k] = float(v)

        def set_edge(self, *a, **k) -> None:
            pass

    gl = GroundedLanguageController(device=torch.device("cpu"))
    emb = np.ones(64, dtype=np.float32) / np.sqrt(64)
    gl.embedder.embed = lambda _t: emb  # type: ignore[method-assign]
    g = _G()
    g.nodes = {}
    gl.ensure_graph_nodes(g)
    loss = gl.alignment_step(g, {"com_z": 0.2, "posture_stability": 0.3})
    assert loss is not None
    assert gl._align_steps == 1


def test_grounded_language_enabled_default() -> None:
    from engine.grounded_language import grounded_language_enabled

    with mock.patch.dict(os.environ, {"RKK_GROUNDED_LANG": "1"}, clear=False):
        assert grounded_language_enabled()
