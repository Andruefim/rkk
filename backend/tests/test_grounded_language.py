"""Grounded Language: adapters, vector store, graph wiring."""
from __future__ import annotations

import os
from unittest import mock

import numpy as np
import torch

from engine.causal_gnn import LanguageAdapterIn, LanguageAdapterOut
from engine.grounded_language import (
    FallbackEmbeddingClient,
    GroundedLanguageController,
    OllamaEmbeddingClient,
    SemanticVectorStore,
    command_tag_for_text,
    intent_speak_node_ids,
    motor_intents_from_tag,
    phrase_for_human_task,
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


def test_fallback_embedder_deterministic() -> None:
    fb = FallbackEmbeddingClient(embed_dim=64)
    a = fb.embed("Встань, ты упал")
    b = fb.embed("Встань, ты упал")
    c = fb.embed("Step forward")
    assert a is not None and b is not None and c is not None
    assert a.shape == (64,)
    assert np.allclose(a, b)
    assert not np.allclose(a, c)


def test_ollama_embedder_falls_back_without_ollama() -> None:
    client = OllamaEmbeddingClient(embed_dim=64)
    with mock.patch.object(client, "_embed_ollama", return_value=None):
        emb = client.embed("recover now")
    assert emb is not None
    assert emb.shape == (64,)
    assert float(np.linalg.norm(emb)) > 0.99


def test_ingest_command_fallback_without_ollama() -> None:
    class _G:
        nodes: dict[str, float] = {}

        def set_node(self, k: str, v: float) -> None:
            self.nodes[k] = float(v)

        def set_edge(self, a: str, b: str, w: float, *, alpha: float = 0.1) -> None:
            pass

    gl = GroundedLanguageController(device=torch.device("cpu"))
    with mock.patch.object(gl.embedder, "_embed_ollama", return_value=None):
        graph = _G()
        graph.nodes = {"intent_stop_recover": 0.5, "intent_stride": 0.5}
        out = gl.ingest_command(graph, "Встань")
    assert out["ok"] is True
    for nid in sensory_node_ids(8):
        assert nid in graph.nodes


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


def test_motor_intents_from_tag() -> None:
    recover = motor_intents_from_tag("recover")
    assert recover["intent_stop_recover"] == 0.72
    assert motor_intents_from_tag("") == {}


def test_ingest_records_language_interventions() -> None:
    class _G:
        nodes: dict[str, float] = {}
        _int_buffer: list = []

        def set_node(self, k: str, v: float) -> None:
            self.nodes[k] = float(v)

        def set_edge(self, *a, **k) -> None:
            pass

        def record_language_interventions(self, obs_before, obs_after):
            for nid in sensory_node_ids(8):
                if nid not in self.nodes:
                    continue
                before = float(obs_before.get(nid, 0.5))
                after = float(obs_after.get(nid, 0.5))
                if abs(after - before) >= 1e-4:
                    self._int_buffer.append(
                        {"idx": 0, "val": after, "source": "grounded_language"}
                    )
            return len(self._int_buffer)

    gl = GroundedLanguageController(device=torch.device("cpu"))
    fake = np.random.randn(64).astype(np.float32)
    fake /= np.linalg.norm(fake) + 1e-9
    gl.embedder.embed = lambda _t: fake  # type: ignore[method-assign]
    graph = _G()
    graph.nodes = {"intent_stop_recover": 0.5}
    gl.ingest_command(graph, "Встань")
    assert graph._int_buffer
    assert graph._int_buffer[0]["source"] == "grounded_language"


def test_motor_arbiter_grounded_language_source() -> None:
    from engine.motor_arbiter import DEFAULT_SOURCE_PRECISION, MotorArbiter

    assert DEFAULT_SOURCE_PRECISION["grounded_language"] == 0.68
    arb = MotorArbiter()
    arb.register_from_dict(
        "grounded_language",
        motor_intents_from_tag("recover"),
    )
    assert arb._intents
    assert arb._intents[0].source == "grounded_language"
    assert arb._intents[0].stop_recover == 0.72


def test_pending_grounded_motor_cleared_after_drain() -> None:
    from engine.motor_arbiter import MotorArbiter
    from tests.conftest import AgiLoopSim

    sim = AgiLoopSim()
    sim._motor_arbiter = MotorArbiter()
    sim._ensure_grounded_motor_drain_hook()

    sim._queue_grounded_motor_intent("recover")
    assert sim._pending_grounded_motor is not None
    sim._drain_pending_grounded_motor()
    assert sim._pending_grounded_motor is None

    sim._motor_arbiter.begin_tick()
    assert len(sim._motor_arbiter._intents) == 0

    sim._queue_grounded_motor_intent("locomote")
    sim._drain_pending_grounded_motor()
    sim._drain_pending_grounded_motor()
    assert sim._pending_grounded_motor is None
    assert len(sim._motor_arbiter._intents) == 1


def test_ingest_command_keyword_beats_embed_false_positive() -> None:
    class _G:
        nodes: dict[str, float] = {}

        def set_node(self, k: str, v: float) -> None:
            self.nodes[k] = float(v)

        def set_edge(self, a: str, b: str, w: float, *, alpha: float = 0.1) -> None:
            pass

    gl = GroundedLanguageController(device=torch.device("cpu"))
    unstable = np.random.randn(64).astype(np.float32)
    unstable /= np.linalg.norm(unstable) + 1e-9
    gl.embedder.embed = lambda _t: unstable  # type: ignore[method-assign]
    gl.store.add("Теряю равновесие", "unstable", unstable)
    graph = _G()
    graph.nodes = {"intent_stop_recover": 0.5, "intent_stride": 0.5}
    cmd = "подойди к объекту перед тобой и дотронься его"
    out = gl.ingest_command(graph, cmd, apply_motor_patch=False)
    assert out["ok"] is True
    assert out.get("tag") != "unstable"
    assert command_tag_for_text(cmd) == ""


def test_phrase_for_human_task_uses_stage_not_balance_tag() -> None:
    phrase = phrase_for_human_task(
        "подойди к объекту перед тобой и дотронься его",
        stage_kind="approach",
    )
    assert phrase == "Иду к объекту"
    assert "равновес" not in phrase.lower()
