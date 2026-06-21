"""Tests for AGI architecture loop closures (review roadmap P0–P3)."""
from __future__ import annotations

import os
from unittest import mock

import numpy as np

from engine.neuro_symbolic.bridge import NeuroSymbolicBridge
from engine.neuro_symbolic.engine import SymbolicCognitiveEngine
from engine.neuro_symbolic.knowledge_graph import KnowledgeGraph, bootstrap_humanoid_ontology
from engine.neuro_symbolic.planner import discover_actions_from_graph, HUMANOID_ACTIONS
from engine.neuro_symbolic.predicates import ground_humanoid_state
from engine.system2.controller import (
    System2Controller,
    _meta_plan_every,
    _inner_voice_macro_bias,
)
from engine.system2.learned_student import LearnedMacroStudent, WmPlannerStudent
from engine.working_memory import WorkingMemoryBuffer


def test_wm_planner_student_learns_from_outcome() -> None:
    st = WmPlannerStudent()
    obs = {
        "com_z": 0.25,
        "posture_stability": 0.3,
        "target_dist": 0.5,
        "foot_contact_l": 0.4,
        "foot_contact_r": 0.4,
        "com_x": 0.41,
    }
    st.record_plan("RECOVER_POSTURE", "intent_stop_recover", 0.68, obs)
    w_before = st._W.copy()
    st.learn_from_outcome(True, obs, d_com_z=0.12, d_posture=0.15)
    assert not np.allclose(w_before, st._W)


def test_working_memory_read_write_ttl() -> None:
    wm = WorkingMemoryBuffer(capacity=4)
    wm.write("active_macro", 1.0, text="LOCOMOTE_DELIVERY", tick=10, ttl_ticks=100)
    assert wm.read_text("active_macro") == "LOCOMOTE_DELIVERY"
    assert wm.has("active_macro")
    removed = wm.decay(200)
    assert removed == 1
    assert not wm.has("active_macro")


def test_kg_surprise_and_forgetting() -> None:
    kg = KnowledgeGraph()
    n0 = len(kg.triples)
    assert kg.learn_from_surprise("PathBlocked", 0.9, 0.2, tick=100)
    assert len(kg.triples) > n0
    removed = kg.forget_stale(200_000, max_age_ticks=50)
    assert removed >= 0


def test_symbolic_precision_changes_wm_prediction() -> None:
    """Top-down precision must alter GNN forward (not dead storage)."""
    import torch
    from engine.causal_gnn import CausalGNNCore

    d, hidden = 6, 8
    core = CausalGNNCore(d=d, hidden=hidden, device=torch.device("cpu"))
    X = torch.rand(2, d)
    a = torch.zeros(2, d)
    base = core.forward_dynamics(X, a)
    pw = torch.ones(d)
    pw[2] = 4.0
    pw[4] = 3.5
    boosted = core.forward_dynamics(X, a, precision_weights=pw)
    assert not torch.allclose(base, boosted, atol=1e-5)


def test_causal_graph_builds_precision_tensor() -> None:
    from engine.causal_graph import CausalGraph
    import torch

    g = CausalGraph(device=torch.device("cpu"))
    g._node_ids = ["com_z", "intent_stride", "posture_stability"]
    g._d = 3
    g.apply_symbolic_precision({"intent_stride": 3.2, "posture_stability": 2.1})
    g.apply_attention_focus(["intent_stride"], {"intent_stride": 3.2})
    pw = g._build_precision_weights_tensor(batch_size=2, core_d=3)
    assert pw is not None
    assert float(pw[0, 1]) > 2.0


def test_bridge_symbolic_precision() -> None:
    bridge = NeuroSymbolicBridge()
    ctx = bridge.priors_for_active_inference(
        "LOCOMOTE_DELIVERY",
        {"posture_stability": 0.88, "com_z": 0.52, "self_goal_active": 0.9},
        {"intent_stride": 0.5},
    )
    assert ctx.precision_weights
    assert ctx.attention_focus

    class _G:
        _node_ids = ["intent_stride", "posture_stability"]
        _symbolic_precision: dict[str, float] = {}
        _attention_gate = None

        def apply_symbolic_precision(self, w: dict[str, float]) -> None:
            self._symbolic_precision.update(w)

        def apply_attention_focus(self, focus, pw=None) -> None:
            self._attention_gate = focus

    g = _G()
    bridge.apply_symbolic_precision_to_graph(g, ctx)
    assert g._symbolic_precision.get("intent_stride", 0) > 1.0


def test_discovered_actions_from_graph() -> None:
    st = ground_humanoid_state({"posture_stability": 0.85, "com_z": 0.55})
    nodes = {"intent_wave": 0.72, "intent_stride": 0.5}
    actions = discover_actions_from_graph(nodes, st)
    names = {a.name for a in actions}
    assert "WaveGesture" in names
    assert len(actions) >= len(HUMANOID_ACTIONS)


def test_symbolic_engine_hypotheses() -> None:
    eng = SymbolicCognitiveEngine()
    st = ground_humanoid_state(
        {"posture_stability": 0.15, "com_z": 0.2, "intent_stride": 0.7}
    )
    hyps = eng.generate_hypotheses(st)
    assert any(h.suggested_macro == "RECOVER_POSTURE" for h in hyps)
    rev = eng.suggest_goal_revision(st, "IDLE")
    assert rev is not None
    assert rev["macro"] == "RECOVER_POSTURE"


def test_meta_plan_every_adaptive() -> None:
    base = 48
    assert _meta_plan_every(base, 0.2, fallen=False) < base
    assert _meta_plan_every(base, 0.9, fallen=False) > base
    assert _meta_plan_every(base, 0.5, fallen=True) < base


def test_inner_voice_macro_bias() -> None:
    class _IV:
        def get_active_concepts(self):
            return [("HIGH_FALL_RISK", 0.9)]

    class _Sim:
        _inner_voice = _IV()

    assert _inner_voice_macro_bias(_Sim()) == "RECOVER_POSTURE"


def test_counterfactual_batch_on_graph() -> None:
    from engine.causal_graph import CausalGraph

    g = CausalGraph(device="cpu")
    g._node_ids = ["com_z", "intent_stride"]
    g._d = 2
    g.nodes = {"com_z": 0.5, "intent_stride": 0.5}
    base = dict(g.nodes)
    out = g.counterfactual_predict(base, "intent_stride", 0.7)
    assert "com_z" in out
    batch = g.propagate_counterfactual_batch(base, [("intent_stride", 0.7)])
    assert len(batch) == 1


def test_instrumental_task_bonus() -> None:
    from engine.intristic_objective import instrumental_task_bonus, instrumental_task_enabled

    with mock.patch.dict(os.environ, {"RKK_INTRINSIC_INSTRUMENTAL": "1"}, clear=False):
        assert instrumental_task_enabled()
        b = instrumental_task_bonus("LOCOMOTE_DELIVERY", True)
        assert b > 0


def test_system2_controller_has_working_memory() -> None:
    with mock.patch.dict(os.environ, {"RKK_SYSTEM2": "0"}, clear=False):
        c = System2Controller()
        assert c.working_memory is not None
        c.working_memory.write("goal_x", 0.8, tick=1)
        assert c.working_memory.read("goal_x") == 0.8


def test_learned_macro_student_gradient() -> None:
    st = LearnedMacroStudent()
    obs = {
        "com_z": 0.3,
        "posture_stability": 0.35,
        "target_dist": 0.5,
        "foot_contact_l": 0.5,
        "foot_contact_r": 0.5,
        "com_x": 0.41,
    }
    w0 = st._W.copy()
    st.learn("RECOVER_POSTURE", True, obs, d_com_z=0.1, d_posture=0.08)
    assert not np.allclose(w0, st._W)
