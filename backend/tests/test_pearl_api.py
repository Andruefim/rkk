"""Unit tests for Pearl-style API on CausalGraph (Phase A)."""
from __future__ import annotations

import os

import torch

from engine.causal_graph import CausalGraph


def _tiny_graph(device: torch.device) -> CausalGraph:
    g = CausalGraph(device)
    for nid in ("com_z", "intent_stride", "floor_friction"):
        g.set_node(nid, 0.5)
    return g


def test_observe_vs_intervene():
    g = _tiny_graph(torch.device("cpu"))
    base = {"com_z": 0.5, "intent_stride": 0.5, "floor_friction": 0.5}
    y0 = g.observe_predict(base)
    y1 = g.intervene_predict(base, "intent_stride", 0.8)
    assert isinstance(y0, dict) and isinstance(y1, dict)
    # Intervention should generally alter prediction vs passive rollout when core trains
    assert set(y0.keys()) == set(g._node_ids)


def test_counterfactual_clamps_u():
    g = _tiny_graph(torch.device("cpu"))
    base = {"com_z": 0.41, "intent_stride": 0.5, "floor_friction": 0.77}
    u_ids = frozenset({"floor_friction"})
    pred_l2 = g.intervene_predict(base, "intent_stride", 0.9)
    pred_l3 = g.counterfactual_predict(base, "intent_stride", 0.9, exogenous_ids=u_ids)
    assert pred_l3["floor_friction"] == base["floor_friction"]
    # If WM moves floor_friction under do(), L3 differs from L2 on that coordinate
    if pred_l2.get("floor_friction") != base["floor_friction"]:
        assert pred_l3["floor_friction"] != pred_l2.get("floor_friction")


def test_empty_u_set_skips_clamp():
    g = _tiny_graph(torch.device("cpu"))
    base = {"com_z": 0.5, "intent_stride": 0.5, "floor_friction": 0.5}
    pred = g.counterfactual_predict(base, "intent_stride", 0.8, exogenous_ids=frozenset())
    assert pred == g.intervene_predict(base, "intent_stride", 0.8)


def test_value_critical_veto_injected_state():
    os.environ["RKK_VALUE_VETO"] = "1"
    from types import SimpleNamespace

    import torch

    from engine.value_layer import ValueLayer, critical_body_diag, value_veto_enabled

    assert value_veto_enabled()
    vl = ValueLayer()
    nodes = {"com_z": 0.1, "posture_stability": 0.1, "intero_energy": 0.5}
    d = critical_body_diag(nodes)
    assert d["fall_imminent"] >= 0.5
    g = _tiny_graph(torch.device("cpu"))
    temporal = SimpleNamespace(h_slow=torch.zeros(1))
    res = vl.check_action(
        variable="intent_stride",
        value=0.85,
        current_nodes=nodes,
        graph=g,
        temporal=temporal,
        current_phi=0.5,
        engine_tick=5000,
    )
    assert res.reason.value == "critical_veto"
    os.environ.pop("RKK_VALUE_VETO", None)
