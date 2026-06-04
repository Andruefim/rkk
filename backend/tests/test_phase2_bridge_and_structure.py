"""Phase 2 gate smoke: bridge WM loss, discovery split, v-structure ensemble, skill chains."""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from engine.causal_graph import CausalGraph
from engine.concept_store import CONCEPT_DEFS
from engine.features.humanoid.constants import VAR_NAMES
from engine.graph_ensemble import WeightedGraphEnsemble
from engine.skill_library import Skill, SkillLibrary


def _tiny_graph(d: int = 8) -> CausalGraph:
    g = CausalGraph(torch.device("cpu"))
    g.set_env_preset("humanoid")
    for i, nid in enumerate(VAR_NAMES[:d]):
        g.set_node(nid, 0.4 + 0.01 * i)
    g._rebuild_core()
    for _ in range(20):
        obs = {nid: float(g.nodes.get(nid, 0.5)) for nid in g._node_ids}
        g.record_observation(obs)
    return g


def test_predict_concept_logits_and_l_bridge_in_train_step_seq(monkeypatch):
    monkeypatch.setenv("RKK_BRIDGE_LOSS_WEIGHT", "0.20")
    monkeypatch.setenv("RKK_BRIDGE_LOSS_EVERY", "1")
    monkeypatch.setenv("RKK_WM_SEQ_LEN", "8")
    g = _tiny_graph(10)
    label = CONCEPT_DEFS[0][0]
    state = [float(g.nodes.get(n, 0.5)) for n in g._node_ids]
    for _ in range(12):
        g.record_bridge_transition(state, [label])
    seq = [[state for _ in range(8)]]
    g._seq_buffer.append((seq[0], 1.0))
    g._seq_buffer.append((seq[0], 1.0))

    z = g._encode_states_latent_flat(
        torch.tensor([state], dtype=torch.float32, device=g.device)
    )
    assert z is not None
    logits = g.predict_concept_logits(z)
    assert logits.shape == (1, len(CONCEPT_DEFS))

    out = g._train_step_seq()
    assert out is not None
    assert "l_bridge" in out
    assert np.isfinite(out["l_bridge"])
    assert out["l_bridge"] >= 0.0


def test_edge_age_and_discovery_new_frac_snapshot():
    g = _tiny_graph(6)
    a, b, c = g._node_ids[0], g._node_ids[1], g._node_ids[2]
    g.set_edge(a, b, 0.4, 0.1)
    fields = g.discovery_snapshot_fields()
    assert "discovery_new_frac" in fields
    assert "edge_age_at_activation" in fields
    assert fields["discovery_new_count"] >= 1
    g.tick_edge_ages()
    assert g._edge_age.get((a, b), 0) >= 1


def test_vstructure_ensemble_orientations_differ():
    d = 6
    ens = WeightedGraphEnsemble(d, torch.device("cpu"), n=4)
    W0 = ens.W_stack[0].clone()
    ens.apply_vstructure_orientations(0, 2, 4, n_orientations=4)
    diffs = [
        float((ens.W_stack[k] - W0).abs().sum().item()) for k in range(4)
    ]
    assert max(diffs) > 0.0
    assert not all(abs(x - diffs[0]) < 1e-9 for x in diffs[1:])


def test_skill_chain_depth_and_pe_gate(monkeypatch):
    monkeypatch.setenv("RKK_SKILL_CHAIN_MAX_DEPTH", "4")
    monkeypatch.setenv("RKK_SKILL_CHAIN_PE_MAX", "0.25")

    class SimStub:
        _skill_chain: list[str]
        _skill_exec: dict | None

        def __init__(self):
            self._skill_chain = []
            self._skill_exec = None
            self.agent = type("A", (), {"env": type("E", (), {"observe": lambda s: {"com_z": 0.9, "posture_stability": 0.9, "foot_contact_l": 0.9, "foot_contact_r": 0.9}})()})()

    from engine.features.simulation.mixin_skills import SimulationSkillsMixin

    sim = SimStub()
    mixin = SimulationSkillsMixin()
    mixin._skill_chain = []
    mixin._skill_exec = None
    mixin.agent = sim.agent
    mixin.current_world = "humanoid"
    mixin._fixed_root_active = False
    mixin.tick = 100
    mixin._skill_library = SkillLibrary()
    sk = Skill(
        name="hold_stance",
        precondition=lambda s: True,
        action_sequence=[("intent_stride", 0.5)],
        postcondition=lambda s: True,
    )
    st = {"com_z": 0.9, "posture_stability": 0.9, "foot_contact_l": 0.9, "foot_contact_r": 0.9}
    assert mixin._skill_chain_max_depth() == 4
    mixin._skill_exec = {"skill": sk, "index": 0, "obs_before": st, "chain_depth": 4}
    assert not mixin._maybe_chain_next_skill(
        sk, st, prediction_error=0.1, obs_before_init=st
    )
    mixin._skill_exec = {"skill": sk, "index": 0, "obs_before": st, "chain_depth": 0}
    assert not mixin._maybe_chain_next_skill(
        sk, st, prediction_error=0.5, obs_before_init=st
    )
