"""Чекпоинт обучаемых подсистем: pack/unpack, отложенное восстановление, genome-приоры."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import torch
import torch.nn as nn

from engine.checkpoint_modules import (
    apply_pending_learnable_modules,
    pack_learnable_modules,
    pending_module_keys,
    unpack_learnable_modules,
)


class _TrainableOwner:
    """Минимальный аналог System1/ReflexStabilizer: сеть + оптимизатор + счётчик."""

    def __init__(self, d_in: int = 4, d_out: int = 2):
        self.net = nn.Linear(d_in, d_out)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.train_steps = 0

    def train_once(self) -> None:
        self.optim.zero_grad()
        loss = self.net(torch.ones(1, self.net.in_features)).pow(2).sum()
        loss.backward()
        self.optim.step()
        self.train_steps += 1


def _make_sim(with_motor_cortex: bool = False):
    from engine.motor_cortex import MotorCortexLibrary as _MotorCortexLibrary

    device = torch.device("cpu")
    graph = SimpleNamespace(_optim=None, _ensemble=None, _traj_head=None, _concept_bridge_head=None)
    agent = SimpleNamespace(system1=_TrainableOwner(), temporal=SimpleNamespace(optim=None), graph=graph)
    sim = SimpleNamespace(
        agent=agent,
        device=device,
        _visual_env=None,
        _cerebellum=None,
        _reflex_stabilizer=_TrainableOwner(6, 3),
        _locomotion_controller=None,
        _inner_voice=None,
        _proprio=None,
        _demon=None,
        _slot_dynamics=None,
        _motor_cortex=_MotorCortexLibrary(device) if with_motor_cortex else None,
    )
    # ReflexStabilizer держит оптимизатор в `opt`, а не `optim`.
    sim._reflex_stabilizer.opt = sim._reflex_stabilizer.optim
    return sim


def _weights(owner) -> torch.Tensor:
    return owner.net.weight.detach().clone()


def test_pack_unpack_restores_weights_and_counters():
    src = _make_sim()
    for _ in range(5):
        src.agent.system1.train_once()
        src._reflex_stabilizer.train_once()
    payload = copy.deepcopy(pack_learnable_modules(src))

    dst = _make_sim()
    assert not torch.allclose(_weights(dst.agent.system1), _weights(src.agent.system1))

    out = unpack_learnable_modules(dst, payload)

    assert "system1" in out["applied"]
    assert "reflex_stabilizer" in out["applied"]
    assert torch.allclose(_weights(dst.agent.system1), _weights(src.agent.system1))
    assert torch.allclose(_weights(dst._reflex_stabilizer), _weights(src._reflex_stabilizer))
    assert dst.agent.system1.train_steps == src.agent.system1.train_steps


def test_optimizer_state_survives_roundtrip():
    src = _make_sim()
    for _ in range(4):
        src.agent.system1.train_once()
    payload = copy.deepcopy(pack_learnable_modules(src))

    dst = _make_sim()
    unpack_learnable_modules(dst, payload)

    src_steps = [s["step"] for s in src.agent.system1.optim.state.values()]
    dst_steps = [s["step"] for s in dst.agent.system1.optim.state.values()]
    assert src_steps and dst_steps == src_steps


def test_motor_cortex_restored_after_lazy_creation():
    from engine.motor_cortex import MotorCortexLibrary as _MotorCortexLibrary

    src = _make_sim(with_motor_cortex=True)
    prog = src._motor_cortex.ensure_program("walk")
    with torch.no_grad():
        for p in prog.net.parameters():
            p.add_(0.25)
    prog.train_steps = 42
    src._motor_cortex.cpg_weight = 0.37
    payload = copy.deepcopy(pack_learnable_modules(src))

    dst = _make_sim(with_motor_cortex=False)
    out = unpack_learnable_modules(dst, payload)

    assert "motor_cortex" not in out["applied"]
    assert "motor_cortex" in pending_module_keys(dst)

    dst._motor_cortex = _MotorCortexLibrary(dst.device)
    applied = apply_pending_learnable_modules(dst)

    assert "motor_cortex" in applied
    assert pending_module_keys(dst) == []
    restored = dst._motor_cortex.programs["walk"]
    assert restored.train_steps == 42
    assert abs(dst._motor_cortex.cpg_weight - 0.37) < 1e-6
    for a, b in zip(restored.net.parameters(), prog.net.parameters()):
        assert torch.allclose(a, b)


def test_vision_cortex_weights_persist():
    from engine.causal_vision import make_visual_cortex

    device = torch.device("cpu")
    src = _make_sim()
    src._visual_env = SimpleNamespace(cortex=make_visual_cortex(device, n_slots=4))
    with torch.no_grad():
        for p in src._visual_env.cortex.parameters():
            p.mul_(0.5).add_(0.01)
    src._visual_env.cortex.n_train = 17
    payload = copy.deepcopy(pack_learnable_modules(src))

    dst = _make_sim()
    dst._visual_env = SimpleNamespace(cortex=make_visual_cortex(device, n_slots=4))
    out = unpack_learnable_modules(dst, payload)

    assert "vision_cortex" in out["applied"]
    assert dst._visual_env.cortex.n_train == 17
    for a, b in zip(dst._visual_env.cortex.parameters(), src._visual_env.cortex.parameters()):
        assert torch.allclose(a, b)


def test_genome_priors_do_not_overwrite_learned_edges():
    from engine.causal_graph import CausalGraph
    from engine.genome.priors import CAUSAL_PRIORS, apply_causal_priors

    graph = CausalGraph(torch.device("cpu"))
    prior = next(
        (p for p in CAUSAL_PRIORS if p["from"] in graph._node_ids and p["to"] in graph._node_ids),
        None,
    )
    if prior is None:
        graph.rebind_variables(
            [str(p["from"]) for p in CAUSAL_PRIORS[:1]] + [str(CAUSAL_PRIORS[0]["to"])],
            {},
        )
        prior = CAUSAL_PRIORS[0]

    i = graph._node_ids.index(prior["from"])
    j = graph._node_ids.index(prior["to"])
    learned = 0.83
    with torch.no_grad():
        graph._core.W[i, j] = learned

    apply_causal_priors(graph, only_missing=True)
    assert abs(float(graph._core.W[i, j].item()) - learned) < 1e-6

    apply_causal_priors(graph, only_missing=False)
    assert abs(float(graph._core.W[i, j].item()) - learned) > 1e-3
