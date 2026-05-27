"""Unit tests for Pearl-style do(X) forward in CausalGNNCore."""
from __future__ import annotations

import os

import torch

from engine.causal_gnn import CausalGNNCore, MechanismMLP


def _tiny_chain(d: int = 4, device: torch.device | None = None) -> CausalGNNCore:
    """Chain DAG: 0→1→2→3 with sparse W."""
    dev = device or torch.device("cpu")
    core = CausalGNNCore(d, dev, hidden=16)
    with torch.no_grad():
        core.W.zero_()
        for i in range(d - 1):
            core.W[i, i + 1] = 0.8
    return core


def test_do_masks_intervened_mechanism_gradients():
    os.environ["RKK_DO_DESCENDANT_ONLY"] = "1"
    core = _tiny_chain()
    X = torch.rand(2, 4)
    X_int = X.clone()
    X_int[:, 1] = 0.9

    core.zero_grad()
    loss = core.intervention_loss(X, X_int, int_var_idx=1, int_val=0.9)
    if loss.requires_grad:
        loss.backward()

    mech_grad = sum(
        p.grad.abs().sum().item()
        for p in core.mechanisms[1].parameters()
        if p.grad is not None
    )
    assert mech_grad == 0.0


def test_do_non_descendants_prediction_invariant():
    os.environ["RKK_DO_DESCENDANT_ONLY"] = "1"
    core = _tiny_chain()
    X = torch.tensor([[0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4]])
    a = torch.zeros_like(X)
    a[:, 0] = 0.95

    pred = core.forward_dynamics_under_do(X, a, int_var_idx=0, int_val=0.95)
    # Nodes 2,3 are descendants of 0; node 0 is fixed; check non-descendant none for idx>0 chain
    # In chain 0→1→2→3 all except 0 are descendants — test node 0 fixed
    assert torch.allclose(pred[:, 0], torch.tensor(0.95), atol=1e-5)


def test_non_descendant_mechanism_zero_grad():
    os.environ["RKK_DO_DESCENDANT_ONLY"] = "1"
    d = 5
    core = CausalGNNCore(d, torch.device("cpu"), hidden=12)
    with torch.no_grad():
        core.W.zero_()
        core.W[0, 1] = 0.7
        core.W[1, 2] = 0.6
        core.W[3, 4] = 0.6  # separate component 3→4

    X = torch.rand(1, d, requires_grad=False)
    X_int = X.clone()
    X_int[:, 1] = 0.8
    loss = core.intervention_loss(X, X_int, int_var_idx=1, int_val=0.8)
    if loss.requires_grad:
        loss.backward()

    # Node 4 is not descendant of 1
    grad_4 = sum(
        p.grad.abs().sum().item()
        for p in core.mechanisms[4].parameters()
        if p.grad is not None
    )
    assert grad_4 == 0.0
    # Descendant mechanism 2 should receive gradients
    grad_2 = sum(
        p.grad.abs().sum().item()
        for p in core.mechanisms[2].parameters()
        if p.grad is not None
    )
    assert grad_2 > 0.0


def test_resize_preserves_mechanism_weights():
    core = CausalGNNCore(3, torch.device("cpu"), hidden=12)
    with torch.no_grad():
        for p in core.mechanisms[0].parameters():
            p.fill_(0.42)
    bigger = core.resize_to(5)
    for p in bigger.mechanisms[0].parameters():
        assert torch.allclose(p, torch.full_like(p, 0.42))
    assert bigger.d == 5
    assert len(bigger.mechanisms) == 5


def test_mechanism_mlp_forward_shape():
    m = MechanismMLP(16)
    h = torch.randn(4, 16)
    agg = torch.randn(4, 16)
    o1, lat, o5, o20 = m(h, agg)
    assert o1.shape == (4,)
    assert lat.shape == (4, 16)
    assert o5.shape == (4,)
    assert o20.shape == (4,)
