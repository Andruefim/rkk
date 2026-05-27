"""Tests for v-structure detection and ensemble posterior."""
from __future__ import annotations

import numpy as np
import torch

from engine.graph_ensemble import WeightedGraphEnsemble
from engine.structure_learning import detect_v_structures, orient_colliders


def _collider_scm(n: int = 500, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    c = a + b + rng.normal(scale=0.05, size=n)
    return np.column_stack([a, b, c])


def test_v_structure_detection_collider():
    data = _collider_scm()
    colliders = detect_v_structures(data, margin=0.1)
    assert any(t[1] == 2 for t in colliders)


def test_orient_colliders():
    edges = orient_colliders([(0, 2, 1)], d=4)
    pairs = {(e.from_idx, e.to_idx) for e in edges}
    assert (0, 2) in pairs and (1, 2) in pairs


def test_ensemble_posterior_update():
    ens = WeightedGraphEnsemble(4, torch.device("cpu"), n=3)
    ll = torch.tensor([0.0, -2.0, -5.0])
    ens.update_posterior(ll)
    p = ens.posterior()
    assert p.argmax() == 0
    assert ens.entropy() < np.log(3)


def test_ensemble_posterior_mean_shape():
    ens = WeightedGraphEnsemble(5, torch.device("cpu"), n=4)
    mean = ens.posterior_mean()
    assert mean.shape == (5, 5)
