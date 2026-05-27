"""WeightedGraphEnsemble posterior Bayes update."""
from __future__ import annotations

import torch

from engine.graph_ensemble import WeightedGraphEnsemble


def test_ensemble_posterior_update_shifts_weights():
    d = 4
    dev = torch.device("cpu")
    ens = WeightedGraphEnsemble(d, dev, n=3)
    p0 = ens.posterior().detach().clone()

    ll = torch.tensor([0.0, 2.0, -1.0], device=dev)
    ens.update_posterior(ll)
    p1 = ens.posterior()

    assert int(p1.argmax()) == 1
    assert not torch.allclose(p0, p1)


def test_posterior_mean_shape():
    ens = WeightedGraphEnsemble(5, torch.device("cpu"), n=2)
    mean = ens.posterior_mean()
    assert mean.shape == (5, 5)
