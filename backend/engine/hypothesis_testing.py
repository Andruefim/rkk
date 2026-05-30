"""
hypothesis_testing.py — Expected Information Gain over graph ensemble (Phase 2).

EIG between forward_dynamics predictions under W_1..W_N for candidate do(var).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from engine.graph_ensemble import WeightedGraphEnsemble


def intent_vars_in_graph(graph: Any) -> list[str]:
    ids = list(getattr(graph, "_node_ids", []) or [])
    return [
        str(n)
        for n in ids
        if str(n).startswith("intent_") or str(n).startswith("phys_intent_")
    ]


def _align_step_pred(t: torch.Tensor) -> torch.Tensor:
    """Normalize forward_dynamics output to (1, d) for ensemble EIG/LL."""
    x = t.detach().float()
    while x.dim() > 2:
        x = x.squeeze(0)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    return x


def _predict_with_W(
    graph: Any,
    core: Any,
    W: torch.Tensor,
    X: torch.Tensor,
    a: torch.Tensor,
) -> torch.Tensor:
    """One-step prediction using core mechanisms but alternate W."""
    saved = core.W.data.clone()
    try:
        with torch.no_grad():
            d = min(W.shape[0], core.W.shape[0])
            core.W[:d, :d].copy_(W[:d, :d])
        fn = getattr(graph, "forward_dynamics", None)
        if callable(fn):
            return fn(X, a)
        return core.forward_dynamics(X, a)
    finally:
        with torch.no_grad():
            core.W.copy_(saved)


def eig_for_action(
    graph: Any,
    obs: dict[str, float],
    candidate_interventions: list[tuple[str, float]],
    *,
    ensemble: WeightedGraphEnsemble | None = None,
) -> float:
    """
    EIG for candidate interventions: posterior-weighted ensemble disagreement.
    """
    ens = ensemble or getattr(graph, "_ensemble", None)
    core = getattr(graph, "_core", None)
    if ens is None or core is None or not candidate_interventions:
        return 0.0

    ids = list(getattr(graph, "_node_ids", []))
    if not ids:
        return 0.0
    d = len(ids)
    dev = getattr(graph, "device", torch.device("cpu"))
    weights = ens.posterior().detach()

    vec = [float(obs.get(nid, graph.nodes.get(nid, 0.5))) for nid in ids]
    X = torch.tensor([vec], dtype=torch.float32, device=dev)

    best_eig = 0.0
    for var, val in candidate_interventions:
        if var not in ids:
            continue
        idx = ids.index(var)
        a = torch.zeros(1, d, dtype=torch.float32, device=dev)
        a[0, idx] = float(val)

        preds: list[torch.Tensor] = []
        for k in range(ens.n):
            Wk = ens.W_stack[k] * ens.mask
            preds.append(_align_step_pred(_predict_with_W(graph, core, Wk, X, a)))

        if len(preds) < 2:
            continue

        stack = torch.stack(preds, dim=0)
        mean_pred = (weights.view(-1, 1, 1) * stack).sum(dim=0)
        js = torch.tensor(0.0, device=dev)
        for k in range(stack.shape[0]):
            wk = weights[k]
            if float(wk.item()) <= 1e-8:
                continue
            p = stack[k]
            m = 0.5 * (p + mean_pred)
            js = js + wk * (
                F.mse_loss(p, m, reduction="mean")
                + F.mse_loss(mean_pred, m, reduction="mean")
            )
        eig = float(js.item())
        best_eig = max(best_eig, eig)

    return best_eig


def ensemble_log_likelihood_per_member(
    graph: Any,
    cf_pred: dict[str, float],
    cf_obs: dict[str, float],
    *,
    ensemble: WeightedGraphEnsemble | None = None,
    sigma: float = 0.12,
) -> torch.Tensor:
    """Per-hypothesis Gaussian log-likelihood on all observed counterfactual keys."""
    ens = ensemble or getattr(graph, "_ensemble", None)
    if ens is None:
        return torch.zeros(1)

    keys = [k for k in cf_obs if k in cf_pred]
    if not keys:
        return torch.zeros(ens.n, device=ens.device)

    dev = ens.device
    ll = torch.zeros(ens.n, device=dev)
    inv_var = 1.0 / max(sigma * sigma, 1e-6)
    for k in range(ens.n):
        total = 0.0
        for key in keys:
            err = float(cf_obs[key]) - float(cf_pred.get(key, 0.5))
            total += -0.5 * inv_var * err * err
        ll[k] = total
    ll = ll - ll.max()
    return ll


def ensemble_log_likelihood_fast(
    graph: Any,
    obs_before: dict[str, float],
    obs_after: dict[str, float],
    *,
    ensemble: WeightedGraphEnsemble | None = None,
) -> torch.Tensor:
    """
    Batched per-hypothesis LL from W_stack only (no GNN forward per member).
    ~100× cheaper than ensemble_log_likelihood on GPU; still differentiates weights.
    """
    ens = ensemble or getattr(graph, "_ensemble", None)
    if ens is None:
        return torch.zeros(1)

    ids = list(getattr(graph, "_node_ids", []))
    d = len(ids)
    if d == 0:
        return torch.zeros(ens.n, device=ens.device)

    dev = ens.device
    with torch.no_grad():
        X = torch.tensor(
            [[float(obs_before.get(n, 0.5)) for n in ids]],
            dtype=torch.float32,
            device=dev,
        )
        Y = torch.tensor(
            [[float(obs_after.get(n, 0.5)) for n in ids]],
            dtype=torch.float32,
            device=dev,
        )
        W = ens.W_stack * ens.mask.unsqueeze(0)
        d_use = min(d, W.shape[-1])
        Xn = X[:, :d_use].expand(ens.n, -1, -1)
        Yn = Y[:, :d_use].unsqueeze(0).expand(ens.n, -1, -1)
        pred = torch.bmm(Xn, W[:, :d_use, :d_use].transpose(1, 2))
        err = (pred - Yn).abs().mean(dim=(1, 2))
        ll = -err * 50.0
        ll = (ll - ll.mean()) * 5.0
    return ll


def ensemble_log_likelihood(
    graph: Any,
    obs_before: dict[str, float],
    obs_after: dict[str, float],
    var: str,
    val: float,
    ensemble: WeightedGraphEnsemble | None = None,
) -> torch.Tensor:
    """Per-hypothesis negative MSE as log-likelihood proxy for Bayes update."""
    ens = ensemble or getattr(graph, "_ensemble", None)
    core = getattr(graph, "_core", None)
    if ens is None or core is None:
        return torch.zeros(1)

    ids = list(getattr(graph, "_node_ids", []))
    if var not in ids:
        return torch.zeros(ens.n, device=ens.device)

    d = len(ids)
    dev = getattr(graph, "device", torch.device("cpu"))
    X = torch.tensor(
        [[float(obs_before.get(n, 0.5)) for n in ids]],
        dtype=torch.float32,
        device=dev,
    )
    Y = torch.tensor(
        [[float(obs_after.get(n, 0.5)) for n in ids]],
        dtype=torch.float32,
        device=dev,
    )
    if Y.shape[-1] != d:
        return torch.zeros(ens.n, device=dev)
    idx = ids.index(var)
    a = torch.zeros(1, d, dtype=torch.float32, device=dev)
    a[0, idx] = float(val)

    ll = torch.zeros(ens.n, device=dev)
    delta = (Y - X).abs()
    w_dim = delta / delta.sum().clamp(min=1e-6)
    for k in range(ens.n):
        Wk = ens.W_stack[k] * ens.mask
        pred = _align_step_pred(_predict_with_W(graph, core, Wk, X, a))
        err = (pred - Y).abs()
        mse = float((err * w_dim).sum().item())
        ll[k] = -mse * 50.0
    ll = (ll - ll.mean()) * 5.0
    return ll


def snapshot_eig_top(
    graph: Any,
    obs: dict[str, float],
    candidates: list[tuple[str, float]] | None = None,
) -> dict:
    """Export for UI snapshot."""
    ids = intent_vars_in_graph(graph)[:12]
    if candidates is None:
        candidates = [(v, float(obs.get(v, graph.nodes.get(v, 0.5)))) for v in ids]
    eig = eig_for_action(graph, obs, candidates)
    ens = getattr(graph, "_ensemble", None)
    out: dict[str, Any] = {"eig_top_action": round(eig, 6), "n_candidates": len(candidates)}
    if ens is not None:
        out["ensemble_weights"] = ens.snapshot().get("weights")
        out["ensemble_entropy"] = ens.snapshot().get("entropy")
    return out
