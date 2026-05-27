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


def _predict_with_W(
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
    EIG for candidate interventions: Jensen-Shannon divergence of ensemble preds.

    Returns scalar EIG (higher = more disambiguating action).
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
            preds.append(_predict_with_W(core, Wk, X, a))

        if len(preds) < 2:
            continue

        stack = torch.stack(preds, dim=0)  # (N, B, d)
        mean_pred = stack.mean(dim=0)
        js = 0.0
        for k in range(stack.shape[0]):
            p = stack[k]
            m = 0.5 * (p + mean_pred)
            js += 0.5 * (
                F.mse_loss(p, m, reduction="mean")
                + F.mse_loss(mean_pred, m, reduction="mean")
            )
        eig = float(js.item())
        best_eig = max(best_eig, eig)

    return best_eig


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
    idx = ids.index(var)
    a = torch.zeros(1, d, dtype=torch.float32, device=dev)
    a[0, idx] = float(val)

    ll = torch.zeros(ens.n, device=dev)
    for k in range(ens.n):
        Wk = ens.W_stack[k] * ens.mask
        pred = _predict_with_W(core, Wk, X, a)
        mse = F.mse_loss(pred, Y).item()
        ll[k] = -mse
    return ll


def snapshot_eig_top(
    graph: Any,
    obs: dict[str, float],
    candidates: list[tuple[str, float]] | None = None,
) -> dict:
    """Export for UI snapshot."""
    ids = list(getattr(graph, "_node_ids", []))[:12]
    if candidates is None:
        candidates = [(v, float(obs.get(v, 0.5))) for v in ids if v.startswith("intent_")]
    eig = eig_for_action(graph, obs, candidates)
    return {"eig_top_action": round(eig, 6), "n_candidates": len(candidates)}
