"""
hypothesis_testing.py — Expected Information Gain over graph ensemble (Phase 2).

EIG between forward_dynamics predictions under W_1..W_N for candidate do(var).
"""
from __future__ import annotations

import os
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


def _eig_batch_w_enabled() -> bool:
    return os.environ.get("RKK_EIG_BATCH_W", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _align_step_pred(t: torch.Tensor) -> torch.Tensor:
    """Normalize forward_dynamics output to (1, d) for ensemble EIG/LL."""
    x = t.detach().float()
    while x.dim() > 2:
        x = x.squeeze(0)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    return x


def _pad_W_batch(W: torch.Tensor, core_d: int) -> torch.Tensor:
    """Pad ensemble W (graph d) to CausalGNNCore.d."""
    cd = int(core_d)
    if W.shape[-1] == cd and W.shape[-2] == cd:
        return W
    pad_r = max(0, cd - int(W.shape[-2]))
    pad_c = max(0, cd - int(W.shape[-1]))
    if pad_r or pad_c:
        return F.pad(W, (0, pad_c, 0, pad_r))
    return W[..., :cd, :cd]


def _pad_wm_inputs(X: torch.Tensor, a: torch.Tensor, core_d: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad active graph state to CausalGNNCore.d without CausalGraph.forward_dynamics overhead."""
    cd = int(core_d)
    if X.shape[-1] < cd:
        pad = cd - int(X.shape[-1])
        X = F.pad(X, (0, pad))
        a = F.pad(a, (0, pad))
    elif X.shape[-1] > cd:
        X = X[..., :cd]
        a = a[..., :cd]
    return X, a


def _predict_with_W(
    graph: Any,
    core: Any,
    W: torch.Tensor,
    X: torch.Tensor,
    a: torch.Tensor,
) -> torch.Tensor:
    """Single hypothesis — fallback when batched path unavailable."""
    saved = core.W.data.clone()
    try:
        with torch.no_grad():
            d = min(W.shape[0], core.W.shape[0])
            core.W[:d, :d].copy_(W[:d, :d])
            Xc, ac = _pad_wm_inputs(X, a, int(getattr(core, "d", X.shape[-1])))
            pred = core.forward_dynamics(Xc, ac)
        out_d = int(getattr(graph, "_d", pred.shape[-1]))
        return pred[..., :out_d]
    finally:
        with torch.no_grad():
            core.W.copy_(saved)


def _ensemble_js_eig(stack: torch.Tensor, weights: torch.Tensor) -> float:
    """Posterior-weighted Jensen–Shannon proxy; stack (N, d) or (N, 1, d)."""
    if stack.dim() == 2:
        stack = stack.unsqueeze(1)
    if stack.shape[0] < 2:
        return 0.0
    w = weights.view(-1, 1, 1)
    mean_pred = (w * stack).sum(dim=0, keepdim=True)
    m = 0.5 * (stack + mean_pred)
    per_k = F.mse_loss(stack, m, reduction="none").mean(dim=(1, 2)) + F.mse_loss(
        mean_pred.expand_as(stack), m, reduction="none"
    ).mean(dim=(1, 2))
    return float((w.squeeze(-1).squeeze(-1) * per_k).sum().item())


def _predict_ensemble_members(
    graph: Any,
    core: Any,
    ens: WeightedGraphEnsemble,
    Xc: torch.Tensor,
    a: torch.Tensor,
    out_d: int,
) -> torch.Tensor:
    """(N, out_d) predictions; one batched forward when core.forward_dynamics_batched_W exists."""
    fn = getattr(core, "forward_dynamics_batched_W", None)
    core_d = int(core.d)
    Wm = _pad_W_batch(ens.W_stack * ens.mask.unsqueeze(0), core_d)
    if callable(fn):
        with torch.no_grad():
            pred = fn(Wm, a, Xc)
        return pred[..., :out_d]
    preds: list[torch.Tensor] = []
    for k in range(ens.n):
        Wk = Wm[k]
        preds.append(_align_step_pred(_predict_with_W(graph, core, Wk, Xc, a)))
    return torch.stack([p.squeeze(0) for p in preds], dim=0)


def _forward_candidates_batched(
    core: Any,
    ens: WeightedGraphEnsemble,
    Xc: torch.Tensor,
    a_batch: torch.Tensor,
    out_d: int,
) -> torch.Tensor:
    """K candidates × N hypotheses in one forward: returns (K, N, out_d)."""
    fn = getattr(core, "forward_dynamics_batched_W", None)
    if not callable(fn):
        raise RuntimeError("forward_dynamics_batched_W missing")
    K = int(a_batch.shape[0])
    N = int(ens.n)
    cd = int(core.d)
    W_base = _pad_W_batch(ens.W_stack * ens.mask.unsqueeze(0), cd)
    W_rep = W_base.unsqueeze(0).expand(K, N, cd, cd).reshape(K * N, cd, cd)
    X_rep = Xc.expand(K * N, -1)
    a_rep = a_batch.repeat_interleave(N, dim=0)
    with torch.no_grad():
        pred = fn(W_rep, a_rep, X_rep)
    return pred[..., :out_d].view(K, N, out_d)


def eig_for_action(
    graph: Any,
    obs: dict[str, float],
    candidate_interventions: list[tuple[str, float]],
    *,
    ensemble: WeightedGraphEnsemble | None = None,
    return_best: bool = False,
) -> float | tuple[float, str | None, float]:
    """
    EIG for candidate interventions: posterior-weighted ensemble disagreement.
    return_best=True → (best_eig, best_var, best_val) for GoalImagination (one pass over candidates).
    """
    ens = ensemble or getattr(graph, "_ensemble", None)
    core = getattr(graph, "_core", None)
    if ens is None or core is None or not candidate_interventions:
        if return_best:
            return 0.0, None, 0.5
        return 0.0

    ids = list(getattr(graph, "_node_ids", []))
    if not ids:
        if return_best:
            return 0.0, None, 0.5
        return 0.0
    d = len(ids)
    out_d = int(getattr(graph, "_d", d))
    dev = getattr(graph, "device", torch.device("cpu"))
    weights = ens.posterior().detach()
    id_index = {nid: i for i, nid in enumerate(ids)}

    vec = [float(obs.get(nid, graph.nodes.get(nid, 0.5))) for nid in ids]
    X = torch.tensor([vec], dtype=torch.float32, device=dev)
    core_d = int(getattr(core, "d", d))
    Xc, _ = _pad_wm_inputs(X, torch.zeros(1, d, device=dev), core_d)

    valid: list[tuple[str, float]] = []
    for var, val in candidate_interventions:
        if var in id_index:
            valid.append((var, float(val)))

    best_eig = 0.0
    best_var: str | None = None
    best_val = 0.5

    use_batch = (
        _eig_batch_w_enabled()
        and hasattr(core, "forward_dynamics_batched_W")
        and len(valid) > 0
    )

    if use_batch and len(valid) > 1:
        a_batch = torch.zeros(len(valid), core_d, dtype=torch.float32, device=dev)
        for ki, (var, val) in enumerate(valid):
            a_batch[ki, id_index[var]] = val
        try:
            pred_kn = _forward_candidates_batched(core, ens, Xc, a_batch, out_d)
            for ki, (var, val) in enumerate(valid):
                eig = _ensemble_js_eig(pred_kn[ki], weights)
                if eig > best_eig:
                    best_eig = eig
                    best_var = var
                    best_val = val
        except Exception:
            use_batch = False

    if not use_batch or len(valid) <= 1:
        Wm = _pad_W_batch(ens.W_stack * ens.mask.unsqueeze(0), core_d)
        for var, val in valid:
            a = torch.zeros(1, core_d, dtype=torch.float32, device=dev)
            a[0, id_index[var]] = val
            if _eig_batch_w_enabled() and hasattr(core, "forward_dynamics_batched_W"):
                stack = _predict_ensemble_members(graph, core, ens, Xc, a, out_d)
            else:
                rows: list[torch.Tensor] = []
                for k in range(ens.n):
                    rows.append(
                        _align_step_pred(
                            _predict_with_W(graph, core, Wm[k], Xc, a)
                        ).squeeze(0)
                    )
                stack = torch.stack(rows, dim=0)
            eig = _ensemble_js_eig(stack, weights)
            if eig > best_eig:
                best_eig = eig
                best_var = var
                best_val = val

    if return_best:
        return best_eig, best_var, best_val
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
    out_d = int(getattr(graph, "_d", d))
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
    core_d = int(getattr(core, "d", d))
    a = torch.zeros(1, core_d, dtype=torch.float32, device=dev)
    a[0, idx] = float(val)
    Xc, ac = _pad_wm_inputs(X, a, core_d)

    with torch.no_grad():
        pred_n = _predict_ensemble_members(graph, core, ens, Xc, ac, out_d)
    delta = (Y - X).abs()
    w_dim = delta / delta.sum().clamp(min=1e-6)
    err = (pred_n.unsqueeze(1) - Y).abs()
    mse = (err * w_dim).sum(dim=(1, 2))
    ll = -mse * 50.0
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
