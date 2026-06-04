"""
Track B4: spectral graph fingerprint + orthogonal Procrustes alignment (humanoid → cartpole).

Cross-topology transfer seeds target W from aligned eigenstructure of a reference subgraph,
without requiring B0 role maps on the target env.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes

from engine.role_types import TRANSFER_ROLE_TYPES, build_role_map


def _env_flag(key: str, default: bool = False) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


def spectral_k() -> int:
    return max(2, min(32, _env_int("RKK_SPECTRAL_K", 8)))


def spectral_align_thresh() -> float:
    return _env_float("RKK_SPECTRAL_ALIGN_THRESH", 0.55)


def spectral_transfer_enabled() -> bool:
    return _env_flag("RKK_SPECTRAL_TRANSFER_ENABLED", False)


# Canonical variable ids for cross-topology stubs (Phase 4 cartpole).
CARTPOLE_VARIABLE_IDS: tuple[str, ...] = (
    "cart_pos",
    "cart_vel",
    "pole_angle",
    "pole_vel",
    "pole_angular_vel",
    "action_force",
    "upright",
    "balance_stability",
)

# Phase 4 gate alias + Track H grid_nav stub.
GRID_NAV_VARIABLE_IDS: tuple[str, ...] = (
    "pos_x",
    "pos_y",
    "goal_x",
    "goal_y",
    "action_dir",
)

GRID_CONTROL_VARIABLE_IDS = GRID_NAV_VARIABLE_IDS

# Track H1 symbolic_control stub.
SYMBOLIC_CONTROL_VARIABLE_IDS: tuple[str, ...] = (
    "rule_0",
    "rule_1",
    "rule_2",
    "rule_3",
    "action_select",
)


def spectral_fingerprint(W_subgraph: torch.Tensor, k: int | None = None) -> torch.Tensor:
    """
    Top-k eigenvectors of W W^T (graph Laplacian-style spectral signature).
    Returns (d, k) with columns ordered by descending eigenvalue.
    """
    k = k if k is not None else spectral_k()
    W = W_subgraph.detach().float()
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"W_subgraph must be square 2D, got {tuple(W.shape)}")
    d = W.shape[0]
    k = max(1, min(k, d))
    gram = W @ W.T
    gram = 0.5 * (gram + gram.T)
    vals, vecs = torch.linalg.eigh(gram)
    return vecs[:, -k:].contiguous()


def _pad_rows(M: np.ndarray, rows: int) -> np.ndarray:
    if M.shape[0] >= rows:
        return M[:rows, :]
    pad = np.zeros((rows - M.shape[0], M.shape[1]), dtype=M.dtype)
    return np.vstack([M, pad])


def procrustes_align(F_new: torch.Tensor, F_ref: torch.Tensor) -> torch.Tensor:
    """
    Orthogonal map R (k×k) aligning spectral subspaces: F_new @ R ≈ F_ref (same k).
    scipy.linalg.orthogonal_procrustes on (d×k) embeddings with row padding when d differs.
    """
    A = F_new.detach().float().cpu().numpy()
    B = F_ref.detach().float().cpu().numpy()
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("fingerprints must be 2D (d, k)")
    k = min(A.shape[1], B.shape[1])
    if k < 1:
        return torch.eye(1, dtype=torch.float32)
    rows = max(A.shape[0], B.shape[0])
    A_k = _pad_rows(A[:, :k], rows)
    B_k = _pad_rows(B[:, :k], rows)
    R_np, _ = orthogonal_procrustes(A_k, B_k)
    return torch.from_numpy(R_np.astype(np.float32))


def spectral_similarity(F_new: torch.Tensor, F_ref: torch.Tensor) -> float:
    """Cosine similarity after Procrustes alignment in spectral coefficient space."""
    R = procrustes_align(F_new, F_ref)
    k = min(F_new.shape[1], F_ref.shape[1], R.shape[0])
    if k < 1:
        return 0.0
    A = F_new[:, :k].float()
    B = F_ref[:, :k].float()
    aligned = A @ R[:k, :k].to(A.device)
    rows = min(aligned.shape[0], B.shape[0])
    a = aligned[:rows].reshape(-1)
    b = B[:rows].reshape(-1)
    na, nb = torch.linalg.norm(a), torch.linalg.norm(b)
    if float(na) < 1e-9 or float(nb) < 1e-9:
        return 0.0
    return float((a @ b) / (na * nb))


def _chain_adjacency(n: int) -> np.ndarray:
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 1):
        A[i, i + 1] = 1.0
    return A


def _adjacency_to_W(A: np.ndarray, *, scale: float = 0.18) -> np.ndarray:
    d = A.shape[0]
    W = np.zeros((d, d), dtype=np.float64)
    for i in range(d):
        for j in range(d):
            if i != j and A[i, j] > 0.5:
                W[i, j] = scale * float(A[i, j])
    return W


def _reference_subgraph_W(
    W: np.ndarray,
    node_ids: list[str],
    *,
    env_preset: str = "humanoid",
    use_roles: bool = True,
) -> tuple[np.ndarray, list[str]]:
    W = np.asarray(W, dtype=np.float64)
    if not use_roles:
        return W, list(node_ids)
    role_map = build_role_map(node_ids, env_preset=env_preset)
    idx = [i for i, nid in enumerate(node_ids) if role_map.get(nid, "") in TRANSFER_ROLE_TYPES]
    if len(idx) < 3:
        return W, list(node_ids)
    sub_ids = [node_ids[i] for i in idx]
    W_sub = W[np.ix_(idx, idx)]
    return W_sub, sub_ids


def transfer_W_spectral(
    W_ref: np.ndarray,
    node_ids_ref: list[str],
    node_ids_target: list[str],
    *,
    env_ref: str = "humanoid",
    env_target: str = "cartpole",
    k: int | None = None,
    align_thresh: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Cross-topology init: align reference spectral fingerprint to a target chain template.

    Returns (W_target, meta) with keys: similarity, accepted, env_ref, env_target.
    """
    k = k if k is not None else spectral_k()
    thresh = align_thresh if align_thresh is not None else spectral_align_thresh()
    W_sub, _ = _reference_subgraph_W(
        W_ref, node_ids_ref, env_preset=env_ref, use_roles=(env_ref in ("humanoid", "humanoid_variant"))
    )
    F_ref = spectral_fingerprint(torch.from_numpy(W_sub), k=k)

    d_tgt = len(node_ids_target)
    if d_tgt < 2:
        return np.zeros((max(d_tgt, 1), max(d_tgt, 1))), {"similarity": 0.0, "accepted": False}

    # Target topology template: directed chain (cartpole pole→cart; grid: pos→goal).
    if env_target in ("cartpole",):
        order = list(node_ids_target)
    elif env_target in ("grid_nav", "grid_control"):
        preferred = [v for v in GRID_NAV_VARIABLE_IDS if v in node_ids_target]
        order = preferred or list(node_ids_target)
    else:
        order = list(node_ids_target)

    A_tpl = _chain_adjacency(len(order))
    W_tpl = _adjacency_to_W(A_tpl)
    F_tpl = spectral_fingerprint(torch.from_numpy(W_tpl), k=k)
    sim = spectral_similarity(F_tpl, F_ref)
    accepted = sim >= thresh

    W_out = np.zeros((d_tgt, d_tgt), dtype=np.float64)
    id_to_i = {nid: i for i, nid in enumerate(node_ids_target)}
    edge_scale = 0.22 * max(sim, 0.15) if accepted else 0.08
    for i in range(len(order) - 1):
        a, b = order[i], order[i + 1]
        ia, ib = id_to_i.get(a), id_to_i.get(b)
        if ia is None or ib is None:
            continue
        W_out[ia, ib] = edge_scale
        if env_target == "cartpole" and b in ("cart_vel", "pole_vel"):
            W_out[ib, ia] = 0.5 * edge_scale  # weak feedback loop

    meta = {
        "similarity": round(sim, 4),
        "accepted": accepted,
        "env_ref": env_ref,
        "env_target": env_target,
        "spectral_k": k,
        "align_thresh": thresh,
        "n_target": d_tgt,
        "n_ref_sub": int(W_sub.shape[0]),
    }
    return W_out, meta


def apply_spectral_transfer_to_graph(
    graph: Any,
    W_ref: np.ndarray,
    node_ids_ref: list[str],
    node_ids_target: list[str] | None = None,
    *,
    env_ref: str = "humanoid",
    env_target: str = "cartpole",
    alpha: float = 0.72,
    force: bool = False,
) -> dict[str, Any]:
    """
    Seed edges on ``graph`` for target variable ids via spectral transfer.
    No-op when RKK_SPECTRAL_TRANSFER_ENABLED=0 unless ``force``.
    """
    if not spectral_transfer_enabled() and not force:
        return {"edges_set": 0, "skipped": True}
    tgt = list(node_ids_target or graph._node_ids)
    W_tgt, meta = transfer_W_spectral(
        W_ref, node_ids_ref, tgt, env_ref=env_ref, env_target=env_target
    )
    id_to_nid = {i: nid for i, nid in enumerate(tgt)}
    count = 0
    d = W_tgt.shape[0]
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            w = float(W_tgt[i, j])
            if abs(w) < 1e-4:
                continue
            fr, to = id_to_nid.get(i), id_to_nid.get(j)
            if fr is None or to is None:
                continue
            if fr not in graph._node_ids:
                graph.set_node(fr, 0.5)
            if to not in graph._node_ids:
                graph.set_node(to, 0.5)
            graph.set_edge(fr, to, w, alpha=alpha)
            count += 1
    meta["edges_set"] = count
    meta["skipped"] = False
    return meta


def graph_adjacency_numpy(graph: Any) -> tuple[np.ndarray, list[str]]:
    """Dense W from causal graph core (active nodes only)."""
    ids = list(graph._node_ids)
    d = len(ids)
    W = np.zeros((d, d), dtype=np.float64)
    if d == 0:
        return W, ids
    core = getattr(graph, "_core", None)
    if core is not None and hasattr(core, "W_masked"):
        wm = core.W_masked().detach().cpu().numpy()
        dd = min(d, wm.shape[0])
        W[:dd, :dd] = wm[:dd, :dd]
    else:
        for e in getattr(graph, "edges", {}).values():
            fr, to = e.from_, e.to
            if fr in ids and to in ids:
                W[ids.index(fr), ids.index(to)] = float(e.weight)
    return W, ids


def humanoid_to_cartpole_transfer(
    graph: Any,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Convenience: snapshot humanoid W → spectral seed cartpole nodes on ``graph``."""
    W_ref, ref_ids = graph_adjacency_numpy(graph)
    for nid in CARTPOLE_VARIABLE_IDS:
        if nid not in graph._node_ids:
            graph.set_node(nid, 0.5)
    if hasattr(graph, "_rebuild_core"):
        graph._rebuild_core()
    return apply_spectral_transfer_to_graph(
        graph,
        W_ref,
        ref_ids,
        list(CARTPOLE_VARIABLE_IDS),
        env_ref=str(getattr(graph, "_env_preset", "humanoid")),
        env_target="cartpole",
        force=force,
    )
