"""
Track E: abstract causal skeleton (CMI topology) and cross-env transfer without W weights.

Skeleton captures adjacency + scale structure + feedback motifs; transfer seeds only
topology into target W (fixed small edge prior), not source weight magnitudes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import torch

from engine.genome.spectral import (
    CARTPOLE_VARIABLE_IDS,
    GRID_CONTROL_VARIABLE_IDS,
    GRID_NAV_VARIABLE_IDS,
    SYMBOLIC_CONTROL_VARIABLE_IDS,
)


def _env_flag(key: str, default: bool = False) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


def skeleton_cmi_thresh() -> float:
    return _env_float("RKK_SKELETON_CMI_THRESH", 0.12)


def skeleton_transfer_enabled() -> bool:
    return _env_flag("RKK_SKELETON_TRANSFER_ENABLED", False)


def skeleton_min_motif_match() -> float:
    return _env_float("RKK_SKELETON_MIN_MOTIF_MATCH", 0.40)


SKELETON_EDGE_PRIOR: float = 0.15


@dataclass
class CausalSkeleton:
    adjacency: np.ndarray
    scale_structure: str  # "hierarchical" | "feedback"
    feedback_loops: list[tuple[int, int]] = field(default_factory=list)
    node_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "scale_structure": self.scale_structure,
            "feedback_loops": list(self.feedback_loops),
            "n_nodes": int(self.adjacency.shape[0]),
            "n_edges": int((self.adjacency > 0.5).sum()),
            "node_ids": list(self.node_ids),
        }


def _observations_matrix(
    obs_data: Iterable[dict[str, float] | list[float]],
    node_ids: list[str],
) -> np.ndarray:
    rows: list[list[float]] = []
    for item in obs_data:
        if isinstance(item, dict):
            rows.append([float(item.get(nid, 0.5)) for nid in node_ids])
        else:
            vec = list(item)
            if len(vec) >= len(node_ids):
                rows.append([float(x) for x in vec[: len(node_ids)]])
    if not rows:
        return np.zeros((0, len(node_ids)), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def _gaussian_cmi_proxy_matrix(X: np.ndarray) -> np.ndarray:
    """
    Pairwise dependence proxy from observations (n, d).
    Uses squared partial correlation magnitude as CMI surrogate for skeleton edges.
    """
    n, d = X.shape
    if n < 6 or d < 2:
        return np.zeros((d, d), dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    std = Xc.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    Xc /= std
    corr = np.corrcoef(Xc.T)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    cmi = np.abs(corr)
    np.fill_diagonal(cmi, 0.0)
    return cmi


def _detect_scale_structure(adj: np.ndarray) -> str:
    d = adj.shape[0]
    if d < 2:
        return "hierarchical"
    upper = float(np.triu(adj, k=1).sum())
    lower = float(np.tril(adj, k=-1).sum())
    total = upper + lower + 1e-9
    if upper / total >= 0.62:
        return "hierarchical"
    if lower / total >= 0.62:
        return "hierarchical"
    return "feedback"


def _find_feedback_loops(adj: np.ndarray) -> list[tuple[int, int]]:
    loops: list[tuple[int, int]] = []
    d = adj.shape[0]
    for i in range(d):
        for j in range(i + 1, d):
            if adj[i, j] > 0.5 and adj[j, i] > 0.5:
                loops.append((i, j))
    return loops


def extract_causal_skeleton(
    W: np.ndarray | torch.Tensor,
    obs_data: Iterable[dict[str, float] | list[float]],
    role_map: dict[str, str] | None = None,
    *,
    node_ids: list[str] | None = None,
    cmi_thresh: float | None = None,
) -> CausalSkeleton:
    """
    Build binary adjacency from observation CMI (not from |W| magnitudes).
    ``W`` supplies node ordering only when ``node_ids`` omitted.
    """
    thresh = cmi_thresh if cmi_thresh is not None else skeleton_cmi_thresh()
    W_np = np.asarray(
        W.detach().cpu().numpy() if isinstance(W, torch.Tensor) else W,
        dtype=np.float64,
    )
    if node_ids is None:
        d = W_np.shape[0]
        node_ids = [f"v{i}" for i in range(d)]
    node_ids = list(node_ids)
    d = len(node_ids)

    if role_map:
        keep = [i for i, nid in enumerate(node_ids) if role_map.get(nid, "concept") != "concept"]
        if len(keep) >= 2:
            node_ids = [node_ids[i] for i in keep]
            d = len(node_ids)

    X = _observations_matrix(obs_data, node_ids)
    if X.shape[0] < 4:
        # Fallback: weak topology hint from W sign pattern only (binary, not weights).
        W_use = W_np[:d, :d] if W_np.shape[0] >= d else np.zeros((d, d))
        cmi = (np.abs(W_use) > thresh).astype(np.float64)
        np.fill_diagonal(cmi, 0.0)
    else:
        cmi = _gaussian_cmi_proxy_matrix(X)

    adj = (cmi >= thresh).astype(np.float64)
    np.fill_diagonal(adj, 0.0)
    scale = _detect_scale_structure(adj)
    loops = _find_feedback_loops(adj)
    return CausalSkeleton(
        adjacency=adj,
        scale_structure=scale,
        feedback_loops=loops,
        node_ids=node_ids,
    )


def skeleton_similarity(sk_a: CausalSkeleton, sk_b: CausalSkeleton) -> float:
    """Jaccard overlap on directed edges + scale-structure bonus."""
    A = sk_a.adjacency > 0.5
    B = sk_b.adjacency > 0.5
    if A.shape != B.shape:
        m = min(A.shape[0], B.shape[0])
        A = A[:m, :m]
        B = B[:m, :m]
    inter = float(np.logical_and(A, B).sum())
    union = float(np.logical_or(A, B).sum()) + 1e-9
    jacc = inter / union
    bonus = 0.1 if sk_a.scale_structure == sk_b.scale_structure else 0.0
    loop_bonus = 0.05 if sk_a.feedback_loops and sk_b.feedback_loops else 0.0
    return min(1.0, jacc + bonus + loop_bonus)


def _env_variable_ids(env: str | Any) -> list[str]:
    if hasattr(env, "variable_ids"):
        return list(env.variable_ids)
    preset = str(env) if isinstance(env, str) else str(getattr(env, "preset", ""))
    if preset in ("cartpole",):
        return list(CARTPOLE_VARIABLE_IDS)
    if preset in ("grid_nav", "grid_control"):
        return list(GRID_NAV_VARIABLE_IDS)
    if preset in ("symbolic_control",):
        return list(SYMBOLIC_CONTROL_VARIABLE_IDS)
    return list(GRID_CONTROL_VARIABLE_IDS)


def _motif_chain_indices(n: int, structure: str) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for i in range(n - 1):
        edges.append((i, i + 1))
    if structure == "feedback" and n >= 3:
        edges.append((n - 1, 0))
    return edges


def match_motifs(sk_ref: CausalSkeleton, env_type: str) -> dict[int, int]:
    """
    Map reference skeleton node indices → target env indices by motif shape.
    """
    tgt_ids = _env_variable_ids(env_type)
    n_tgt = len(tgt_ids)
    n_ref = sk_ref.adjacency.shape[0]
    n = min(n_ref, n_tgt)
    if n < 2:
        return {}
    mapping: dict[int, int] = {i: i for i in range(n)}
    if env_type == "cartpole" and n_ref >= 4 and n_tgt >= 4:
        # Prefer pole→cart hierarchy on cartpole indices.
        ref_order = list(range(n))
        tgt_order = [tgt_ids.index(v) for v in CARTPOLE_VARIABLE_IDS if v in tgt_ids][:n]
        mapping = {ref_order[i]: tgt_order[i] for i in range(min(len(ref_order), len(tgt_order)))}
    return mapping


def seed_W_from_motif(
    W_init: np.ndarray | torch.Tensor,
    motif_map: dict[int, int],
    sk_ref: CausalSkeleton,
    *,
    edge_prior: float = SKELETON_EDGE_PRIOR,
) -> torch.Tensor:
    """Write topology-only prior into W_init (does not copy source weight magnitudes)."""
    if isinstance(W_init, torch.Tensor):
        W = W_init.clone().float()
    else:
        W = torch.from_numpy(np.asarray(W_init, dtype=np.float32)).clone()
    d = W.shape[0]
    for i, j in zip(*np.where(sk_ref.adjacency > 0.5)):
        ti, tj = motif_map.get(int(i)), motif_map.get(int(j))
        if ti is None or tj is None:
            continue
        if ti >= d or tj >= d or ti == tj:
            continue
        W[ti, tj] = edge_prior
    for a, b in sk_ref.feedback_loops:
        ta, tb = motif_map.get(a), motif_map.get(b)
        if ta is not None and tb is not None and ta < d and tb < d:
            W[tb, ta] = 0.5 * edge_prior
    return W


def _W_for_env(
    W_init: np.ndarray | torch.Tensor,
    env_type: str,
) -> torch.Tensor:
    tgt_ids = _env_variable_ids(env_type)
    d = len(tgt_ids)
    if isinstance(W_init, torch.Tensor):
        W = torch.zeros(d, d, dtype=torch.float32)
    else:
        W = torch.zeros(d, d, dtype=torch.float32)
    w0 = np.asarray(
        W_init.detach().cpu().numpy() if isinstance(W_init, torch.Tensor) else W_init,
        dtype=np.float32,
    )
    dd = min(d, w0.shape[0]) if w0.ndim == 2 else 0
    if dd > 0:
        W[:dd, :dd] = torch.from_numpy(w0[:dd, :dd])
    return W


def transfer_skeleton_to_env(
    sk_ref: CausalSkeleton,
    W_init: np.ndarray | torch.Tensor,
    env: str | Any,
    *,
    force: bool = False,
) -> torch.Tensor:
    """Seed target W with skeleton topology for ``env`` (cartpole / grid_nav / grid_control)."""
    env_type = env if isinstance(env, str) else str(getattr(env, "preset", env))
    W = _W_for_env(W_init, env_type)
    if not skeleton_transfer_enabled() and not force:
        return W
    motif_map = match_motifs(sk_ref, env_type)
    if not motif_map:
        n = min(sk_ref.adjacency.shape[0], W.shape[0])
        motif_map = {i: i for i in range(n)}
    return seed_W_from_motif(W, motif_map, sk_ref)


def transfer_skeleton_nonphys(
    sk_ref: CausalSkeleton,
    W_init: np.ndarray | torch.Tensor,
    env_type: str,
    role_discovery_map: dict[str, str] | None = None,
) -> torch.Tensor:
    """
    Track H2: skeleton transfer to non-physical stubs using role-discovery node order.
    """
    _ = role_discovery_map  # reserved for finer mapping in Phase 6a
    return transfer_skeleton_to_env(sk_ref, W_init, env_type, force=True)


def apply_skeleton_to_graph(
    graph: Any,
    sk_ref: CausalSkeleton,
    *,
    env_target: str = "cartpole",
    force: bool = False,
) -> dict[str, Any]:
    """Apply skeleton transfer to live causal graph edges."""
    from engine.genome.spectral import graph_adjacency_numpy

    W_np, ids = graph_adjacency_numpy(graph)
    tgt_ids = _env_variable_ids(env_target)
    for nid in tgt_ids:
        if nid not in graph._node_ids:
            graph.set_node(nid, 0.5)
    d = len(graph._node_ids)
    W_init = np.zeros((d, d), dtype=np.float32)
    W_t = transfer_skeleton_to_env(sk_ref, W_init, env_target, force=force)
    W_out = W_t.detach().cpu().numpy()
    id_to_i = {nid: i for i, nid in enumerate(graph._node_ids)}
    count = 0
    for i in range(d):
        for j in range(d):
            w = float(W_out[i, j])
            if abs(w) < 1e-5:
                continue
            fr = graph._node_ids[i]
            to = graph._node_ids[j]
            graph.set_edge(fr, to, w, alpha=0.7)
            count += 1
    sim_hint = skeleton_min_motif_match()
    return {
        "edges_set": count,
        "env_target": env_target,
        "motif_match_min": sim_hint,
        "skeleton_edges": int((sk_ref.adjacency > 0.5).sum()),
    }


def extract_skeleton_from_graph(
    graph: Any,
    *,
    role_map: dict[str, str] | None = None,
) -> CausalSkeleton:
    """Build skeleton from graph obs buffer + node order."""
    from engine.genome.spectral import graph_adjacency_numpy

    W, ids = graph_adjacency_numpy(graph)
    obs_rows = getattr(graph, "_obs_buffer", []) or []
    if role_map is None and hasattr(graph, "role_type_map"):
        try:
            role_map = graph.role_type_map()
        except Exception:
            role_map = None
    return extract_causal_skeleton(W, obs_rows, role_map, node_ids=ids)
