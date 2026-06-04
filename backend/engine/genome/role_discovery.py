"""
Track C6: spectral role discovery in unknown environments (no B0 role map).

Maps new-env nodes to promoted ``learned_*`` role types via spectral fingerprint
similarity to universal learned role signatures.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

from engine.genome.learned_roles import LearnedRoleEntry, learned_roles
from engine.genome.spectral import spectral_fingerprint, spectral_similarity
from engine.latent_confounder import signature_similarity


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


def c6_enabled() -> bool:
    return _env_flag("RKK_C6_ENABLED", False)


def role_discovery_thresh() -> float:
    return _env_float("RKK_ROLE_DISCOVERY_THRESH", 0.65)


def role_discovery_top_k() -> int:
    return max(1, _env_int("RKK_ROLE_DISCOVERY_TOP_K", 1))


def _local_subgraph_W(W: np.ndarray, center: int, *, radius: int = 1) -> np.ndarray:
    d = W.shape[0]
    nodes = {center}
    frontier = {center}
    for _ in range(radius):
        nxt: set[int] = set()
        for i in frontier:
            for j in range(d):
                if abs(W[i, j]) > 1e-5 or abs(W[j, i]) > 1e-5:
                    nxt.add(j)
        nodes |= nxt
        frontier = nxt
    idx = sorted(nodes)
    return W[np.ix_(idx, idx)]


def _fingerprint_for_node(W: np.ndarray, node_index: int, k: int) -> torch.Tensor:
    sub = _local_subgraph_W(W, node_index, radius=1)
    if sub.shape[0] < 2:
        sub = W
    return spectral_fingerprint(torch.from_numpy(sub), k=k)


def _signature_to_fingerprint(sig: list[float], k: int) -> torch.Tensor:
    """Embed flat signature into (d, k) fingerprint for spectral compare."""
    v = np.asarray(sig, dtype=np.float64)
    if v.size == 0:
        return torch.zeros(2, k)
    d = max(2, min(16, int(np.ceil(v.size / max(k, 1)))))
    mat = np.zeros((d, k), dtype=np.float64)
    flat = v[: d * k]
    if flat.size < d * k:
        flat = np.pad(flat, (0, d * k - flat.size))
    mat[:, :] = flat.reshape(d, k)
    return torch.from_numpy(mat.astype(np.float32))


def discover_roles_in_new_env(
    graph: Any,
    *,
    W: np.ndarray | None = None,
    node_ids: list[str] | None = None,
    learned: dict[str, LearnedRoleEntry] | None = None,
    force: bool = False,
) -> dict[str, str]:
    """
    Assign ``learned_*`` role types to target nodes by spectral similarity.

    Returns ``{node_id: learned_role_type}`` for assignments above
    ``RKK_ROLE_DISCOVERY_THRESH``. Updates graph node meta when assignments exist.
    """
    if not c6_enabled() and not force:
        return {}
    registry = learned if learned is not None else learned_roles
    if not registry:
        return {}

    from engine.genome.spectral import graph_adjacency_numpy, spectral_k

    if W is None or node_ids is None:
        W, node_ids = graph_adjacency_numpy(graph)
    else:
        node_ids = list(node_ids)
        W = np.asarray(W, dtype=np.float64)

    k = spectral_k()
    thresh = role_discovery_thresh()
    top_k = role_discovery_top_k()
    assignments: dict[str, str] = {}
    used_nodes: set[str] = set()

    for entry in registry.values():
        if len(entry.worlds) < 1:
            continue
        F_ref = _signature_to_fingerprint(entry.signature, k)
        scores: list[tuple[float, str]] = []
        for i, nid in enumerate(node_ids):
            if nid in used_nodes:
                continue
            F_loc = _fingerprint_for_node(W, i, k)
            sim = spectral_similarity(F_loc, F_ref)
            sig_loc = F_loc.detach().cpu().numpy().reshape(-1)
            sig_ref = np.asarray(entry.signature, dtype=np.float64)
            blend = 0.65 * sim + 0.35 * signature_similarity(sig_loc, sig_ref)
            if blend >= thresh:
                scores.append((blend, nid))
        scores.sort(key=lambda x: -x[0])
        for _, nid in scores[:top_k]:
            assignments[nid] = entry.role_type
            used_nodes.add(nid)
            _tag_node_role(graph, nid, entry.role_type)

    return assignments


def _tag_node_role(graph: Any, node_id: str, role_type: str) -> None:
    meta = getattr(graph, "_node_meta", {}).get(node_id)
    if meta is not None:
        meta.role_type = role_type
    if hasattr(graph, "nodes") and node_id in graph.nodes:
        graph.nodes[node_id] = float(graph.nodes.get(node_id, 0.5))


def discovery_snapshot_fields(assignments: dict[str, str]) -> dict[str, Any]:
    return {
        "role_discovery_enabled": c6_enabled(),
        "role_discovery_assignments": dict(assignments),
        "role_discovery_count": len(assignments),
        "role_discovery_learned_roles": len(learned_roles),
    }
