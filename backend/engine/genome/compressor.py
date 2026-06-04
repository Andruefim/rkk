"""
genome/compressor.py — Offline low-rank compression of learned W matrices (Phase 3 / Track B2).

Role-typed subgraph: compress only transferable roles (motor, posture, contact,
proprioceptive, intent) for cross-world init with the same variable_ids.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np

from engine.role_types import TRANSFER_ROLE_TYPES, build_role_map


def genome_rank() -> int:
    try:
        return max(2, min(64, int(os.environ.get("RKK_GENOME_RANK", "8"))))
    except ValueError:
        return 8


def genome_min_worlds() -> int:
    try:
        return max(1, int(os.environ.get("RKK_GENOME_MIN_WORLDS", "2")))
    except ValueError:
        return 2


def role_subgraph_indices(
    node_ids: list[str],
    role_map: dict[str, str],
    *,
    roles: frozenset[str] | None = None,
) -> list[int]:
    """Indices into full W for nodes whose role is in ``roles`` (default TRANSFER_ROLE_TYPES)."""
    use_roles = roles if roles is not None else TRANSFER_ROLE_TYPES
    return [i for i, nid in enumerate(node_ids) if role_map.get(nid, "") in use_roles]


def extract_role_submatrix(
    W: np.ndarray,
    node_ids: list[str],
    role_map: dict[str, str] | None = None,
) -> tuple[np.ndarray, list[int], list[str]]:
    """W_sub on role-typed nodes only; returns (W_sub, full_indices, sub_ids)."""
    role_map = role_map or build_role_map(node_ids)
    idx = role_subgraph_indices(node_ids, role_map)
    if not idx:
        raise ValueError("role subgraph is empty — check role_map and TRANSFER_ROLE_TYPES")
    sub_ids = [node_ids[i] for i in idx]
    W_sub = np.asarray(W, dtype=np.float64)[np.ix_(idx, idx)]
    return W_sub, idx, sub_ids


def compress_adjacency(W: np.ndarray, rank: int | None = None) -> dict:
    """
    SVD low-rank factorization W ≈ U V^T, reconstruct sparse adjacency.
    Returns dict with U, V, W_reconstructed, edge_list.
    """
    k = rank if rank is not None else genome_rank()
    W = np.asarray(W, dtype=np.float64)
    d = W.shape[0]
    np.fill_diagonal(W, 0.0)
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    k = min(k, len(s))
    Uk = U[:, :k] * np.sqrt(s[:k])
    Vk = Vt[:k, :].T * np.sqrt(s[:k])
    W_rec = Uk @ Vk.T
    threshold = float(os.environ.get("RKK_GENOME_EDGE_THRESH", "0.05"))
    edges: list[tuple[int, int, float]] = []
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            w = float(W_rec[i, j])
            if abs(w) >= threshold:
                edges.append((i, j, w))
    return {
        "U": Uk.astype(np.float32),
        "V": Vk.astype(np.float32),
        "W_reconstructed": W_rec.astype(np.float32),
        "edge_list": edges,
        "rank": k,
        "d": d,
    }


def compress_adjacency_role_subgraph(
    W: np.ndarray,
    node_ids: list[str],
    role_map: dict[str, str] | None = None,
    rank: int | None = None,
) -> dict:
    """Low-rank compression on role-typed subgraph; stores mapping back to full graph."""
    W_sub, full_idx, sub_ids = extract_role_submatrix(W, node_ids, role_map)
    role_map = role_map or build_role_map(node_ids)
    sub = compress_adjacency(W_sub, rank=rank)
    sub["node_ids"] = np.asarray(sub_ids, dtype="U64")
    sub["full_indices"] = np.array(full_idx, dtype=np.int32)
    sub["role_types"] = np.array(
        [role_map.get(nid, "") for nid in sub_ids], dtype="U16"
    )
    sub["role_subgraph"] = np.array([1], dtype=np.int8)
    sub["d_full"] = np.array([len(node_ids)])
    return sub


def aggregate_matrices(matrices: Iterable[np.ndarray]) -> np.ndarray:
    """Mean of multiple W snapshots (e.g. flat/slope/stairs presets)."""
    mats = [np.asarray(m, dtype=np.float64) for m in matrices]
    if not mats:
        raise ValueError("No matrices to aggregate")
    stack = np.stack(mats, axis=0)
    return stack.mean(axis=0)


def save_compressed_genome(result: dict, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "U": result["U"],
        "V": result["V"],
        "W_reconstructed": result["W_reconstructed"],
        "edge_list": np.array(result["edge_list"], dtype=np.float32),
        "rank": np.array([result["rank"]]),
        "d": np.array([result["d"]]),
    }
    if "node_ids" in result:
        payload["node_ids"] = result["node_ids"]
        payload["full_indices"] = result["full_indices"]
        payload["role_types"] = result["role_types"]
        payload["role_subgraph"] = result["role_subgraph"]
        payload["d_full"] = result["d_full"]
    np.savez_compressed(out, **payload)
    return out


def load_compressed_genome(path: str | Path) -> dict:
    data = np.load(path, allow_pickle=False)
    edge_raw = data["edge_list"]
    edges = [(int(r[0]), int(r[1]), float(r[2])) for r in edge_raw]
    out: dict = {
        "U": data["U"],
        "V": data["V"],
        "W_reconstructed": data["W_reconstructed"],
        "edge_list": edges,
        "rank": int(data["rank"][0]),
        "d": int(data["d"][0]),
    }
    if "role_subgraph" in data:
        out["role_subgraph"] = True
        out["node_ids"] = [str(x).strip() for x in np.atleast_1d(data["node_ids"])]
        out["full_indices"] = [int(x) for x in data["full_indices"]]
        out["role_types"] = [str(x) for x in data["role_types"]]
        out["d_full"] = int(data["d_full"][0])
    return out


def apply_role_subgraph_to_graph(graph, data: dict, *, alpha: float = 0.75) -> int:
    """
    Cross-world init: map compressed role-subgraph edges onto graph by node id.
    Same topology (humanoid ↔ humanoid_variant) → identical variable_ids.
    """
    if not data.get("role_subgraph"):
        return 0
    raw_ids = data.get("node_ids")
    if raw_ids is None:
        return 0
    sub_ids = [str(x).strip() for x in np.atleast_1d(raw_ids)]
    if not sub_ids:
        return 0
    id_to_idx = {nid: i for i, nid in enumerate(graph._node_ids)}
    sub_to_full: list[int | None] = [id_to_idx.get(nid) for nid in sub_ids]
    W_rec = np.asarray(data["W_reconstructed"], dtype=np.float64)
    threshold = float(os.environ.get("RKK_GENOME_EDGE_THRESH", "0.05"))
    count = 0
    d_sub = W_rec.shape[0]
    for i in range(d_sub):
        fi = sub_to_full[i]
        if fi is None:
            continue
        for j in range(d_sub):
            fj = sub_to_full[j]
            if fj is None or i == j:
                continue
            w = float(W_rec[i, j])
            if abs(w) < threshold:
                continue
            fr = graph._node_ids[fi]
            to = graph._node_ids[fj]
            graph.set_edge(fr, to, w, alpha=alpha)
            count += 1
    return count
