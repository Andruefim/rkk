"""
genome/compressor.py — Offline low-rank compression of learned W matrices (Phase 3).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np


def genome_rank() -> int:
    try:
        return max(2, min(64, int(os.environ.get("RKK_GENOME_RANK", "8"))))
    except ValueError:
        return 8


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
    np.savez_compressed(
        out,
        U=result["U"],
        V=result["V"],
        W_reconstructed=result["W_reconstructed"],
        edge_list=np.array(result["edge_list"], dtype=np.float32),
        rank=np.array([result["rank"]]),
        d=np.array([result["d"]]),
    )
    return out


def load_compressed_genome(path: str | Path) -> dict:
    data = np.load(path, allow_pickle=False)
    edge_raw = data["edge_list"]
    edges = [(int(r[0]), int(r[1]), float(r[2])) for r in edge_raw]
    return {
        "U": data["U"],
        "V": data["V"],
        "W_reconstructed": data["W_reconstructed"],
        "edge_list": edges,
        "rank": int(data["rank"][0]),
        "d": int(data["d"][0]),
    }
