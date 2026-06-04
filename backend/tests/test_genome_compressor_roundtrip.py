"""Genome compressor roundtrip tests."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from engine.genome.compressor import (
    compress_adjacency,
    compress_adjacency_role_subgraph,
    load_compressed_genome,
    save_compressed_genome,
)
from engine.role_types import build_role_map


def test_genome_compressor_roundtrip():
    rng = np.random.default_rng(7)
    W = rng.normal(scale=0.1, size=(10, 10))
    W = np.triu(W, 1)
    result = compress_adjacency(W, rank=4)
    assert result["W_reconstructed"].shape == (10, 10)
    assert len(result["edge_list"]) >= 0

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "prior.npz"
        save_compressed_genome(result, path)
        loaded = load_compressed_genome(path)
        assert loaded["d"] == 10
        assert loaded["rank"] == 4
        np.testing.assert_allclose(
            loaded["W_reconstructed"], result["W_reconstructed"], atol=1e-5
        )


def test_genome_role_subgraph_compress_roundtrip():
    node_ids = [f"n{i}" for i in range(8)]
    role_map = {nid: "proprioceptive" for nid in node_ids[:4]}
    role_map.update({nid: "intent" for nid in node_ids[4:6]})
    role_map.update({nid: "posture" for nid in node_ids[6:]})
    rng = np.random.default_rng(11)
    W = rng.normal(scale=0.1, size=(8, 8))
    result = compress_adjacency_role_subgraph(W, node_ids, role_map, rank=3)
    assert result.get("role_subgraph") is not None
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "role.npz"
        save_compressed_genome(result, path)
        loaded = load_compressed_genome(path)
        assert loaded["role_subgraph"]
        assert len(loaded["node_ids"]) == result["d"]
