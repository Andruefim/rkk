"""Genome compressor roundtrip tests."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from engine.genome.compressor import (
    compress_adjacency,
    load_compressed_genome,
    save_compressed_genome,
)


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
