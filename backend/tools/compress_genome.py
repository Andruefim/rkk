#!/usr/bin/env python3
"""Offline tool: compress logged W matrices into genome/compressed_prior.npz."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from engine.genome.compressor import (
    aggregate_matrices,
    compress_adjacency,
    genome_rank,
    save_compressed_genome,
)


def _load_w_from_log(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if "W" in obj:
            return np.array(obj["W"], dtype=np.float64)
        if "wm" in obj:
            return np.array(obj["wm"], dtype=np.float64)
    raise ValueError(f"Unsupported log format: {path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Compress causal W into low-rank genome prior")
    p.add_argument(
        "inputs",
        nargs="+",
        help="Paths to .npy/.json W snapshots (flat, slope, stairs, ...)",
    )
    p.add_argument(
        "-o",
        "--output",
        default=str(ROOT / "backend" / "engine" / "genome" / "compressed_prior.npz"),
    )
    p.add_argument("-k", "--rank", type=int, default=None)
    args = p.parse_args()

    mats = [_load_w_from_log(Path(x)) for x in args.inputs]
    W_mean = aggregate_matrices(mats)
    result = compress_adjacency(W_mean, rank=args.rank or genome_rank())
    out = save_compressed_genome(result, args.output)
    print(f"Saved compressed genome: {out} (rank={result['rank']}, edges={len(result['edge_list'])})")


if __name__ == "__main__":
    main()
