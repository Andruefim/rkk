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
    compress_adjacency_role_subgraph,
    genome_min_worlds,
    genome_rank,
    save_compressed_genome,
)
from engine.role_types import build_role_map


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
    p.add_argument(
        "--role-subgraph",
        action="store_true",
        help="Compress only transferable role-typed subgraph (Track B2)",
    )
    p.add_argument(
        "--node-ids",
        type=str,
        default="",
        help="Comma-separated node ids (default: humanoid VAR_NAMES)",
    )
    args = p.parse_args()

    mats = [_load_w_from_log(Path(x)) for x in args.inputs]
    if len(mats) < genome_min_worlds():
        raise SystemExit(
            f"need at least {genome_min_worlds()} W inputs (got {len(mats)})"
        )
    W_mean = aggregate_matrices(mats)
    if args.role_subgraph:
        from engine.role_types import humanoid_variable_ids_for_roles

        node_ids = [
            x.strip()
            for x in (args.node_ids or "").split(",")
            if x.strip()
        ] or humanoid_variable_ids_for_roles()
        role_map = build_role_map(node_ids)
        result = compress_adjacency_role_subgraph(
            W_mean[: len(node_ids), : len(node_ids)],
            node_ids,
            role_map,
            rank=args.rank or genome_rank(),
        )
    else:
        result = compress_adjacency(W_mean, rank=args.rank or genome_rank())
    out = save_compressed_genome(result, args.output)
    print(f"Saved compressed genome: {out} (rank={result['rank']}, edges={len(result['edge_list'])})")


if __name__ == "__main__":
    main()
