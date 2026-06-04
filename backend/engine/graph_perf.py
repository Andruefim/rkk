"""Guards for UI/live ticks when vision/concepts inflate graph size (d≫72)."""
from __future__ import annotations

import os
from typing import Any


def large_graph_node_cap() -> int:
    try:
        return max(48, int(os.environ.get("RKK_LARGE_GRAPH_NODE_CAP", "96")))
    except ValueError:
        return 96


def graph_active_node_count(graph: Any) -> int:
    nids = getattr(graph, "_node_ids", None)
    if nids is not None:
        return len(nids)
    nodes = getattr(graph, "nodes", None)
    return len(nodes) if nodes is not None else 0


def is_large_graph(graph: Any) -> bool:
    return graph_active_node_count(graph) > large_graph_node_cap()


def snapshot_edges_max_for_graph(graph: Any) -> int:
    """Cap WS edge sampling — full W probe on d=256 is what freezes tick ~49."""
    try:
        base = int(os.environ.get("RKK_SNAPSHOT_EDGES_MAX", "512"))
    except ValueError:
        base = 512
    if base <= 0:
        return 0
    n = graph_active_node_count(graph)
    cap = large_graph_node_cap()
    if n <= cap:
        return base
    try:
        heavy_lim = int(os.environ.get("RKK_SNAPSHOT_EDGES_MAX_LARGE", "0"))
    except ValueError:
        heavy_lim = 0
    return max(0, heavy_lim)


def temporal_rebuild_min_interval() -> int:
    try:
        return max(1, int(os.environ.get("RKK_TEMPORAL_REBUILD_EVERY", "8")))
    except ValueError:
        return 8
