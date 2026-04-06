"""
Фаза 1: детектор повторяющихся подграфов (зачаток понятий).

Ищет ориентированные пути фиксированной длины в текущем срезе рёбер GNN,
группирует по «абстрактной» сигнатуре (l/r → общий префикс @ для конечностей),
отбирает сигнатуры с числом вхождений ≥ min_count и средним α по рёбрам ≥ порога.

Опционально: только при плато RSI (счётчик интервенций без роста discovery).
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.causal_graph import CausalGraph

# Суффиксы после l/r для нормализации симметричных цепочек
_LIMB_SUFFIXES = frozenset(
    {"hip", "knee", "ankle", "shoulder", "elbow", "foot_z"}
)


def _abstract_node(name: str) -> str:
    if name in ("lfoot_z", "rfoot_z"):
        return "@foot_z"
    if len(name) >= 2 and name[0] in ("l", "r"):
        rest = name[1:]
        if rest in _LIMB_SUFFIXES:
            return "@" + rest
    if name.startswith(("spine_", "neck_")):
        return "@" + name
    return name


def _edge_list_dicts(graph: CausalGraph) -> list[dict]:
    return [e.as_dict() for e in graph.edges]


def _build_adj(
    edges: list[dict], w_thresh: float
) -> dict[str, list[tuple[str, float, float]]]:
    adj: dict[str, list[tuple[str, float, float]]] = {}
    for e in edges:
        w = float(e.get("weight", 0))
        if abs(w) < w_thresh:
            continue
        f = e.get("from_") or e.get("from")
        t = e.get("to")
        if not f or not t:
            continue
        a = float(e.get("alpha_trust", 0))
        adj.setdefault(str(f), []).append((str(t), w, a))
    return adj


def _path_alpha_mean(adj: dict[str, list[tuple[str, float, float]]], path: list[str]) -> float:
    s = 0.0
    n = 0
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        for t, _w, al in adj.get(a, []):
            if t == b:
                s += al
                n += 1
                break
    return s / max(1, n)


def _enumerate_paths(
    adj: dict[str, list[tuple[str, float, float]]],
    path_edges: int,
    max_paths: int,
) -> list[list[str]]:
    out: list[list[str]] = []

    def dfs(node: str, chain: list[str], vis: frozenset[str]) -> None:
        if len(out) >= max_paths:
            return
        if len(chain) - 1 == path_edges:
            out.append(list(chain))
            return
        for nb, _w, _a in adj.get(node, []):
            if nb in vis:
                continue
            dfs(nb, chain + [nb], vis | {nb})

    for start in adj:
        if len(out) >= max_paths:
            break
        dfs(start, [start], frozenset({start}))
    return out


def detect_proto_concepts(
    graph: CausalGraph,
    *,
    path_edges: int | None = None,
    min_count: int | None = None,
    w_thresh: float | None = None,
    min_alpha_mean: float | None = None,
    agent_plateau_counter: int = 0,
    plateau_ticks_required: int | None = None,
) -> list[dict]:
    try:
        pe = int(os.environ.get("RKK_CONCEPT_PATH_EDGES", "3")) if path_edges is None else path_edges
    except ValueError:
        pe = 3
    try:
        mc = int(os.environ.get("RKK_CONCEPT_MIN_COUNT", "2")) if min_count is None else min_count
    except ValueError:
        mc = 2
    try:
        wt = float(os.environ.get("RKK_CONCEPT_W_THRESH", "0.05")) if w_thresh is None else w_thresh
    except ValueError:
        wt = 0.05
    try:
        amin = float(os.environ.get("RKK_CONCEPT_MIN_ALPHA", "0.12")) if min_alpha_mean is None else min_alpha_mean
    except ValueError:
        amin = 0.12
    try:
        ptk = int(os.environ.get("RKK_CONCEPT_PLATEAU_TICKS", "0")) if plateau_ticks_required is None else plateau_ticks_required
    except ValueError:
        ptk = 0
    try:
        max_paths = int(os.environ.get("RKK_CONCEPT_MAX_PATHS", "5000"))
    except ValueError:
        max_paths = 5000

    if ptk > 0 and agent_plateau_counter < ptk:
        return []

    edges = _edge_list_dicts(graph)
    adj = _build_adj(edges, wt)
    paths = _enumerate_paths(adj, pe, max_paths=max_paths)

    buckets: dict[tuple[str, ...], list[list[str]]] = {}
    for p in paths:
        sig = tuple(_abstract_node(x) for x in p)
        buckets.setdefault(sig, []).append(p)

    concepts: list[dict] = []
    for sig, plist in buckets.items():
        if len(plist) < mc:
            continue
        rep = plist[0]
        am = _path_alpha_mean(adj, rep)
        if am < amin:
            continue
        edge_pairs = [{"from_": rep[i], "to": rep[i + 1]} for i in range(len(rep) - 1)]
        concepts.append(
            {
                "pattern": list(sig),
                "pattern_nodes_example": rep,
                "uses": len(plist),
                "alpha_mean": round(am, 4),
                "edges": edge_pairs,
            }
        )

    concepts.sort(key=lambda x: (-x["uses"], -x["alpha_mean"]))
    for i, c in enumerate(concepts):
        c["id"] = f"c{i}"
    return concepts


def concept_by_id(concepts: list[dict], cid: str) -> dict | None:
    for c in concepts:
        if c.get("id") == cid:
            return c
    return None
