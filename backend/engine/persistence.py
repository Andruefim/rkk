"""
Фаза 1: персистентная память графа и temporal (.rkk через torch.save).

Сохраняет: node_ids, nodes, GNN/NOTEARS state_dict, буферы obs/int, замороженные рёбра,
параметры TemporalBlankets (частичная подгрузка при смене d).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from engine.causal_graph import CausalGraph, USE_GNN


def default_memory_path() -> Path:
    raw = os.environ.get("RKK_MEMORY_PATH", "state/autosave.rkk")
    return Path(raw)


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _load_partial_state_dict(module: torch.nn.Module, sd: dict[str, torch.Tensor]) -> None:
    cur = module.state_dict()
    dev = next(module.parameters()).device
    for k, v in sd.items():
        if k not in cur:
            continue
        t = cur[k]
        tv = v.to(dtype=t.dtype)
        if t.shape == tv.shape:
            t.copy_(tv.to(dev))
        elif k == "W" and t.dim() == 2 and tv.dim() == 2:
            a = min(t.shape[0], tv.shape[0])
            b = min(t.shape[1], tv.shape[1])
            t[:a, :b].copy_(tv[:a, :b].to(dev))
        elif "weight" in k and t.dim() == 2 and tv.dim() == 2:
            a = min(t.shape[0], tv.shape[0])
            b = min(t.shape[1], tv.shape[1])
            t[:a, :b].copy_(tv[:a, :b].to(dev))
        elif t.shape == tv.shape:
            t.copy_(tv.to(dev))


def pack_graph(graph: CausalGraph) -> dict[str, Any]:
    core_sd = graph._core.state_dict() if graph._core is not None else {}
    frozen = sorted(graph._frozen_edge_set) if graph._frozen_edge_set else []
    concept_meta: dict[str, Any] = {}
    for k, v in graph._concept_meta.items():
        concept_meta[str(k)] = {
            "members": list(v.get("members", [])),
            "pattern": list(v.get("pattern", [])),
            "detector_id": str(v.get("detector_id", "")),
        }
    return {
        "node_ids": list(graph._node_ids),
        "nodes": {k: float(v) for k, v in graph.nodes.items()},
        "obs_buffer": [list(row) for row in graph._obs_buffer],
        "int_buffer": list(graph._int_buffer),
        "train_losses": list(graph.train_losses),
        "frozen_edges": [list(p) for p in frozen],
        "concept_meta": concept_meta,
        "use_gnn": USE_GNN,
        "core_state_dict": _cpu_state_dict(graph._core) if graph._core is not None else {},
    }


def pack_temporal(agent) -> dict[str, Any]:
    tb = agent.temporal
    return {
        "d_input": tb.d_input,
        "fast": _cpu_state_dict(tb.fast),
        "slow": _cpu_state_dict(tb.slow),
        "integrator": _cpu_state_dict(tb.integrator),
        "context_proj": _cpu_state_dict(tb.context_proj),
        "h_fast": tb.h_fast.detach().cpu().clone(),
        "h_slow": tb.h_slow.detach().cpu().clone(),
        "slow_context": tb.slow_context.detach().cpu().clone(),
        "step_count": tb._step_count,
    }


def unpack_graph(agent, data: dict[str, Any]) -> None:
    g = agent.graph
    node_ids = list(data["node_ids"])
    nodes = {k: float(v) for k, v in data["nodes"].items()}
    g.rebind_variables(node_ids, nodes)
    fe = data.get("frozen_edges") or []
    g._frozen_edge_set = {(str(a), str(b)) for a, b in fe}
    obs = data.get("obs_buffer") or []
    g._obs_buffer = [list(map(float, row)) for row in obs][-g.BUFFER_SIZE * 4 :]
    g._int_buffer = list(data.get("int_buffer") or [])[-g.BUFFER_SIZE :]
    tl = data.get("train_losses") or []
    g.train_losses = [float(x) for x in tl][-100:]
    if g._core is not None and data.get("core_state_dict"):
        _load_partial_state_dict(g._core, data["core_state_dict"])
    g._concept_meta = {}
    for k, v in (data.get("concept_meta") or {}).items():
        g._concept_meta[str(k)] = {
            "members": list(v.get("members", [])),
            "pattern": list(v.get("pattern", [])),
            "detector_id": str(v.get("detector_id", "")),
        }
    g.refresh_concept_aggregates()
    g._sync_frozen_W_into_core()


def unpack_temporal(agent, data: dict[str, Any], device: torch.device) -> None:
    from engine.temporal import TemporalBlankets

    d_in = int(data["d_input"])
    if agent.temporal.d_input != d_in:
        agent.temporal = TemporalBlankets(d_input=d_in, device=device)
    tb = agent.temporal
    _load_partial_state_dict(tb.fast, data.get("fast", {}))
    _load_partial_state_dict(tb.slow, data.get("slow", {}))
    _load_partial_state_dict(tb.integrator, data.get("integrator", {}))
    _load_partial_state_dict(tb.context_proj, data.get("context_proj", {}))
    if "h_fast" in data:
        hf = data["h_fast"]
        if hf.shape == tb.h_fast.shape:
            tb.h_fast = hf.to(device=device, dtype=tb.h_fast.dtype)
    if "h_slow" in data:
        hs = data["h_slow"]
        if hs.shape == tb.h_slow.shape:
            tb.h_slow = hs.to(device=device, dtype=tb.h_slow.dtype)
    if "slow_context" in data:
        sc = data["slow_context"]
        if sc.shape == tb.slow_context.shape:
            tb.slow_context = sc.to(device=device, dtype=tb.slow_context.dtype)
    tb._step_count = int(data.get("step_count", 0))


def save_simulation(sim, path: Path | str | None = None) -> dict[str, Any]:
    path = Path(path) if path else default_memory_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rkk_version": 2,
        "tick": int(sim.tick),
        "current_world": sim.current_world,
        "graph": pack_graph(sim.agent.graph),
        "temporal": pack_temporal(sim.agent),
        "materialized_detector_concept_ids": list(
            getattr(sim, "_materialized_detector_concept_ids", set())
        ),
    }
    torch.save(payload, path)
    return {"ok": True, "path": str(path.resolve())}


def load_simulation(sim, path: Path | str | None = None) -> dict[str, Any]:
    path = Path(path) if path else default_memory_path()
    if not path.is_file():
        return {"ok": False, "error": f"file not found: {path}"}
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    if not isinstance(payload, dict) or "graph" not in payload:
        return {"ok": False, "error": "invalid .rkk payload"}
    unpack_graph(sim.agent, payload["graph"])
    unpack_temporal(sim.agent, payload.get("temporal", {}), sim.device)
    sim.tick = int(payload.get("tick", sim.tick))
    mid = set(payload.get("materialized_detector_concept_ids", []))
    if hasattr(sim, "_materialized_detector_concept_ids"):
        sim._materialized_detector_concept_ids = {str(x) for x in mid}
        for _nid, meta in sim.agent.graph._concept_meta.items():
            d = str(meta.get("detector_id", ""))
            if d:
                sim._materialized_detector_concept_ids.add(d)
    return {"ok": True, "path": str(path.resolve()), "tick": sim.tick}


def autosave_every_ticks() -> int:
    try:
        return max(0, int(os.environ.get("RKK_MEMORY_AUTOSAVE_EVERY", "500")))
    except ValueError:
        return 0
