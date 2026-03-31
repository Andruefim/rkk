"""
CausalGraph — Python/PyTorch port of CausalGraph.ts
Добавляет:
  - GPU-ускоренный MDL через torch
  - NOTEARS-совместимую матрицу смежности W
  - Pinned memory буферы для Host↔Device (улучшение 4)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field

@dataclass
class Edge:
    from_:  str
    to:     str
    weight: float
    alpha_trust: float
    intervention_count: int = 1

    def as_dict(self):
        return {
            "from_": self.from_, "to": self.to,
            "weight": self.weight, "alpha_trust": self.alpha_trust,
            "intervention_count": self.intervention_count,
        }


class CausalGraph:
    """
    DAG с MDL-метрикой, do()-оператором и alpha trust.
    Состояние хранится и в Python (для логики), и в GPU-тензорах (для NOTEARS).
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.nodes:  dict[str, float] = {}   # id → current value
        self.edges:  list[Edge]       = []
        self._mdl_cache: float | None = None

        # Pinned memory буфер для быстрой передачи Host↔Device (улучшение 4)
        self._obs_buffer: torch.Tensor | None = None

    # ── Nodes ──────────────────────────────────────────────────────────────────
    def set_node(self, id_: str, value: float = 0.0) -> None:
        self._mdl_cache = None
        self.nodes[id_] = value

    # ── Edges ──────────────────────────────────────────────────────────────────
    def set_edge(self, from_: str, to: str, weight: float, alpha: float) -> None:
        self._mdl_cache = None
        for e in self.edges:
            if e.from_ == from_ and e.to == to:
                e.weight = weight
                e.alpha_trust = min(1.0, alpha)
                e.intervention_count += 1
                return
        self.edges.append(Edge(from_=from_, to=to, weight=weight, alpha_trust=alpha))

    def remove_edge(self, from_: str, to: str) -> None:
        self._mdl_cache = None
        self.edges = [e for e in self.edges if not (e.from_ == from_ and e.to == to)]

    # ── MDL (GPU-ускоренный) ───────────────────────────────────────────────────
    @property
    def mdl_size(self) -> float:
        if self._mdl_cache is not None:
            return self._mdl_cache
        if not self.edges:
            return 0.0

        # Векторизованный расчёт на GPU
        weights = torch.tensor([e.weight      for e in self.edges], device=self.device)
        alphas  = torch.tensor([e.alpha_trust for e in self.edges], device=self.device)

        # Неопределённость ребра = (1 - |w|) * (1 - alpha)
        uncertainty = (1 - weights.abs()) * (1 - alphas)
        mdl = (1 + uncertainty).sum().item()

        self._mdl_cache = mdl
        return mdl

    # ── Propagate ──────────────────────────────────────────────────────────────
    def propagate(self, variable: str, value: float) -> dict[str, float]:
        """Forward pass через граф. Используется для предсказания до do()."""
        state = {**self.nodes}
        state[variable] = value

        visited = {variable}
        queue   = [variable]
        while queue:
            cur = queue.pop(0)
            for e in self.edges:
                if e.from_ != cur:
                    continue
                delta = (state[cur] - self.nodes.get(cur, state[cur]))
                state[e.to] = state.get(e.to, 0) + e.weight * delta * e.alpha_trust
                if e.to not in visited:
                    visited.add(e.to)
                    queue.append(e.to)
        return state

    # ── Alpha mean ─────────────────────────────────────────────────────────────
    @property
    def alpha_mean(self) -> float:
        if not self.edges:
            return 0.05
        return sum(e.alpha_trust for e in self.edges) / len(self.edges)

    # ── Edge uncertainty ───────────────────────────────────────────────────────
    def edge_uncertainty(self, from_: str, to: str) -> float:
        for e in self.edges:
            if e.from_ == from_ and e.to == to:
                return (1 - abs(e.weight)) * (1 - e.alpha_trust)
        return 1.0  # неизвестное ребро = максимальная неопределённость

    # ── Adjacency matrix (для NOTEARS) ─────────────────────────────────────────
    def adjacency_matrix(self, node_ids: list[str]) -> torch.Tensor:
        d = len(node_ids)
        idx = {n: i for i, n in enumerate(node_ids)}
        W   = torch.zeros(d, d, device=self.device)
        for e in self.edges:
            if e.from_ in idx and e.to in idx:
                W[idx[e.from_], idx[e.to]] = e.weight
        return W

    # ── Serialise ──────────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "nodes": self.nodes,
            "edges": [e.as_dict() for e in self.edges],
            "mdl":   self.mdl_size,
        }

    def clone(self) -> "CausalGraph":
        g = CausalGraph(self.device)
        g.nodes = dict(self.nodes)
        g.edges = [Edge(**e.as_dict()) for e in self.edges]
        return g
