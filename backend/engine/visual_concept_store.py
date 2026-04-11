"""
visual_concept_store.py — L4 визуальные концепты (слоты → phys), отдельно от SemanticConceptStore.

Используется в simulation._l4_worker_loop и sync-пути при RKK_L4_WORKER=0.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class VisualConcept:
    cid: str
    label: str
    slot_idx: int
    phys_vars: list[str] = field(default_factory=list)
    corr_scores: dict[str, float] = field(default_factory=dict)
    uses: int = 1
    stable_frames: int = 1
    created_tick: int = 0


class VisualConceptStore:
    """Slot–physics concept mining for Phase 2 L4 (not the inner-voice embedding table)."""

    def __init__(self, n_slots: int, variable_ids: list[str]):
        self.n_slots = int(n_slots)
        self.variable_ids = list(variable_ids)
        self._inject_queue: list[VisualConcept] = []
        self._injected_cids: set[str] = set()
        self._concepts: dict[str, VisualConcept] = {}

    def update(
        self,
        *,
        slot_vecs: torch.Tensor,
        slot_values: list[float],
        variability: list[float],
        phys_obs: dict[str, float],
        tick: int,
        graph_node_ids: list[str],
    ) -> list[VisualConcept]:
        new_concepts: list[VisualConcept] = []
        gset = set(graph_node_ids)
        phys_keys = [k for k in phys_obs if k in gset and not str(k).startswith("slot_")]

        n = min(self.n_slots, len(slot_values), len(variability))
        for k in range(n):
            try:
                var_k = float(variability[k])
            except (TypeError, ValueError, IndexError):
                continue
            if var_k < 0.12:
                continue
            try:
                sv = float(slot_values[k])
            except (TypeError, ValueError, IndexError):
                continue

            best_p: str | None = None
            best_score = 0.0
            for pk in phys_keys:
                try:
                    pv = float(phys_obs[pk])
                except (TypeError, ValueError):
                    continue
                score = abs(pv - 0.5) * (0.5 + abs(sv - 0.5))
                if score > best_score:
                    best_score = score
                    best_p = pk

            if best_p is None or best_score < 0.08:
                continue

            cid = uuid.uuid4().hex[:12]
            vc = VisualConcept(
                cid=cid,
                label=f"slot{k}_{best_p}",
                slot_idx=k,
                phys_vars=[best_p],
                corr_scores={best_p: float(min(1.0, best_score * 1.5))},
                uses=1,
                stable_frames=int(min(20, 1 + var_k * 30)),
                created_tick=tick,
            )
            new_concepts.append(vc)
            self._concepts[cid] = vc

        self._inject_queue.extend(new_concepts)
        return new_concepts

    def inject_into_graph(self, graph) -> int:
        added = 0
        while self._inject_queue:
            c = self._inject_queue.pop(0)
            if c.cid in self._injected_cids:
                continue
            self._injected_cids.add(c.cid)
            node_name = f"concept_{c.cid[:4]}"
            if node_name in graph.nodes:
                continue
            uses = int(c.uses)
            val = float(uses / (uses + 10)) if uses >= 0 else 0.0
            try:
                graph.set_node(node_name, val)
            except Exception:
                continue
            slot_key = f"slot_{c.slot_idx}"
            if slot_key in graph.nodes:
                try:
                    graph.set_edge(slot_key, node_name, 0.15, 0.05)
                except Exception:
                    pass
            for phys_var, corr in dict(c.corr_scores).items():
                if phys_var not in graph.nodes:
                    continue
                corr_f = float(corr)
                w = float(np.clip(abs(corr_f) * 0.5, 0.06, 0.4))
                sign = 1.0 if corr_f > 0 else -1.0
                try:
                    graph.set_edge(node_name, phys_var, sign * w, 0.06)
                except Exception:
                    pass
            added += 1
        return added

    def snapshot(self) -> dict[str, Any]:
        return {
            "kind": "visual_l4",
            "n_slots": self.n_slots,
            "n_tracked": len(self._concepts),
            "queue": len(self._inject_queue),
            "injected": len(self._injected_cids),
        }
