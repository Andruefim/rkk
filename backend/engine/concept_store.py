"""
Фаза 2, часть 3: формирование концептов из устойчивых паттернов SlotAttention.

Расширяет идеи slot_lexicon: стабильный слот + корреляция с физ. переменными → узел concept_* в GNN.
"""
from __future__ import annotations

import math
import uuid
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.causal_graph import CausalGraph

# Пороги срабатывания формирования концепта
CONCEPT_STABILITY_THRESHOLD = 0.82  # cosine similarity между соседними кадрами
CONCEPT_MIN_FRAMES = 24  # сколько кадров подряд должен быть стабильным
CONCEPT_MIN_VARIABILITY = 0.03  # слот должен иметь динамику
CONCEPT_PHYS_CORR_THRESHOLD = 0.40  # |corr| > threshold


@dataclass
class Concept:
    """Один устойчивый концепт — объект или отношение, открытый агентом."""

    cid: str
    label: str | None
    mean_slot_vec: np.ndarray
    slot_idx: int
    phys_vars: list[str]
    corr_scores: dict[str, float]
    uses: int = 0
    stable_frames: int = 0
    created_tick: int = 0


class ConceptStore:
    """
    Формирует концепты из устойчивых паттернов SlotAttention.

    Вызывается периодически из simulation.py при visual mode.
    Добавляет concept_* узлы в GNN через CausalGraph.set_node / set_edge.
    """

    def __init__(self, n_slots: int, variable_ids: list[str]):
        self.n_slots = n_slots
        self.variable_ids = [
            v for v in variable_ids
            if not str(v).startswith("slot_") and not str(v).startswith("concept_")
        ]
        self.concepts: dict[str, Concept] = {}

        self._slot_vec_history: list[np.ndarray] = []
        self._phys_val_history: list[dict[str, float]] = []
        self._stable_streak: list[int] = [0] * n_slots

        self._max_history = 128

    def _sync_variable_ids(self, graph_node_ids: list[str] | None) -> None:
        if graph_node_ids is None:
            return
        self.variable_ids = [
            v for v in graph_node_ids
            if not str(v).startswith("slot_") and not str(v).startswith("concept_")
        ]

    def update(
        self,
        slot_vecs: torch.Tensor,
        slot_values: list[float],
        variability: list[float],
        phys_obs: dict[str, float],
        tick: int,
        *,
        graph_node_ids: list[str] | None = None,
    ) -> list[Concept]:
        """
        Обновляем историю и проверяем условия формирования концепта.
        Возвращает список новых концептов (обычно пустой).
        """
        self._sync_variable_ids(graph_node_ids)

        sv_np = slot_vecs.detach().cpu().float().numpy()
        self._slot_vec_history.append(sv_np)
        self._phys_val_history.append(dict(phys_obs))

        if len(self._slot_vec_history) > self._max_history:
            self._slot_vec_history.pop(0)
            self._phys_val_history.pop(0)

        new_concepts: list[Concept] = []
        if len(self._slot_vec_history) < CONCEPT_MIN_FRAMES:
            return new_concepts

        prev = self._slot_vec_history[-2]
        curr = sv_np

        for k in range(self.n_slots):
            var_k = float(variability[k]) if k < len(variability) else 0.0

            if var_k < CONCEPT_MIN_VARIABILITY:
                self._stable_streak[k] = 0
                continue

            sim = float(
                F.cosine_similarity(
                    torch.from_numpy(prev[k]).unsqueeze(0),
                    torch.from_numpy(curr[k]).unsqueeze(0),
                    dim=1,
                ).item()
            )
            if not math.isfinite(sim):
                self._stable_streak[k] = 0
                continue

            if sim >= CONCEPT_STABILITY_THRESHOLD:
                self._stable_streak[k] += 1
            else:
                self._stable_streak[k] = 0

            if self._stable_streak[k] < CONCEPT_MIN_FRAMES:
                continue

            if self._concept_exists_for_slot(k, curr[k]):
                self._update_existing(k, curr[k])
                continue

            phys_corrs = self._compute_phys_correlations(k)
            corr_vars = {
                v: c for v, c in phys_corrs.items()
                if abs(c) >= CONCEPT_PHYS_CORR_THRESHOLD
            }

            if not corr_vars:
                continue

            cid = str(uuid.uuid4()).replace("-", "")[:8]
            concept = Concept(
                cid=cid,
                label=None,
                mean_slot_vec=curr[k].copy(),
                slot_idx=k,
                phys_vars=list(corr_vars.keys()),
                corr_scores=dict(corr_vars),
                stable_frames=self._stable_streak[k],
                created_tick=tick,
            )
            self.concepts[cid] = concept
            new_concepts.append(concept)
            print(
                f"[ConceptStore] New concept {cid}: slot_{k}, "
                f"phys={list(corr_vars.keys())[:3]}, "
                f"stable={self._stable_streak[k]}"
            )

        return new_concepts

    def _concept_exists_for_slot(self, slot_idx: int, vec: np.ndarray) -> bool:
        for c in self.concepts.values():
            if c.slot_idx != slot_idx:
                continue
            sim = float(
                F.cosine_similarity(
                    torch.from_numpy(c.mean_slot_vec).unsqueeze(0),
                    torch.from_numpy(vec).unsqueeze(0),
                    dim=1,
                ).item()
            )
            if math.isfinite(sim) and sim > 0.90:
                return True
        return False

    def _update_existing(self, slot_idx: int, vec: np.ndarray) -> None:
        for c in self.concepts.values():
            if c.slot_idx == slot_idx:
                c.mean_slot_vec = 0.95 * c.mean_slot_vec + 0.05 * vec
                c.uses += 1
                break

    def _compute_phys_correlations(self, slot_idx: int) -> dict[str, float]:
        """Корреляция Пирсона: норма вектора слота vs каждая физ. переменная."""
        tlen = len(self._slot_vec_history)
        if tlen < 16:
            return {}

        slot_series = np.array([
            float(np.linalg.norm(h[slot_idx]))
            for h in self._slot_vec_history
        ])

        corrs: dict[str, float] = {}
        for var in self.variable_ids:
            phys_series = np.array([
                float(h.get(var, 0.5))
                for h in self._phys_val_history
            ])
            if len(phys_series) != len(slot_series):
                continue
            if np.std(phys_series) < 1e-6 or np.std(slot_series) < 1e-6:
                continue
            corr = float(np.corrcoef(slot_series, phys_series)[0, 1])
            if not np.isnan(corr):
                corrs[var] = corr

        return corrs

    def inject_into_graph(self, graph: CausalGraph) -> int:
        """Добавляем новые концепты как узлы в GNN. Возвращает число новых узлов."""
        added = 0
        for cid, concept in self.concepts.items():
            node_name = f"concept_{cid[:4]}"
            if node_name in graph.nodes:
                continue
            val = float(concept.uses / (concept.uses + 10))
            graph.set_node(node_name, val)

            slot_key = f"slot_{concept.slot_idx}"
            if slot_key in graph.nodes:
                graph.set_edge(slot_key, node_name, 0.15, 0.05)

            for phys_var, corr in concept.corr_scores.items():
                if phys_var not in graph.nodes:
                    continue
                w = float(np.clip(abs(corr) * 0.5, 0.06, 0.4))
                sign = 1.0 if corr > 0 else -1.0
                graph.set_edge(node_name, phys_var, sign * w, 0.06)

            added += 1
        return added

    def get_active_concepts(
        self, slot_values: list[float], threshold: float = 0.55
    ) -> list[Concept]:
        """Концепты, чьи слоты сейчас активны."""
        active: list[Concept] = []
        for c in self.concepts.values():
            k = c.slot_idx
            if k < len(slot_values) and slot_values[k] >= threshold:
                active.append(c)
        return active

    def snapshot(self) -> dict:
        return {
            "n_concepts": len(self.concepts),
            "concepts": [
                {
                    "cid": c.cid,
                    "label": c.label,
                    "slot_idx": c.slot_idx,
                    "phys_vars": c.phys_vars[:4],
                    "uses": c.uses,
                    "stable_frames": c.stable_frames,
                    "created_tick": c.created_tick,
                }
                for c in self.concepts.values()
            ],
        }
