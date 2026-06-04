"""
Track I1: EWC / PackNet-lite on role-subgraph W — stable-edge Fisher, catastrophic forgetting guard.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import numpy as np
import torch

from engine.genome.compressor import role_subgraph_indices
from engine.role_types import build_role_map


def _env_flag(key: str, default: bool = False) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def ewc_enabled() -> bool:
    return _env_flag("RKK_EWC_ENABLED", False)


def ewc_lambda() -> float:
    try:
        return float(os.environ.get("RKK_EWC_LAMBDA", "1000"))
    except ValueError:
        return 1000.0


def ewc_packnet() -> bool:
    return _env_flag("RKK_EWC_PACKNET", False)


def ewc_roles_only() -> bool:
    return _env_flag("RKK_EWC_ROLES_ONLY", True)


def ewc_stable_age_min() -> int:
    try:
        return max(0, int(os.environ.get("RKK_EWC_STABLE_AGE_MIN", "200")))
    except ValueError:
        return 200


def ewc_graph_change_thresh() -> float:
    try:
        return float(os.environ.get("RKK_EWC_GRAPH_CHANGE_THRESH", "0.20"))
    except ValueError:
        return 0.20


def _role_edge_pairs(graph: Any) -> list[tuple[str, str, int, int]]:
    """(from, to, i, j) for role-subgraph indices in full W order."""
    ids = list(graph._node_ids)
    if not ids:
        return []
    role_map = graph.role_type_map() if hasattr(graph, "role_type_map") else build_role_map(ids)
    idx = role_subgraph_indices(ids, role_map) if ewc_roles_only() else list(range(len(ids)))
    pairs: list[tuple[str, str, int, int]] = []
    for ii, i in enumerate(idx):
        for jj, j in enumerate(idx):
            if i == j:
                continue
            pairs.append((ids[i], ids[j], i, j))
    return pairs


def subgraph_hash(graph: Any) -> str:
    """Hash of role-subgraph topology (edge presence + stable ages)."""
    edges: list[tuple[str, str, float, int]] = []
    thr = float(getattr(graph, "EDGE_THRESH", 0.05))
    core = graph._core
    if core is None:
        return hashlib.md5(b"empty").hexdigest()
    with torch.no_grad():
        W = core.W_masked()[: graph._d, : graph._d]
    for fr, to, i, j in _role_edge_pairs(graph):
        w = float(W[i, j].item())
        if abs(w) < thr:
            continue
        age = int(graph._edge_age.get((fr, to), 0))
        edges.append((fr, to, round(w, 4), age))
    payload = json.dumps(sorted(edges), sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()


def subgraph_change_ratio(old_hash: str, new_hash: str) -> float:
    if not old_hash or old_hash == new_hash:
        return 0.0
    return 1.0


class ElasticRoleProtector:
    STABLE_EDGE_AGE_MIN: int = 200
    GRAPH_CHANGE_THRESH: float = 0.20

    def __init__(self) -> None:
        self.STABLE_EDGE_AGE_MIN = ewc_stable_age_min()
        self.GRAPH_CHANGE_THRESH = ewc_graph_change_thresh()
        self._last_hash: str = ""
        self._W_anchor: torch.Tensor | None = None
        self._fisher: torch.Tensor | None = None
        self._packnet_mask: torch.Tensor | None = None
        self._archived_fisher: dict[str, float] = {}
        self._ewc_recompute_count = 0
        self._continual_forgetting_ratio = 0.0
        self._stable_edge_count = 0
        self._penalty_last = 0.0

    def edge_age(self, graph: Any, from_: str, to: str) -> int:
        return int(graph._edge_age.get((from_, to), 0))

    def should_recompute(self, graph: Any) -> bool:
        current = subgraph_hash(graph)
        if not self._last_hash:
            self._last_hash = current
            return True
        if current == self._last_hash:
            return False
        change_ratio = subgraph_change_ratio(self._last_hash, current)
        self._last_hash = current
        return change_ratio >= self.GRAPH_CHANGE_THRESH

    def archive_pruned_edge(self, edge_id: str, fisher_val: float) -> None:
        self._archived_fisher[str(edge_id)] = float(fisher_val)

    def compute_fisher(self, graph: Any, obs_buffer: list | None = None) -> torch.Tensor:
        """Diagonal Fisher on role-subgraph; zero on unstable edges (age < min)."""
        d = graph._d
        fisher = torch.zeros(d, d, dtype=torch.float32, device=graph.device)
        if graph._core is None or d < 1:
            return fisher

        with torch.no_grad():
            W = graph._core.W_masked()[:d, :d].detach().float()

        obs = obs_buffer or getattr(graph, "_obs_buffer", []) or []
        resid_scale = 1.0
        if len(obs) >= 4:
            rows = []
            ids = graph._node_ids
            for item in obs[-32:]:
                if isinstance(item, dict):
                    rows.append([float(item.get(nid, 0.5)) for nid in ids])
            if rows:
                X = np.asarray(rows, dtype=np.float64)
                resid_scale = float(1.0 + np.std(X))

        stable_n = 0
        for fr, to, i, j in _role_edge_pairs(graph):
            age = self.edge_age(graph, fr, to)
            w = float(W[i, j].item())
            est = (w * w) * resid_scale
            if age >= self.STABLE_EDGE_AGE_MIN:
                fisher[i, j] = est
                stable_n += 1
            key = f"{fr}->{to}"
            if key in self._archived_fisher:
                fisher[i, j] = max(float(fisher[i, j]), self._archived_fisher[key])

        self._fisher = fisher
        self._stable_edge_count = stable_n
        return fisher

    def anchor_weights(self, graph: Any) -> torch.Tensor:
        if graph._core is None:
            return torch.zeros(0)
        with torch.no_grad():
            W = graph._core.W_masked()[: graph._d, : graph._d].detach().float().clone()
        self._W_anchor = W
        return W

    def ewc_penalty(
        self,
        W_current: torch.Tensor,
        W_anchor: torch.Tensor | None = None,
        fisher: torch.Tensor | None = None,
    ) -> torch.Tensor:
        anchor = W_anchor if W_anchor is not None else self._W_anchor
        f = fisher if fisher is not None else self._fisher
        if anchor is None or f is None:
            return torch.tensor(0.0, device=W_current.device)
        diff = (W_current - anchor) ** 2
        if ewc_packnet() and self._packnet_mask is not None:
            m = self._packnet_mask.to(diff.device)
            diff = diff * m
        pen = ewc_lambda() * (diff * f.to(diff.device)).sum()
        self._penalty_last = float(pen.item())
        return pen

    def on_world_switch(self, graph: Any, world_id: str) -> None:
        """PackNet-lite: freeze anchor mask from previous world before re-anchor."""
        _ = world_id
        if ewc_packnet() and self._W_anchor is not None and graph._core is not None:
            with torch.no_grad():
                W = graph._core.W_masked()[: graph._d, : graph._d].detach().float()
            if self._packnet_mask is None:
                self._packnet_mask = torch.ones_like(W)
            delta = (W - self._W_anchor).abs()
            self._packnet_mask = (self._packnet_mask * (delta < 0.02).float()).clamp(0, 1)
        self.anchor_weights(graph)
        self.compute_fisher(graph)
        self._ewc_recompute_count += 1

    def maybe_update(self, graph: Any, *, world_switch: bool = False) -> dict[str, float]:
        if not ewc_enabled():
            return {}
        if world_switch or self.should_recompute(graph):
            if self._W_anchor is None:
                self.anchor_weights(graph)
            self.compute_fisher(graph)
            self._ewc_recompute_count += 1
        return self.metrics()

    def apply_train_penalty(self, graph: Any, loss: torch.Tensor) -> torch.Tensor:
        if not ewc_enabled() or graph._core is None:
            return loss
        with torch.no_grad():
            W = graph._core.W_masked()[: graph._d, : graph._d]
        pen = self.ewc_penalty(W)
        if float(pen.item()) > 0:
            return loss + pen
        return loss

    def update_forgetting_ratio(self, baseline_sr: float, current_sr: float) -> None:
        if baseline_sr <= 1e-6:
            self._continual_forgetting_ratio = 0.0
            return
        drop = max(0.0, (baseline_sr - current_sr) / baseline_sr)
        self._continual_forgetting_ratio = float(np.clip(drop, 0.0, 1.0))

    def metrics(self) -> dict[str, float | int]:
        return {
            "ewc_enabled": int(ewc_enabled()),
            "ewc_stable_edge_count": int(self._stable_edge_count),
            "ewc_recompute_count": int(self._ewc_recompute_count),
            "continual_forgetting_ratio": round(self._continual_forgetting_ratio, 4),
            "ewc_penalty_last": round(self._penalty_last, 6),
            "ewc_archived_edges": len(self._archived_fisher),
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": ewc_enabled(),
            "packnet": ewc_packnet(),
            "roles_only": ewc_roles_only(),
            "lambda": ewc_lambda(),
            "stable_age_min": self.STABLE_EDGE_AGE_MIN,
            "graph_change_thresh": self.GRAPH_CHANGE_THRESH,
            **self.metrics(),
        }
