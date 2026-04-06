"""
Фаза 1 — LocalReflex: кинематические цепи только из соседей URDF.

- Метаданные цепей в snapshot (snapshot_chains_metadata).
- RKK_LOCAL_REFLEX / RKK_LOCAL_REFLEX_TRAIN: по умолчанию вкл.; 0/off — выкл.
  Train: мини-NOTEARS по каждой полной цепи, один шаг Adam после основного train_step.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from engine.environment_humanoid import KINEMATIC_CHAINS

if TYPE_CHECKING:
    from engine.causal_graph import CausalGraph


def local_reflex_enabled() -> bool:
    return os.environ.get("RKK_LOCAL_REFLEX", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def local_reflex_train_enabled() -> bool:
    return os.environ.get("RKK_LOCAL_REFLEX_TRAIN", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def snapshot_chains_metadata(graph_node_ids: list[str]) -> dict[str, Any]:
    """Какие цепи полностью присутствуют в текущем GNN (по именам узлов)."""
    gset = set(graph_node_ids)
    covered = []
    partial = []
    for chain in KINEMATIC_CHAINS:
        if all(j in gset for j in chain):
            covered.append(list(chain))
        elif any(j in gset for j in chain):
            partial.append(list(chain))
    return {
        "enabled_flag": local_reflex_enabled(),
        "chains_full": covered,
        "chains_partial": partial,
        "n_chain_types": len(KINEMATIC_CHAINS),
    }


class LocalReflexChainCore:
    """
    Мини-ядро NOTEARS по одной цепи (d = len(chain)), только рёбра i→i+1.
    Опционально: отдельный Adam; не связан с основным CausalGraph до явной интеграции.
    """

    def __init__(self, chain: tuple[str, ...], device: torch.device):
        from engine.causal_graph import NOTEARSCore

        self.chain = tuple(chain)
        self.d = len(chain)
        self.device = device
        self.core = NOTEARSCore(self.d, device)
        with torch.no_grad():
            W = self.core.W
            w = torch.zeros_like(W)
            for i in range(self.d - 1):
                w[i, i + 1] = 0.65
            W.copy_(w)
        self.optim = torch.optim.Adam(self.core.parameters(), lr=1e-2)


def train_chains_parallel(
    *,
    graph: "CausalGraph",
    device: torch.device,
    cores: dict[tuple[str, ...], LocalReflexChainCore],
) -> dict[str, Any]:
    """
    Одна эпоха: по каждой цепи из KINEMATIC_CHAINS, полностью покрытой _node_ids,
    MSE(core(X_t), X_{t+1}) по всем подряд парам obs_buffer.
    """
    obs_b = graph._obs_buffer
    if len(obs_b) < 4:
        return {"trained_chains": 0, "losses": {}, "skipped": "obs_buffer<4"}
    gni = graph._node_ids
    losses: dict[str, float] = {}
    n_ok = 0
    for chain in KINEMATIC_CHAINS:
        if not all(j in gni for j in chain):
            continue
        key = tuple(chain)
        if key not in cores:
            cores[key] = LocalReflexChainCore(chain, device)
        lr = cores[key]
        idx = [gni.index(j) for j in chain]
        xs: list[list[float]] = []
        ys: list[list[float]] = []
        for k in range(len(obs_b) - 1):
            xs.append([float(obs_b[k][ii]) for ii in idx])
            ys.append([float(obs_b[k + 1][ii]) for ii in idx])
        if len(xs) < 3:
            continue
        X = torch.tensor(xs, dtype=torch.float32, device=device)
        Y = torch.tensor(ys, dtype=torch.float32, device=device)
        lr.optim.zero_grad()
        pred = lr.core(X)
        loss = F.mse_loss(pred, Y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lr.core.parameters(), max_norm=1.0)
        lr.optim.step()
        losses["->".join(chain)] = round(float(loss.item()), 5)
        n_ok += 1
    return {"trained_chains": n_ok, "losses": losses}
