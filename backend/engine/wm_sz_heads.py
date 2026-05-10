"""
Двухголовая декомпозиция s/z поверх эмбеддинга состояния (Фаза E).

При RKK_SZ_SPLIT=1 создаётся линейная голова z без изменения основного ядра GNN.
Первые RKK_SZ_FREEZE_S_STEPS шагов — заморозка «s» (не вызывать шаг оптимизатора ядра).

Интеграция в train_step графа — постепенная; модуль держит только голову и счётчик.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn


def sz_split_enabled() -> bool:
    return os.environ.get("RKK_SZ_SPLIT", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def sz_freeze_s_steps() -> int:
    try:
        return max(0, int(os.environ.get("RKK_SZ_FREEZE_S_STEPS", "1000")))
    except ValueError:
        return 1000


class ContextZHead(nn.Module):
    """Проекция вектора состояния (d,) → z_dim."""

    def __init__(self, d_in: int, z_dim: int = 32):
        super().__init__()
        self.proj = nn.Linear(d_in, z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def maybe_build_z_head(d_in: int, device: torch.device) -> ContextZHead | None:
    if not sz_split_enabled():
        return None
    try:
        z_dim = max(8, int(os.environ.get("RKK_SZ_DIM", "32")))
    except ValueError:
        z_dim = 32
    h = ContextZHead(d_in, z_dim).to(device)
    return h
