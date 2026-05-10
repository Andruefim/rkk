"""
PEARL-подобный контекстный posterior для z (Фаза I): скользящее окно последних K
наблюдений + усреднение; опционально различение режима по physics_context из эпизода.

Без отдельного bi-level MAML — только stateless конденсат из истории.
"""
from __future__ import annotations

import os
from collections import deque
import numpy as np


def context_window_k() -> int:
    try:
        return max(4, int(os.environ.get("RKK_PEARL_CONTEXT_K", "16")))
    except ValueError:
        return 16


def pearl_context_enabled() -> bool:
    return os.environ.get("RKK_PEARL_CONTEXT", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


class RollingObservationPosterior:
    """Хранит последние K векторов наблюдения (упорядоченные ключи графа)."""

    def __init__(self, node_ids: list[str], k: int | None = None):
        self._ids = list(node_ids)
        self._k = k if k is not None else context_window_k()
        self._buf: deque[np.ndarray] = deque(maxlen=self._k)
        self._last_physics: dict[str, float] = {}

    def remap_node_ids(self, new_ids: list[str]) -> None:
        """Pad/truncate history when the graph grows (neurogenesis) instead of wiping the buffer."""
        if new_ids == self._ids:
            return
        oid_to_j = {n: j for j, n in enumerate(self._ids)}
        new_buf: deque[np.ndarray] = deque(maxlen=self._k)
        for v in self._buf:
            nv = np.zeros(len(new_ids), dtype=np.float64)
            for i, nid in enumerate(new_ids):
                if nid in oid_to_j:
                    j = oid_to_j[nid]
                    if j < len(v):
                        nv[i] = float(v[j])
            new_buf.append(nv)
        self._ids = list(new_ids)
        self._buf = new_buf

    def push(
        self,
        obs_dict: dict[str, float],
        physics_context: dict[str, float] | None = None,
    ) -> None:
        if physics_context:
            self._last_physics = dict(physics_context)
        vec = np.array(
            [float(obs_dict.get(n, obs_dict.get(f"phys_{n}", 0.5))) for n in self._ids],
            dtype=np.float64,
        )
        self._buf.append(vec)

    def mean_z(self) -> np.ndarray:
        if not self._buf:
            return np.zeros(len(self._ids), dtype=np.float64)
        return np.mean(np.stack(list(self._buf), axis=0), axis=0)

    def last_physics_context(self) -> dict[str, float]:
        return dict(self._last_physics)

    def task_embedding(self) -> np.ndarray:
        """Phase I: combined posterior mean + last ``physics_context`` (regime vs noise)."""
        return self.task_hint_from_physics(self._last_physics)

    def task_hint_from_physics(self, physics_context: dict[str, float]) -> np.ndarray:
        """Простая фича-надстройка: склеить усреднённый z с нормированным physics_context."""
        z = self.mean_z()
        if not physics_context:
            return z
        vals = np.array(list(physics_context.values()), dtype=np.float64)
        if vals.size == 0:
            return z
        pad = min(8, vals.size)
        tail = vals[-pad:] / (np.abs(vals[-pad:]).max() + 1e-6)
        return np.concatenate([z[: max(1, len(z) - pad)], tail])


def physics_task_label(physics_context: dict[str, float]) -> str:
    """Coarse string label for dynamics regime (Phase D/I episodic clustering)."""
    gz = float(physics_context.get("gravity_z", -9.81))
    ff = float(
        physics_context.get(
            "floor_lateral_friction",
            physics_context.get("base_lateral_friction", 1.0),
        )
    )
    return f"g_z={gz:.3f}|μ_floor={ff:.3f}"
