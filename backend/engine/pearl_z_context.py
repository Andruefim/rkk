"""
Phase I — PEARL-like task embedding ``z``: amortized posterior placeholder.

Uses the last K stacked observation vectors plus optional ``physics_context`` from episodic
memory (Phase D) so outer-loop task inference can distinguish simulator regimes.

Not full PEARL/MAML inner-loop SGD; this module provides a deterministic encoding hook for
downstream controllers when ``RKK_PEARL_Z_ENCODER=1``.
"""
from __future__ import annotations

import hashlib
import os
from typing import Sequence

import numpy as np


def pearl_z_encoder_enabled() -> bool:
    return os.environ.get("RKK_PEARL_Z_ENCODER", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _env_int(key: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(key, str(default))))
    except ValueError:
        return default


def encode_z_posterior_placeholder(
    last_obs_vectors: Sequence[Sequence[float]],
    physics_context: dict[str, float] | None = None,
    *,
    z_dim: int | None = None,
) -> list[float]:
    """
    Cheap deterministic pseudo-posterior: tanh projection of stacked recent obs + physics hash.
    Replace with a small amortized net once episodic batches stabilize.
    """
    k = _env_int("RKK_PEARL_Z_K", 8)
    rows = list(last_obs_vectors)[-k:]
    if not rows:
        dim = z_dim or _env_int("RKK_SZ_DIM", 16)
        return [0.0] * max(4, min(128, dim))

    flat = np.concatenate([np.asarray(r, dtype=np.float64).ravel() for r in rows])
    pc = physics_context or {}
    raw = flat.tobytes() + repr(sorted(pc.items())).encode()
    h = hashlib.sha256(raw).digest()
    dim = z_dim or _env_int("RKK_SZ_DIM", 16)
    dim = max(4, min(128, dim))
    out: list[float] = []
    bias = 0.01 * (float(flat.mean()) if flat.size else 0.0)
    for i in range(dim):
        b = h[i % len(h)]
        out.append(float(np.tanh((b - 128) / 64.0 + bias)))
    return out


def physics_task_label(physics_context: dict[str, float]) -> str:
    """Human-readable coarse label for clustering episodes by dynamics regime."""
    gz = float(physics_context.get("gravity_z", -9.81))
    ff = float(physics_context.get("floor_lateral_friction", physics_context.get("base_lateral_friction", 1.0)))
    return f"g_z={gz:.3f}|μ_floor={ff:.3f}"
