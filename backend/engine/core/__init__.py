"""Ядро симуляции: константы, мир humanoid, типы моторики."""
from __future__ import annotations

from engine.core.constants import (
    PHASE_HOLD_TICKS,
    PHASE_NAMES,
    PHASE_THRESHOLDS,
    VISION_GNN_FEED_EVERY,
    agent_loop_hz_from_env,
    cpg_loop_hz_from_env,
    l3_loop_hz_from_env,
    l4_worker_enabled,
)
from engine.core.motor_types import MotorCommandLog, MotorState
from engine.core.world import (
    WORLDS,
    WorldSwitcher,
    _make_env,
    default_bounds,
    resolve_torch_device,
)

__all__ = [
    "PHASE_THRESHOLDS",
    "PHASE_HOLD_TICKS",
    "PHASE_NAMES",
    "VISION_GNN_FEED_EVERY",
    "agent_loop_hz_from_env",
    "cpg_loop_hz_from_env",
    "l3_loop_hz_from_env",
    "l4_worker_enabled",
    "MotorCommandLog",
    "MotorState",
    "WORLDS",
    "WorldSwitcher",
    "_make_env",
    "default_bounds",
    "resolve_torch_device",
]
