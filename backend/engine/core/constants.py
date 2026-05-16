"""Тики, фазы открытия, частоты циклов — общие константы симуляции."""
from __future__ import annotations

import os

PHASE_THRESHOLDS = [0.0, 0.15, 0.30, 0.50, 0.70, 0.88]
PHASE_HOLD_TICKS = 12
PHASE_NAMES = [
    "",
    "Causal Crib",
    "Robotic Explorer",
    "Social Sandbox",
    "Value Lock",
    "Open Reality",
]

# Visual mode: полный GNN→cortex на каждом тике дорог; предсказание для PC обновляем реже
VISION_GNN_FEED_EVERY = 2


def cpg_loop_hz_from_env() -> float:
    """0 = CPG синхронно с тиком агента; >0 = отдельный поток (часто 60)."""
    try:
        hz = float(os.environ.get("RKK_CPG_LOOP_HZ", "0"))
    except ValueError:
        hz = 0.0
    return max(0.0, min(hz, 240.0))


def cpg_during_fixed_root_enabled() -> bool:
    """
    Ритм ног (CPG) при закреплённом pelvis: in-place шаг, intent_stride имеет физический эффект.
    По умолчанию следует RKK_LOCOMOTION_CPG.
    """
    raw = os.environ.get("RKK_CPG_DURING_FIXED_ROOT", "").strip()
    if not raw:
        v = os.environ.get("RKK_LOCOMOTION_CPG", "0").strip().lower()
        return v in ("1", "true", "yes", "on")
    return raw.lower() in ("1", "true", "yes", "on")


def agent_loop_hz_from_env() -> float:
    """0 = полный tick_step в вызывающем потоке (WS); >0 = high-level в daemon."""
    try:
        hz = float(os.environ.get("RKK_AGENT_LOOP_HZ", "0"))
    except ValueError:
        hz = 0.0
    return max(0.0, min(hz, 60.0))


def l3_loop_hz_from_env() -> float:
    """Cadence для L3 planning/imagination."""
    try:
        hz = float(os.environ.get("RKK_L3_LOOP_HZ", "0"))
    except ValueError:
        hz = 0.0
    return max(0.0, min(hz, 30.0))


def l4_worker_enabled() -> bool:
    v = os.environ.get("RKK_L4_WORKER", "1").strip().lower()
    return v in ("1", "true", "yes", "on")
