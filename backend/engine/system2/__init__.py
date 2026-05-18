"""System 2: медленное макро-планирование (цели + приоритетный intent), опционально LLM → дистилляция в студента."""
from __future__ import annotations

from engine.system2.controller import System2Controller, system2_enabled
from engine.system2.wm_planner import (
    plan_s2_wm_candidate,
    s2_wm_gate_strict,
    s2_wm_planner_enabled,
)

__all__ = [
    "System2Controller",
    "system2_enabled",
    "plan_s2_wm_candidate",
    "s2_wm_gate_strict",
    "s2_wm_planner_enabled",
]
