"""System 2: медленное макро-планирование (цели + приоритетный intent), опционально LLM → дистилляция в студента."""
from __future__ import annotations

from engine.system2.controller import System2Controller, system2_enabled

__all__ = ["System2Controller", "system2_enabled"]
