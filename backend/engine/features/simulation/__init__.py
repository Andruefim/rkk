"""Симуляция: класс Simulation, снимки WS/HTTP, фоновые циклы, опциональные подсистемы."""
from __future__ import annotations

from engine.features.simulation.background_loops import BackgroundLoopService
from engine.features.simulation.simulation_main import Simulation
from engine.features.simulation.snapshot import build_simulation_snapshot

__all__ = ["BackgroundLoopService", "Simulation", "build_simulation_snapshot"]
