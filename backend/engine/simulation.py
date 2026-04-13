"""
Точка входа: класс `Simulation` собран в `engine.features.simulation.simulation_main`.
Реэкспорт для совместимости: `from engine.simulation import Simulation, WORLDS`.
"""
from __future__ import annotations

from engine.core.world import WORLDS
from engine.features.simulation.simulation_main import Simulation

__all__ = ["Simulation", "WORLDS"]
