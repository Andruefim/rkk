"""Sleep must pin pelvis via Simulation.enable_fixed_root (not base env)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_humanoid_env_uses_set_fixed_root_not_enable():
    from engine.environment_humanoid import EnvironmentHumanoid

    assert hasattr(EnvironmentHumanoid, "set_fixed_root")
    assert not hasattr(EnvironmentHumanoid, "enable_fixed_root")


def test_sleep_attach_calls_sim_enable_fixed_root():
    from engine.features.simulation.mixin_world import SimulationWorldMixin

    sim = SimulationWorldMixin.__new__(SimulationWorldMixin)
    sim.current_world = "humanoid"
    sim._fixed_root_active = False
    sim._sleep_pinned = False
    sim._sleep_prev_fixed_root = False
    def _enable():
        sim._fixed_root_active = True
        return {"fixed_root": True}

    sim.enable_fixed_root = MagicMock(side_effect=_enable)

    ok = sim._sleep_attach_fixed_root()
    assert ok is True
    assert sim._sleep_pinned is True
    assert sim._sleep_prev_fixed_root is False
    sim.enable_fixed_root.assert_called_once()


def test_sleep_attach_skips_when_already_pinned():
    sim = __import__(
        "engine.features.simulation.mixin_world", fromlist=["SimulationWorldMixin"]
    ).SimulationWorldMixin.__new__(
        __import__(
            "engine.features.simulation.mixin_world",
            fromlist=["SimulationWorldMixin"],
        ).SimulationWorldMixin
    )
    sim.current_world = "humanoid"
    sim._fixed_root_active = True
    sim._sleep_pinned = False
    sim.enable_fixed_root = MagicMock()

    ok = sim._sleep_attach_fixed_root()
    assert ok is False
    assert sim._sleep_prev_fixed_root is True
    sim.enable_fixed_root.assert_not_called()


def test_sleep_detach_restores_only_if_we_pinned():
    from engine.features.simulation.mixin_world import SimulationWorldMixin

    sim = SimulationWorldMixin.__new__(SimulationWorldMixin)
    sim._sleep_pinned = True
    sim._sleep_prev_fixed_root = False
    sim.disable_fixed_root = MagicMock()

    sim._sleep_detach_fixed_root()
    sim.disable_fixed_root.assert_called_once()
    assert sim._sleep_pinned is False

    sim._sleep_pinned = True
    sim._sleep_prev_fixed_root = True
    sim.disable_fixed_root.reset_mock()
    sim._sleep_detach_fixed_root()
    sim.disable_fixed_root.assert_not_called()
