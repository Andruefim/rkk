"""Human task must not be interrupted by fall hard-reset or fixed_root re-attach."""
from __future__ import annotations

import pytest

from engine.features.simulation.mixin_fall import SimulationFallMixin
from engine.task_binding import (
    human_task_embodiment_protected,
    human_task_execution_active,
    task_protect_embodiment_enabled,
)
from tests.conftest import AgiLoopSim
from tests.test_task_tree_integration import _bind_displace_chair_task


def test_human_task_execution_active_with_tree(agi_loop_env: None) -> None:
    sim = AgiLoopSim()
    assert not human_task_execution_active(sim)
    _bind_displace_chair_task(sim)
    assert human_task_execution_active(sim)
    assert human_task_embodiment_protected(sim)


def test_task_protect_embodiment_env_off(agi_loop_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    sim = AgiLoopSim()
    _bind_displace_chair_task(sim)
    monkeypatch.setenv("RKK_TASK_PROTECT_EMBODIMENT", "0")
    assert task_protect_embodiment_enabled() is False
    assert human_task_embodiment_protected(sim) is False


class _FallSimStub(SimulationFallMixin):
    def __init__(self) -> None:
        self.tick = 100
        self._fall_recovery_active = True
        self._fall_recovery_start_tick = 0
        self._fall_recovery_last_progress_tick = 0
        self._fall_recovery_best_score = 0.1
        self._genome_stand_program: list = []
        self._last_fall_reset_tick = -999

    def _add_event(self, *args, **kwargs) -> None:
        pass

    def _genome_fall_recovery_enabled(self) -> bool:
        return False

    def _unwrap_base_env(self, env: object) -> object:
        return env


def test_maybe_recover_skips_hard_reset_during_task(monkeypatch: pytest.MonkeyPatch) -> None:
    sim = _FallSimStub()
    monkeypatch.setenv("RKK_FALL_RECOVERY_STALL_TICKS", "1")
    monkeypatch.setattr(
        "engine.task_binding.human_task_embodiment_protected",
        lambda _sim: True,
    )
    reset_calls: list[int] = []
    sim._try_reset_pose_after_fall = lambda: reset_calls.append(1) or True  # type: ignore[method-assign]

    obs = {
        "com_z": 0.2,
        "posture_stability": 0.2,
        "foot_contact_l": 0.0,
        "foot_contact_r": 0.0,
    }
    assert sim._maybe_recover_or_reset_after_fall(obs, apply_genome_program=False) is False
    assert reset_calls == []


def test_maybe_recover_hard_reset_when_not_protected(monkeypatch: pytest.MonkeyPatch) -> None:
    sim = _FallSimStub()
    sim.tick = 200
    sim._fall_recovery_last_progress_tick = 0
    monkeypatch.setenv("RKK_FALL_RECOVERY_STALL_TICKS", "1")
    monkeypatch.setattr(
        "engine.task_binding.human_task_embodiment_protected",
        lambda _sim: False,
    )
    reset_calls: list[int] = []
    sim._try_reset_pose_after_fall = lambda: reset_calls.append(1) or True  # type: ignore[method-assign]

    obs = {
        "com_z": 0.2,
        "posture_stability": 0.2,
        "foot_contact_l": 0.0,
        "foot_contact_r": 0.0,
    }
    assert sim._maybe_recover_or_reset_after_fall(obs, apply_genome_program=False) is True
    assert reset_calls == [1]
