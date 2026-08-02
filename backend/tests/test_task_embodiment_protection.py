"""Human task must not be interrupted by fall hard-reset or fixed_root re-attach."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from engine.features.simulation.mixin_fall import SimulationFallMixin
from engine.task_binding import (
    human_task_embodiment_protected,
    human_task_execution_active,
    task_protect_embodiment_enabled,
)
from tests.conftest import AgiLoopSim
from tests.test_task_tree_integration import _bind_displace_chair_task, _chair_scene


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
        self._task_fallen_ticks = 0
        self._task_fall_assist_used = False
        self._task_fall_protected_stall_ticks = 0
        self.reset_calls: list[int] = []
        self.agent = SimpleNamespace(
            env=SimpleNamespace(reset_stance=lambda: self.reset_calls.append(1)),
            graph=SimpleNamespace(_obs_buffer=[], _int_buffer=[]),
        )

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
    assert sim.reset_calls == []


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


def test_protected_stall_eventually_assist_reset_once(monkeypatch: pytest.MonkeyPatch) -> None:
    sim = _FallSimStub()
    sim.tick = 200
    sim._fall_recovery_last_progress_tick = 0
    sim._task_fallen_ticks = 120
    monkeypatch.setenv("RKK_FALL_RECOVERY_STALL_TICKS", "1")
    monkeypatch.setenv("RKK_TASK_FALL_ASSIST_TICKS", "120")
    monkeypatch.setattr(
        "engine.task_binding.human_task_embodiment_protected",
        lambda _sim: True,
    )
    monkeypatch.setattr(
        "engine.task_executive.active_tree_stage_kind",
        lambda _sim: "approach",
    )

    obs = {
        "com_z": 0.2,
        "posture_stability": 0.2,
        "foot_contact_l": 0.0,
        "foot_contact_r": 0.0,
    }
    assert sim._maybe_recover_or_reset_after_fall(obs, apply_genome_program=False) is True
    assert sim.reset_calls == [1]
    assert sim._task_fall_assist_used is True
    assert sim._fall_recovery_active is False

    sim._fall_recovery_active = True
    sim._fall_recovery_last_progress_tick = 0
    sim.tick = 400
    assert sim._maybe_recover_or_reset_after_fall(obs, apply_genome_program=False) is False
    assert sim.reset_calls == [1]


def test_assist_reset_refused_during_verify_goal(monkeypatch: pytest.MonkeyPatch) -> None:
    sim = _FallSimStub()
    sim.tick = 200
    sim._fall_recovery_last_progress_tick = 0
    sim._task_fallen_ticks = 200
    monkeypatch.setenv("RKK_FALL_RECOVERY_STALL_TICKS", "1")
    monkeypatch.setenv("RKK_TASK_FALL_ASSIST_TICKS", "120")
    monkeypatch.setattr(
        "engine.task_binding.human_task_embodiment_protected",
        lambda _sim: True,
    )
    monkeypatch.setattr(
        "engine.task_executive.active_tree_stage_kind",
        lambda _sim: "verify_goal",
    )

    obs = {
        "com_z": 0.2,
        "posture_stability": 0.2,
        "foot_contact_l": 0.0,
        "foot_contact_r": 0.0,
    }
    assert sim._maybe_recover_or_reset_after_fall(obs, apply_genome_program=False) is False
    assert sim.reset_calls == []
    assert sim._task_fall_assist_used is False


def test_prolonged_fall_after_assist_fails_approach_retryably(
    agi_loop_sim: AgiLoopSim, monkeypatch: pytest.MonkeyPatch
) -> None:
    sim = agi_loop_sim
    sim.agent.env.set_scene_extras(_chair_scene())
    _bind_displace_chair_task(sim)
    tt = sim._ensure_task_tree()
    active = tt.active_node
    assert active is not None
    assert active.kind == "approach"

    monkeypatch.setattr(
        "engine.task_binding.human_task_embodiment_protected",
        lambda _sim: True,
    )
    monkeypatch.setenv("RKK_TASK_FALL_FAIL_TICKS", "3")
    sim._task_tree_stage_enter_tick = 0
    sim._task_fall_assist_used = True
    sim._task_fallen_after_assist_ticks = 2
    attempts_before = int(active.attempts)

    sim.tick = 500
    sim._tick_human_task(
        fallen=True,
    )

    assert tt.active_node is not None
    assert tt.active_node.kind == "approach"
    assert tt.active_node.attempts == attempts_before + 1
    assert tt.active_node.failure_reason == "fallen_during_approach"


def test_s2_force_reset_deferred_during_human_task(monkeypatch: pytest.MonkeyPatch) -> None:
    from engine.system2.controller import System2Controller

    ctrl = System2Controller.__new__(System2Controller)
    reset_calls: list[int] = []

    class _Base:
        def reset_stance(self) -> None:
            reset_calls.append(1)

    monkeypatch.setattr(
        "engine.task_binding.human_task_embodiment_protected",
        lambda _sim: True,
    )
    assert ctrl._force_reset_stance_base(_Base(), sim=object()) is False
    assert reset_calls == []

    monkeypatch.setattr(
        "engine.task_binding.human_task_embodiment_protected",
        lambda _sim: False,
    )
    assert ctrl._force_reset_stance_base(_Base(), sim=object()) is True
    assert reset_calls == [1]
