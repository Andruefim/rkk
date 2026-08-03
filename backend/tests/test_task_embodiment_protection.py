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
        self._task_fall_start_range = None
        self._current_approach_range = None
        self._task_locked_body_id = 7
        self._task_face_lift_tick = -10_000
        self.reset_calls: list[int] = []
        self.face_lift_calls: list[tuple[float, float]] = []
        self.agent = SimpleNamespace(
            env=SimpleNamespace(
                reset_stance=lambda: self.reset_calls.append(1),
                face_target_and_lift=lambda xy, stand_z=None: (
                    self.face_lift_calls.append((float(xy[0]), float(xy[1])))
                    or {"ok": True, "x": 2.0, "y": 0.0, "yaw": 3.14}
                ),
            ),
            graph=SimpleNamespace(_obs_buffer=[], _int_buffer=[]),
        )

    def _add_event(self, *args, **kwargs) -> None:
        pass

    def _genome_fall_recovery_enabled(self) -> bool:
        return False

    def _unwrap_base_env(self, env: object) -> object:
        return env

    def _task_fall_assist_progress_blocks_reset(self) -> bool:
        start = getattr(self, "_task_fall_start_range", None)
        current = getattr(self, "_current_approach_range", None)
        if start is not None and current is not None:
            return float(start) - float(current) >= 0.15
        return False

    def _locked_contact_target_xy(self) -> tuple[float, float] | None:
        if self._task_locked_body_id is None:
            return None
        return (0.0, 0.0)


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
    """Without progress, unlocked stall may spawn-reset once; locked body must face-lift."""
    sim = _FallSimStub()
    sim.tick = 200
    sim._fall_recovery_last_progress_tick = 0
    sim._task_fallen_ticks = 120
    # No locked body → spawn assist allowed when not progressing.
    sim._task_locked_body_id = None
    monkeypatch.setenv("RKK_FALL_RECOVERY_STALL_TICKS", "1")
    monkeypatch.setenv("RKK_TASK_FALL_ASSIST_TICKS", "120")
    monkeypatch.setenv("RKK_TASK_FACE_LIFT_EVERY", "16")
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
    assert sim.face_lift_calls == []

    # After the one-shot spawn assist, further stalls use in-place face+lift
    # (does not consume another spawn teleport).
    sim._fall_recovery_active = True
    sim._fall_recovery_last_progress_tick = 0
    sim._task_locked_body_id = 7
    sim.tick = 400
    assert sim._maybe_recover_or_reset_after_fall(obs, apply_genome_program=False) is True
    assert sim.reset_calls == [1]
    assert sim.face_lift_calls == [(0.0, 0.0)]
    assert sim._task_fall_assist_used is True


def test_locked_body_stall_never_spawn_teleports(monkeypatch: pytest.MonkeyPatch) -> None:
    """With a contact body locked, stall assist must face+lift — never spawn."""
    sim = _FallSimStub()
    sim.tick = 200
    sim._fall_recovery_last_progress_tick = 0
    sim._task_fallen_ticks = 120
    sim._task_locked_body_id = 7
    sim._task_fall_start_range = None
    sim._current_approach_range = None
    monkeypatch.setenv("RKK_FALL_RECOVERY_STALL_TICKS", "1")
    monkeypatch.setenv("RKK_TASK_FALL_ASSIST_TICKS", "120")
    monkeypatch.setenv("RKK_TASK_FACE_LIFT_EVERY", "16")
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
    assert sim.reset_calls == []
    assert sim.face_lift_calls == [(0.0, 0.0)]
    assert sim._task_fall_assist_used is False
    assert sim._fall_recovery_active is False


def test_protected_stall_face_lifts_when_range_improving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Progress blocks spawn teleport — in-place face+lift toward locked body."""
    sim = _FallSimStub()
    sim.tick = 200
    sim._fall_recovery_last_progress_tick = 0
    sim._task_fallen_ticks = 120
    sim._task_fall_start_range = 4.8
    # Outside the final no-face band (>1.20) but still clear progress.
    sim._current_approach_range = 1.55
    sim._task_approach_best_phys = 1.55
    monkeypatch.setenv("RKK_FALL_RECOVERY_STALL_TICKS", "1")
    monkeypatch.setenv("RKK_TASK_FALL_ASSIST_TICKS", "120")
    monkeypatch.setenv("RKK_TASK_FACE_LIFT_EVERY", "16")
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
    assert sim.reset_calls == []
    assert sim._task_fall_assist_used is False
    assert sim.face_lift_calls == [(0.0, 0.0)]
    assert sim._fall_recovery_active is False


def test_final_band_blocks_face_lift_even_when_progressing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inside phys<=1.20, face-lift must not interrupt the final close."""
    sim = _FallSimStub()
    sim.tick = 200
    sim._fall_recovery_last_progress_tick = 0
    sim._task_fallen_ticks = 120
    sim._task_fall_start_range = 4.8
    sim._current_approach_range = 0.96
    sim._task_approach_best_phys = 0.96
    monkeypatch.setenv("RKK_FALL_RECOVERY_STALL_TICKS", "1")
    monkeypatch.setenv("RKK_TASK_FALL_ASSIST_TICKS", "120")
    monkeypatch.setenv("RKK_TASK_FACE_LIFT_EVERY", "16")
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
    assert sim._maybe_recover_or_reset_after_fall(obs, apply_genome_program=False) is False
    assert sim.reset_calls == []
    assert sim.face_lift_calls == []
    assert sim._task_fall_assist_used is False


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


def test_world_transfer_skipped_during_human_task(
    agi_loop_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Curriculum world switch must not reload URDF mid one-shot task."""
    from engine.intention_cortex import IntentionCortex

    sim = AgiLoopSim()
    _bind_displace_chair_task(sim)
    sim.current_world = "humanoid"
    switched: list[str] = []

    class _Switcher:
        def switch(self, target: str) -> None:
            switched.append(target)

    sim.switcher = _Switcher()
    cortex = IntentionCortex.__new__(IntentionCortex)
    cortex._curriculum_graph = SimpleNamespace(
        transfer_goals_to_world=lambda *a, **k: None
    )
    monkeypatch.setenv("RKK_GOAL_GEN_ENABLED", "1")
    monkeypatch.setenv("RKK_GOAL_WORLD_SWITCH_EVERY", "600")
    cortex._maybe_world_transfer(sim, tick=600)
    assert switched == []

    # Without an active task, switch proceeds.
    sim._task_tree_ctrl = None
    sim._task_binding = SimpleNamespace(active_task=None)
    cortex._maybe_world_transfer(sim, tick=600)
    assert switched == ["humanoid_variant"]
