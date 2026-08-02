"""Gated hard_lock release in vision range mismatch correction."""
from __future__ import annotations

import pytest

from engine.features.simulation.mixin_grounded_language import (
    SimulationGroundedLanguageMixin,
)
from engine.object_working_memory import LatentSceneMemory, ObjectWorkingMemory
from engine.vision_target import VisualTarget


class _RangeCorrectSim(SimulationGroundedLanguageMixin):
    def __init__(self) -> None:
        self.tick = 1000
        self._latent_scene = LatentSceneMemory()
        self._obj_working_memory = ObjectWorkingMemory(self._latent_scene)
        self._manip_resolved_visual = VisualTarget(
            slot_id="slot_7",
            u=0.5,
            v=0.4,
            label="prop",
            confidence=0.6,
            bearing=0.0,
            range_m=3.5,
            diagnostics={"geometry": "objectness_peak"},
        )
        self._vision_range_mismatch_streak = 0
        self._vision_floor_lock_reject_streak = 0
        self._vision_range_correct_until_tick = -1
        self._owm_cached_tick = -1
        self._task_log_prev_vision_range = None
        self._cam = object()

    def _depth_camera_from_sim(self):
        return self._cam


def _seed_locked(sim: _RangeCorrectSim, *, range_m: float = 3.5) -> None:
    scene = sim._latent_scene
    vt = VisualTarget(
        slot_id="slot_7",
        u=0.5,
        v=0.4,
        label="prop",
        confidence=0.6,
        bearing=0.0,
        range_m=range_m,
        diagnostics={"geometry": "objectness_peak"},
    )
    scene.bind_visual_target(vt, tick=1, agent_xy=(0.0, 0.0), agent_forward=(1.0, 0.0))
    assert scene.hard_lock_active is True
    sim._obj_working_memory = ObjectWorkingMemory(scene)
    act = scene.active()
    assert act is not None
    act.range_m = range_m
    assert float(sim._obj_working_memory.range_m) == pytest.approx(range_m)


@pytest.fixture
def vision_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RKK_TASK_RESOLVE", "vision")
    monkeypatch.setenv("RKK_VISION_FLOOR_LOCK_ESCALATE_AFTER", "2")
    monkeypatch.setenv("RKK_VISION_FLOOR_LOCK_FORCE_UNLOCK_AFTER", "3")


def test_range_correct_keeps_hard_lock_when_rebind_rejects(
    vision_on: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    sim = _RangeCorrectSim()
    _seed_locked(sim, range_m=3.5)
    # Need mismatch streak to fire correction immediately.
    sim._vision_range_mismatch_streak = 10

    monkeypatch.setattr(
        sim._latent_scene, "refresh_active_from_live_camera", lambda *a, **k: True
    )
    monkeypatch.setattr(
        sim,
        "_rebind_vision_objectness_peak",
        lambda *a, **k: False,
    )

    events: list[dict] = []

    def _log(event: str, **fields):
        events.append({"event": event, **fields})

    monkeypatch.setattr(
        "engine.features.simulation.mixin_grounded_language.task_log_event",
        _log,
    )

    sim._maybe_correct_vision_range_mismatch(
        1000, oracle_dist=1.4, kind="approach"
    )

    assert sim._latent_scene.hard_lock_active is True
    assert any(e["event"] == "vision_range_correct" for e in events)
    row = next(e for e in events if e["event"] == "vision_range_correct")
    assert row.get("hard_lock_after") is True
    assert row.get("unlocked") is False
    assert row.get("rebind_ok") is False
    assert sim._vision_floor_lock_reject_streak == 1


def test_range_correct_force_unlock_after_reject_streak(
    vision_on: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    sim = _RangeCorrectSim()
    _seed_locked(sim, range_m=3.5)
    sim._vision_range_mismatch_streak = 10
    sim._vision_floor_lock_reject_streak = 2  # next fail → 3 ≥ force_after

    monkeypatch.setattr(
        sim._latent_scene, "refresh_active_from_live_camera", lambda *a, **k: True
    )
    monkeypatch.setattr(sim, "_rebind_vision_objectness_peak", lambda *a, **k: False)

    events: list[dict] = []
    monkeypatch.setattr(
        "engine.features.simulation.mixin_grounded_language.task_log_event",
        lambda event, **fields: events.append({"event": event, **fields}),
    )

    sim._maybe_correct_vision_range_mismatch(
        1100, oracle_dist=1.4, kind="approach"
    )

    assert sim._latent_scene.hard_lock_active is False
    assert any(e["event"] == "vision_range_correct_force_unlock" for e in events)
    row = next(e for e in events if e["event"] == "vision_range_correct")
    assert row.get("force_unlock") is True
    assert row.get("unlocked") is True


def test_successful_rebind_may_unlock(
    vision_on: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    sim = _RangeCorrectSim()
    _seed_locked(sim, range_m=3.5)
    sim._vision_range_mismatch_streak = 10

    monkeypatch.setattr(
        sim._latent_scene, "refresh_active_from_live_camera", lambda *a, **k: True
    )

    def _ok_rebind(*_a, **_k):
        sim._latent_scene.release_hard_lock()
        return True

    monkeypatch.setattr(sim, "_rebind_vision_objectness_peak", _ok_rebind)
    events: list[dict] = []
    monkeypatch.setattr(
        "engine.features.simulation.mixin_grounded_language.task_log_event",
        lambda event, **fields: events.append({"event": event, **fields}),
    )

    sim._maybe_correct_vision_range_mismatch(
        1200, oracle_dist=1.4, kind="approach"
    )

    assert sim._latent_scene.hard_lock_active is False
    assert sim._vision_floor_lock_reject_streak == 0
    row = next(e for e in events if e["event"] == "vision_range_correct")
    assert row.get("rebind_ok") is True
    assert row.get("force_unlock") is False


def test_soft_fix_without_rebind_keeps_lock(
    vision_on: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If soft refresh brings range close enough, do not rebind or unlock."""
    sim = _RangeCorrectSim()
    _seed_locked(sim, range_m=3.5)
    sim._vision_range_mismatch_streak = 10
    rebind_calls = {"n": 0}

    def _refresh(*_a, **_k):
        act = sim._latent_scene.active()
        assert act is not None
        # Soft-correct into acceptable band vs od=1.4 → threshold 1.4*1.40+0.15≈2.11
        act.range_m = 1.8
        return True

    monkeypatch.setattr(sim._latent_scene, "refresh_active_from_live_camera", _refresh)

    def _rebind(*_a, **_k):
        rebind_calls["n"] += 1
        return False

    monkeypatch.setattr(sim, "_rebind_vision_objectness_peak", _rebind)

    sim._maybe_correct_vision_range_mismatch(
        1300, oracle_dist=1.4, kind="approach"
    )

    assert rebind_calls["n"] == 0
    assert sim._latent_scene.hard_lock_active is True


def test_escalate_passes_allow_full_resolve_after_rejects(
    vision_on: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    sim = _RangeCorrectSim()
    _seed_locked(sim, range_m=3.5)
    sim._vision_range_mismatch_streak = 10
    sim._vision_floor_lock_reject_streak = 2  # escalate_after=2 → allow_full=True
    seen: dict[str, bool] = {}

    monkeypatch.setattr(
        sim._latent_scene, "refresh_active_from_live_camera", lambda *a, **k: True
    )

    def _rebind(*_a, **kwargs):
        seen["allow_full"] = bool(kwargs.get("allow_full_resolve"))
        return False

    monkeypatch.setattr(sim, "_rebind_vision_objectness_peak", _rebind)
    monkeypatch.setattr(
        "engine.features.simulation.mixin_grounded_language.task_log_event",
        lambda *_a, **_k: None,
    )

    sim._maybe_correct_vision_range_mismatch(
        1400, oracle_dist=1.4, kind="approach"
    )

    assert seen.get("allow_full") is True


def test_force_unlock_allows_later_rebind_when_object_truly_gone(
    vision_on: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Regression: real FOV loss must not eternal-stick on a dead lock.

    After reject streak force-unlocks, a later gated rebind (new peak / new object)
    can reseat the target — hard_lock was false so bind is not blocked by stale lock.
    """
    sim = _RangeCorrectSim()
    _seed_locked(sim, range_m=3.5)
    sim._vision_range_mismatch_streak = 10
    sim._vision_floor_lock_reject_streak = 2

    monkeypatch.setattr(
        sim._latent_scene, "refresh_active_from_live_camera", lambda *a, **k: True
    )
    monkeypatch.setattr(sim, "_rebind_vision_objectness_peak", lambda *a, **k: False)
    monkeypatch.setattr(
        "engine.features.simulation.mixin_grounded_language.task_log_event",
        lambda *_a, **_k: None,
    )

    sim._maybe_correct_vision_range_mismatch(
        1500, oracle_dist=1.4, kind="approach"
    )
    assert sim._latent_scene.hard_lock_active is False

    # Object "reappears" elsewhere — successful rebind under unlocked scene.
    def _ok(*_a, **_k):
        sim._latent_scene.release_hard_lock()
        vt = VisualTarget(
            slot_id="slot_9",
            u=0.62,
            v=0.41,
            label="prop",
            confidence=0.7,
            bearing=0.2,
            range_m=1.5,
            diagnostics={"geometry": "objectness_peak"},
        )
        sim._latent_scene.bind_visual_target(
            vt, tick=1501, agent_xy=(0.0, 0.0), agent_forward=(1.0, 0.0)
        )
        return True

    monkeypatch.setattr(sim, "_rebind_vision_objectness_peak", _ok)
    sim._vision_range_mismatch_streak = 10
    act = sim._latent_scene.active()
    assert act is not None
    act.range_m = 3.5
    sim._vision_range_correct_until_tick = -1

    sim._maybe_correct_vision_range_mismatch(
        1600, oracle_dist=1.4, kind="approach"
    )
    assert sim._latent_scene.hard_lock_active is True
    assert sim._latent_scene.active_ids == ["slot_9"]
