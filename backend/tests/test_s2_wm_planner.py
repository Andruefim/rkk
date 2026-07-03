"""S2-gated WM planner scoring and gating."""
from __future__ import annotations

import pytest

from engine.system2.controller import (
    System2Controller,
    ensure_sim_system2,
    system2_enabled,
    write_human_command_wm,
)
from engine.system2.wm_planner import (
    S2WmTask,
    _bundle_fallback_quick,
    score_wm_trajectory,
    s2_wm_fast_override_enabled,
    s2_wm_gate_strict,
    s2_wm_max_graph_d,
    s2_wm_planner_enabled,
    task_from_planning_context,
)
from engine.working_memory import WorkingMemoryBuffer


def test_recover_penalizes_stride_when_fallen():
    s0 = {
        "posture_stability": 0.3,
        "com_z": 0.35,
        "intero_energy": 0.6,
        "target_dist": 0.7,
    }
    s1 = dict(s0)
    task = S2WmTask(macro="RECOVER_POSTURE", fallen=True, fallen_override=True)
    torso = score_wm_trajectory(
        s0, s1, task, action_var="intent_torso_forward", action_val=0.72
    )
    stride = score_wm_trajectory(
        s0, s1, task, action_var="intent_stride", action_val=0.62
    )
    assert torso > stride


def test_task_active_on_fallen_override():
    ctx = {"macro": "IDLE", "fallen_override_active": True, "fallen": True}
    t = task_from_planning_context(ctx, {"self_goal_active": 0.2})
    assert t.active
    assert t.macro == "RECOVER_POSTURE"


def test_planner_enabled_by_default():
    assert s2_wm_planner_enabled()
    assert not s2_wm_fast_override_enabled()


def test_bundle_fallback_quick_from_context():
    class _Agent:
        def _features_for_intervention_pair(self, a, b):
            return [0.1, 0.2]

    ctx = {
        "bundle_candidate": {
            "variable": "intent_torso_forward",
            "value": 0.68,
        }
    }
    task = S2WmTask(macro="RECOVER_POSTURE", fallen_override=True)
    cand = _bundle_fallback_quick(ctx, task, _Agent())
    assert cand is not None
    assert cand["variable"] == "intent_torso_forward"
    assert cand.get("s2_wm_fast_override") is True


def test_bundle_fallback_prefers_recovery_schedule_candidate():
    class _Agent:
        def _features_for_intervention_pair(self, a, b):
            return [0.1]

    ctx = {
        "bundle_candidate": {"variable": "intent_torso_forward", "value": 0.68},
        "recovery_schedule_candidate": {
            "variable": "intent_stop_recover",
            "value": 0.72,
            "target": "posture_stability",
        },
    }
    task = S2WmTask(macro="RECOVER_POSTURE", fallen_override=True)
    cand = _bundle_fallback_quick(ctx, task, _Agent())
    assert cand is not None
    assert cand["variable"] == "intent_stop_recover"


def test_recover_improves_posture_score():
    s0 = {"posture_stability": 0.2, "com_z": 0.3, "intero_energy": 0.7}
    s1 = {"posture_stability": 0.35, "com_z": 0.38, "intero_energy": 0.68}
    task = S2WmTask(macro="RECOVER_POSTURE", fallen_override=True)
    sc = score_wm_trajectory(
        s0, s1, task, action_var="intent_torso_forward", action_val=0.72
    )
    assert sc > 0


def test_system2_enabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RKK_SYSTEM2", raising=False)
    assert system2_enabled()


def test_s2_wm_gate_strict_with_task_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RKK_S2_WM_GATE_STRICT", "0")
    monkeypatch.setenv("RKK_TASK_BINDING", "1")
    assert s2_wm_gate_strict()
    monkeypatch.setenv("RKK_TASK_BINDING", "0")
    assert not s2_wm_gate_strict()


def test_human_command_task_active() -> None:
    ctx = {
        "human_task_active": True,
        "macro": "EXPLORE",
        "expected_state": {"slot_0": 0.7},
        "skill_id": "human_command",
    }
    t = task_from_planning_context(ctx, {"self_goal_active": 0.1})
    assert t.active
    assert t.skill_id == "human_command"


def test_s2_wm_max_graph_d_higher_for_human_task() -> None:
    assert s2_wm_max_graph_d(human_task=True) >= s2_wm_max_graph_d(human_task=False)


def test_human_task_wm_pinned_on_eviction() -> None:
    wm = WorkingMemoryBuffer(capacity=3)
    wm.write("human_task_active", 1.0, text="go", tick=1)
    wm.write("a", 0.1, tick=1)
    wm.write("b", 0.2, tick=1)
    wm.write("c", 0.3, tick=1)
    assert wm.has("human_task_active")


def test_write_human_command_wm_lazy_init() -> None:
    class _Sim:
        _system2 = None

    sim = _Sim()
    write_human_command_wm(sim, "подойди к цели", tick=5)
    assert sim._system2 is not None
    assert sim._system2.working_memory.has("human_task_active")
    assert sim._system2.working_memory.read_text("human_task_active") == "подойди к цели"


def test_sync_human_task_wm_backfill() -> None:
    from engine.task_binding import HumanTask, TaskBindingController

    class _Sim:
        def __init__(self) -> None:
            self._task_binding = TaskBindingController()
            self._task_binding._active = HumanTask(
                text="wave",
                expected_state={"slot_0": 0.8},
                max_prediction_error=0.5,
                tick_started=1,
                tick_deadline=500,
            )

    s2 = System2Controller()
    sim = _Sim()
    s2._sync_human_task_wm_from_sim(sim, 10)
    assert s2.working_memory.has("human_task_active")
    assert "wave" in s2.working_memory.read_text("human_task_active")


def test_ensure_sim_system2_respects_disable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RKK_SYSTEM2", "0")

    class _Sim:
        _system2 = None

    sim = _Sim()
    assert ensure_sim_system2(sim) is None
    assert sim._system2 is None
