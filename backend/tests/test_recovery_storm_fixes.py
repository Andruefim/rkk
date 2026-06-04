"""Unit tests for System2 recovery override behavior (post-LLM removal)."""
from __future__ import annotations

from engine.system2.controller import System2Controller
from engine.system2.recovery_schedule import (
    prepare_recovery_steps,
    validate_recovery_plan,
)


def test_defer_sim_fall_hard_reset_while_override():
    c = System2Controller()
    assert not c.defer_sim_fall_hard_reset()
    c._s2_override_active = True
    assert c.defer_sim_fall_hard_reset()


def test_learned_recovery_does_not_defer_without_override(monkeypatch):
    monkeypatch.setenv("RKK_S2_LEARNED_RECOVERY", "1")
    monkeypatch.setenv("RKK_SYSTEM2", "1")
    c = System2Controller()
    assert not c.defer_sim_fall_hard_reset()


def test_validate_recovery_plan_rejects_degenerate():
    plan_too_short = [
        {"ticks": 2, "intent_deltas": {"intent_stop_recover": 0.12}},
        {"ticks": 2, "intent_deltas": {"intent_stop_recover": 0.12}},
    ]
    ok, reason = validate_recovery_plan(plan_too_short)
    assert not ok
    assert reason == "total_too_short"

    valid = [
        {"ticks": 28, "intent_deltas": {"intent_stop_recover": 0.12}},
        {"ticks": 32, "intent_deltas": {"intent_stop_recover": 0.12}},
    ]
    ok, reason = validate_recovery_plan(valid)
    assert ok
    assert reason == ""


def test_prepare_recovery_steps_remediates_index_ticks():
    index_plan = [
        {"ticks": i, "intent_deltas": {"intent_stop_recover": 0.12}}
        for i in range(1, 7)
    ]
    ready, remediated = prepare_recovery_steps(index_plan)
    assert remediated
    assert ready
    ticks = [s["ticks"] for s in ready]
    assert min(ticks) >= 10
    assert sum(ticks) >= 60
