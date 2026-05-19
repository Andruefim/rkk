"""Unit tests for Recovery LLM Replan Storm fixes."""
from __future__ import annotations

import pytest
from concurrent.futures import Future
from engine.system2.controller import (
    System2Controller,
    _recovery_llm_executor,
    _recovery_llm_max_ingest_delay_ticks,
)


def test_recovery_llm_executor_creation():
    exec1 = _recovery_llm_executor()
    exec2 = _recovery_llm_executor()
    assert exec1 is exec2
    assert exec1._thread_name_prefix == "rkk_s2_recovery_llm"


def test_recovery_llm_max_ingest_delay_ticks():
    assert _recovery_llm_max_ingest_delay_ticks() == 90


def test_reject_degenerate_plan():
    controller = System2Controller()
    
    # Empty plan -> Reject
    assert not controller._ingest_recovery_pack(None, sim_tick=100)
    
    # Degenerate: cumulative ticks <= 5 -> Reject
    plan_too_short = (
        [
            {"ticks": 2, "intent_deltas": {"intent_stop_recover": 0.12}},
            {"ticks": 2, "intent_deltas": {"intent_stop_recover": 0.12}},
        ],
        {"com_z": 0.55},
        0.2,
    )
    assert not controller._ingest_recovery_pack(plan_too_short, sim_tick=100)

    # Degenerate: all steps <= 2 ticks -> Reject
    plan_all_steps_short = (
        [
            {"ticks": 2, "intent_deltas": {"intent_stop_recover": 0.12}},
            {"ticks": 1, "intent_deltas": {"intent_stop_recover": 0.12}},
            {"ticks": 2, "intent_deltas": {"intent_stop_recover": 0.12}},
        ],
        {"com_z": 0.55},
        0.2,
    )
    assert not controller._ingest_recovery_pack(plan_all_steps_short, sim_tick=100)

    # Valid: ticks > 5 and not all steps <= 2 -> Ingest successfully
    valid_plan = (
        [
            {"ticks": 10, "intent_deltas": {"intent_stop_recover": 0.12}},
            {"ticks": 20, "intent_deltas": {"intent_stop_recover": 0.12}},
        ],
        {"com_z": 0.55},
        0.2,
    )
    assert controller._ingest_recovery_pack(valid_plan, sim_tick=100)
    assert len(controller._recovery_steps) == 2
    assert controller._recovery_schedule_source == "llm"


def test_late_response_filtration():
    controller = System2Controller()
    
    # Dispatch tick: 100, current tick: 200 (elapsed: 100 ticks > max delay 90) -> Reject
    controller._recovery_llm_dispatch_tick = 100
    fut = Future()
    fut.set_result((
        [
            {"ticks": 10, "intent_deltas": {"intent_stop_recover": 0.12}},
            {"ticks": 20, "intent_deltas": {"intent_stop_recover": 0.12}},
        ],
        {"com_z": 0.55},
        0.2,
    ))
    controller._recovery_llm_future = fut
    
    # Drain at sim_tick = 200
    controller._drain_recovery_llm_future(sim_tick=200)
    
    # Future should be cleared and the error set to "too_late"
    assert controller._recovery_llm_future is None
    assert controller._last_recovery_llm_error == "too_late"
    assert controller._recovery_llm_fail_count == 1
    
    # Dispatch tick: 150, current tick: 200 (elapsed: 50 ticks < max delay 90) -> Accept
    controller._recovery_llm_dispatch_tick = 150
    fut2 = Future()
    fut2.set_result((
        [
            {"ticks": 10, "intent_deltas": {"intent_stop_recover": 0.12}},
            {"ticks": 20, "intent_deltas": {"intent_stop_recover": 0.12}},
        ],
        {"com_z": 0.55},
        0.2,
    ))
    controller._recovery_llm_future = fut2
    
    controller._drain_recovery_llm_future(sim_tick=200)
    assert controller._recovery_llm_future is None
    assert controller._last_recovery_llm_error == ""
    assert controller._recovery_schedule_source == "llm"
