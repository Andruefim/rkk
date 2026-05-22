"""Wave A+: recovery schedule parse + fallback."""
from __future__ import annotations

from engine.system2.recovery_schedule import (
    default_recovery_fallback_steps,
    enrich_recovery_steps,
    llm_ticks_look_like_step_indices,
    prepare_llm_recovery_steps,
    remediate_index_ticks_recovery_plan,
)
from engine.system2.schema import parse_recovery_motor_steps


def test_parse_skips_bad_step_keeps_good():
    raw = {
        "steps": [
            {"ticks": 10, "intent_deltas": {"intent_torso_forward": 0.07}},
            "not_a_dict",
            {"ticks": 20, "intent_deltas": {"intent_stop_recover": 0.08}},
        ]
    }
    steps = parse_recovery_motor_steps(raw)
    assert steps is not None
    assert len(steps) == 2


def test_enrich_empty_deltas_from_bundle():
    steps = [{"ticks": 12, "intent_deltas": {}}]
    out = enrich_recovery_steps(steps)
    assert out[0]["intent_deltas"]


def test_fallback_has_steps():
    fb = default_recovery_fallback_steps()
    assert len(fb) >= 2
    assert fb[0]["intent_deltas"]


def test_detect_index_ticks_pattern():
    index_plan = [
        {"ticks": i, "intent_deltas": {"intent_stop_recover": 0.1}} for i in range(1, 7)
    ]
    assert llm_ticks_look_like_step_indices(index_plan)
    real_plan = [{"ticks": 28, "intent_deltas": {"intent_stop_recover": 0.1}}]
    assert not llm_ticks_look_like_step_indices(real_plan)


def test_prepare_remediates_index_plan():
    index_plan = [
        {"ticks": i, "intent_deltas": {"intent_stop_recover": 0.1}} for i in range(1, 7)
    ]
    ready, remediated = prepare_llm_recovery_steps(index_plan)
    assert remediated
    assert ready
    ticks = [s["ticks"] for s in ready]
    assert min(ticks) >= 10
    assert sum(ticks) >= 60


def test_remediate_preserves_intent_deltas():
    steps = [{"ticks": 1, "intent_deltas": {"intent_torso_forward": 0.09}}]
    out = remediate_index_ticks_recovery_plan(steps)
    assert out[0]["intent_deltas"]["intent_torso_forward"] == 0.09
    assert out[0]["ticks"] >= 10
