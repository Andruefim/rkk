"""Wave A+: recovery schedule parse + fallback."""
from __future__ import annotations

from engine.system2.recovery_schedule import (
    default_recovery_fallback_steps,
    enrich_recovery_steps,
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
