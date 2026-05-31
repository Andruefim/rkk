"""LLM recovery delta sanitization and prone tier1 gates."""
from __future__ import annotations

from engine.system2.recovery_schedule import (
    prepare_recovery_steps,
    sanitize_recovery_intent_deltas,
)
from engine.system2.success_predicates import override_recovered_tier1_ok


def test_sanitize_caps_stop_recover_delta():
    d = sanitize_recovery_intent_deltas(
        {"intent_stop_recover": 0.65, "intent_torso_forward": 0.35}
    )
    assert d["intent_stop_recover"] <= 0.22
    assert d["intent_torso_forward"] == 0.35


def test_prepare_llm_sanitizes_steps():
    steps, _ = prepare_recovery_steps(
        [
            {"ticks": 30, "intent_deltas": {"intent_stop_recover": 0.65}},
            {"ticks": 25, "intent_deltas": {"intent_torso_forward": 0.2}},
            {"ticks": 20, "intent_deltas": {"intent_support_left": 0.15}},
        ]
    )
    assert steps
    assert steps[0]["intent_deltas"]["intent_stop_recover"] <= 0.22


def test_tier1_prone_requires_com_z_lift_not_posture_only():
    obs0 = {
        "com_z": 0.05,
        "posture_stability": 0.0,
        "foot_contact_l": 0.75,
        "foot_contact_r": 0.72,
    }
    obs_posture_only = {
        "com_z": 0.06,
        "posture_stability": 0.99,
        "foot_contact_l": 0.76,
        "foot_contact_r": 0.74,
    }
    ok, _ = override_recovered_tier1_ok(obs_posture_only, obs0)
    assert not ok
