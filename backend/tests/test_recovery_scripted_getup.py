"""Scripted prone get-up schedule."""
from __future__ import annotations

from engine.system2.recovery_schedule import (
    prepare_scripted_getup_steps,
    recovery_scripted_enabled,
    recovery_scripted_lock_until_exhausted,
    scripted_getup_joint_targets,
    scripted_getup_phase_at,
    validate_llm_recovery_plan,
)


def test_scripted_enabled_by_default():
    assert recovery_scripted_enabled()


def test_scripted_plan_valid_and_long_enough():
    steps = prepare_scripted_getup_steps()
    ok, _ = validate_llm_recovery_plan(steps)
    assert ok
    assert len(steps) >= 5
    total = sum(int(s["ticks"]) for s in steps)
    assert total >= 180
    phases = {s.get("phase") for s in steps}
    assert "tuck" in phases
    assert "release_walk" in phases


def test_scripted_phase_and_joint_targets():
    steps = prepare_scripted_getup_steps()
    cum = []
    acc = 0
    for s in steps:
        acc += int(s["ticks"])
        cum.append(acc)
    _i, name = scripted_getup_phase_at(100, 0, cum, steps)
    assert name in ("tuck", "torso_lift", "push_up", "kneel", "release_walk")
    jt = scripted_getup_joint_targets(name)
    assert "spine_pitch" in jt
    assert "lhip" in jt


def test_scripted_lock_default_on():
    assert recovery_scripted_lock_until_exhausted()


def test_scripted_caps_stop_recover_delta():
    steps = prepare_scripted_getup_steps()
    for st in steps:
        sr = (st.get("intent_deltas") or {}).get("intent_stop_recover")
        if sr is not None and sr > 0:
            assert float(sr) <= 0.22
