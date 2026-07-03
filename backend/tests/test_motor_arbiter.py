"""Tests for motor arbiter (Sprint 5.0)."""
from __future__ import annotations

from engine.motor_arbiter import MotorArbiter, MotorIntent, arbitrate


def test_arbitrate_coupling_clamp_locomote() -> None:
    intents = [
        MotorIntent(source="ns_bridge", precision=0.9, coupling=0.78, stride=0.64),
        MotorIntent(source="skill", precision=0.5, coupling=0.95, stride=0.55),
    ]
    merged, conflicts = arbitrate(intents, macro="LOCOMOTE_DELIVERY", current={})
    assert merged["intent_gait_coupling"] <= 0.78 + 0.02
    assert merged["intent_stride"] >= 0.64 - 0.01
    assert conflicts >= 1


def test_motor_arbiter_finalize_writes_state() -> None:
    class _Graph:
        nodes: dict = {}

    class _Env:
        _motor_state = {"intent_stride": 0.5, "intent_gait_coupling": 0.88}

    class _Agent:
        graph = _Graph()
        env = _Env()

    class _MotorState:
        intents: dict = {}
        support_leg = "balanced"

    class _Sim:
        agent = _Agent()
        _motor_state = _MotorState()
        _system2_last = {"macro": "LOCOMOTE_DELIVERY"}

    arb = MotorArbiter()
    arb.register(MotorIntent(source="ns_bridge", precision=0.92, coupling=0.78, stride=0.64))
    arb.finalize(_Sim())
    assert _Sim.agent.env._motor_state["intent_gait_coupling"] <= 0.78 + 0.02
    assert _Sim.agent.env._motor_state["intent_stride"] >= 0.64 - 0.01


def test_support_leg_from_intent() -> None:
    from engine.motor_arbiter import get_support_leg_signal

    assert get_support_leg_signal({"intent_support_left": 0.62, "intent_support_right": 0.38}) == "left"
    assert get_support_leg_signal({"intent_support_left": 0.5, "intent_support_right": 0.5}) == "balanced"


def test_arbitrate_human_task_priority_over_cpg() -> None:
    intents = [
        MotorIntent(source="cpg", precision=0.95, stride=0.82, coupling=0.88),
        MotorIntent(source="reflex", precision=0.90, stride=0.78),
        MotorIntent(source="human_task", precision=0.88, stride=0.42, coupling=0.52),
    ]
    merged, conflicts = arbitrate(intents, human_task_active=True, current={})
    assert merged["intent_stride"] < 0.55
    assert merged["intent_gait_coupling"] < 0.65
    assert conflicts >= 1


def test_arbitrate_s2_wm_over_intention_cortex() -> None:
    intents = [
        MotorIntent(source="intention_cortex", precision=0.72, torso_forward=0.70),
        MotorIntent(source="s2_wm", precision=0.90, torso_forward=0.48),
    ]
    merged, _ = arbitrate(intents, human_task_active=True, current={})
    assert merged["intent_torso_forward"] < 0.58


def test_motor_arbiter_human_task_mode() -> None:
    arb = MotorArbiter()
    arb.set_human_task_active(True)
    assert arb.human_task_active()
    assert arb.should_suppress_substrate()
