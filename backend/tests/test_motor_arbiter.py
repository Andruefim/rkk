"""Tests for motor arbiter (Sprint 5.0)."""
from __future__ import annotations

import math

from engine.motor_arbiter import (
    MotorArbiter,
    MotorIntent,
    arbitrate,
    clamp_torso_during_reach,
    filter_human_task_targets,
    is_balance_critical_intent_field,
)


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
    # Balance fields: reflex/gait keep tier-1.0 weight; human_task balance prec is damped.
    assert merged["intent_stride"] > 0.55
    assert merged["intent_stride"] < 0.82
    assert conflicts >= 1


def test_arbitrate_human_task_wins_on_reach() -> None:
    intents = [
        MotorIntent(source="reflex", precision=0.90, reach_right=0.35),
        MotorIntent(source="human_task", precision=0.88, reach_right=0.82),
    ]
    merged, _ = arbitrate(intents, human_task_active=True, current={})
    assert merged["intent_reach_right"] > 0.70


def test_arbitrate_navigation_wins_balance_during_human_task() -> None:
    intents = [
        MotorIntent(source="s2_wm", precision=0.90, stride=0.62, coupling=0.78),
        MotorIntent(source="ns_bridge", precision=0.92, stride=0.64, coupling=0.76),
        MotorIntent(
            source="navigation",
            precision=0.88,
            stride=0.66,
            coupling=0.28,
            support_left=0.38,
            support_right=0.62,
        ),
    ]
    merged, conflicts = arbitrate(intents, human_task_active=True, current={})
    assert math.isclose(merged["intent_stride"], 0.66, abs_tol=0.01)
    assert merged["intent_gait_coupling"] < 0.35
    assert merged["intent_support_right"] > merged["intent_support_left"]
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
    assert not arb.should_suppress_stabilization()


def test_arbitrate_reflex_precision_on_balance_not_crushed() -> None:
    """Reflex tier keeps multiplier 1.0 on balance-critical fields during human task."""
    intents = [
        MotorIntent(source="reflex", precision=0.90, support_left=0.72, support_right=0.28),
        MotorIntent(source="human_task", precision=0.88, support_left=0.30, support_right=0.70),
    ]
    merged_active, _ = arbitrate(intents, human_task_active=True, current={})
    # Old ladder crushed reflex to ×0.06 → merged support_left ≈ 0.31. Reflex must contribute now.
    assert merged_active["intent_support_left"] > 0.45
    assert merged_active["intent_support_left"] < 0.72


def test_filter_human_task_targets_drops_balance_critical() -> None:
    raw = {
        "intent_reach_right": 0.82,
        "intent_stride": 0.78,
        "intent_support_left": 0.70,
        "intent_torso_forward": 0.72,
        "intent_grasp": 0.65,
    }
    filtered = filter_human_task_targets(raw)
    assert "intent_reach_right" in filtered
    assert "intent_grasp" in filtered
    assert "intent_stride" not in filtered
    assert "intent_support_left" not in filtered
    assert "intent_torso_forward" not in filtered


def test_clamp_torso_during_reach() -> None:
    raw = {
        "intent_reach_right": 0.75,
        "intent_torso_forward": 0.85,
        "intent_lean_forward": 0.80,
    }
    clamped = clamp_torso_during_reach(raw)
    assert clamped["intent_torso_forward"] <= 0.5 + 0.08 + 1e-6
    assert clamped["intent_torso_forward"] >= 0.5 - 0.08 - 1e-6
    assert clamped["intent_lean_forward"] <= 0.5 + 0.08 + 1e-6


def test_register_task_executive_skips_when_fallen() -> None:
    """Gating: fallen / low posture / motor-hold must not register human_task."""

    class _TickMixin:
        tick = 100
        _post_reset_motor_hold_until = 0
        _motor_arbiter = MotorArbiter()
        _task_binding = None
        _task_tree_ctrl = None
        _intention_state = None

        def _canonical_motor_intent_key(self, k: str) -> str:
            return str(k)

        def _env_observe_cached(self) -> dict:
            return {"posture_stability": 0.7}

    from engine.features.simulation.mixin_tick import SimulationTickMixin

    mixin = _TickMixin()
    mixin._motor_arbiter.set_human_task_active(True)
    registered: list[dict] = []
    orig = mixin._motor_arbiter.register_from_dict

    def _capture(source: str, values: dict, **kw):  # type: ignore[no-untyped-def]
        if source == "human_task":
            registered.append(dict(values))
        orig(source, values, **kw)

    mixin._motor_arbiter.register_from_dict = _capture  # type: ignore[method-assign]

    def _call(fallen: bool) -> None:
        registered.clear()
        SimulationTickMixin._register_task_executive_motor_intents(mixin, fallen=fallen)

    class _HT:
        status = "active"
        expected_state = {"intent_reach_right": 0.8, "intent_stride": 0.7}

    class _TB:
        active_task = _HT()

    mixin._task_binding = _TB()
    _call(fallen=True)
    assert registered == []

    _call(fallen=False)
    assert registered and "intent_reach_right" in registered[0]
    assert "intent_stride" not in registered[0]

    mixin._post_reset_motor_hold_until = 200
    _call(fallen=False)
    assert registered == []

    mixin._post_reset_motor_hold_until = 0
    mixin._env_observe_cached = lambda: {"posture_stability": 0.40}  # type: ignore[method-assign, assignment]
    _call(fallen=False)
    assert registered == []


def test_balance_critical_field_classification() -> None:
    assert is_balance_critical_intent_field("intent_stride")
    assert is_balance_critical_intent_field("intent_stop_recover")
    assert not is_balance_critical_intent_field("intent_reach_left")
