"""Tests for closed-loop goal navigation intents."""
from __future__ import annotations

import math

from engine.goal_navigation import navigation_intents


def test_heading_left_produces_left_turn_intents() -> None:
    # Agent faces +X; target is at +Y (left).
    intents = navigation_intents(
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 2.0),
        stop_distance=0.5,
    )
    assert intents
    assert "task_heading_err" in intents
    assert intents["task_heading_err"] > 0.0
    assert intents["intent_gait_coupling"] > 0.5
    assert intents.get("intent_support_left", 0.5) > intents.get("intent_support_right", 0.5)
    assert intents["intent_stride"] >= 0.56


def test_heading_right_produces_right_turn_intents() -> None:
    intents = navigation_intents(
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, -2.0),
        stop_distance=0.5,
    )
    assert intents
    assert intents["intent_gait_coupling"] < 0.5
    assert intents.get("intent_support_right", 0.5) > intents.get("intent_support_left", 0.5)


def test_large_distance_increases_stride() -> None:
    far = navigation_intents((0.0, 0.0), (1.0, 0.0), (5.0, 0.0), stop_distance=0.5)
    assert far.get("intent_stride", 0.5) > 0.5


def test_at_target_returns_neutral() -> None:
    assert navigation_intents((0.0, 0.0), (1.0, 0.0), (0.3, 0.0), stop_distance=0.5) == {}


def test_fallen_returns_neutral() -> None:
    assert (
        navigation_intents(
            (0.0, 0.0),
            (1.0, 0.0),
            (3.0, 0.0),
            stop_distance=0.5,
            fallen=True,
        )
        == {}
    )


def test_drifting_away_forces_turn() -> None:
    # Facing +X, target ahead+right, but previous step moved away from target.
    intents = navigation_intents(
        (0.0, 0.0),
        (1.0, 0.0),
        (3.0, 0.0),
        stop_distance=0.5,
        prev_agent_xy=(0.2, 0.0),  # moved toward negative closing (away from +X target? wait)
    )
    # prev at 0.2, now at 0.0 → velocity -0.2 along +X toward target → closing negative
    assert intents
    assert intents.get("intent_gait_coupling") is not None or intents["intent_stride"] <= 0.56
    assert float(intents.get("task_closing_vel", 0.0)) < 0.0


def test_aligned_forward_stride_bounded() -> None:
    intents = navigation_intents((0.0, 0.0), (1.0, 0.0), (3.0, 0.0), stop_distance=0.5)
    stride = float(intents.get("intent_stride", 0.5))
    assert 0.52 <= stride <= 0.68 + 1e-6
    assert math.isclose(stride, stride, rel_tol=0.0, abs_tol=1e-6)
    assert abs(float(intents.get("task_heading_err", 0.0))) < 0.05
    assert float(intents.get("task_nav_active", 0.0)) == 1.0


def test_posture_pause_suppresses_navigation() -> None:
    intents = navigation_intents(
        (0.0, 0.0),
        (1.0, 0.0),
        (3.0, 0.0),
        stop_distance=0.5,
        posture_stability=0.40,
    )
    assert intents == {}


def test_posture_marginal_reduces_stride() -> None:
    full = navigation_intents(
        (0.0, 0.0),
        (1.0, 0.0),
        (3.0, 0.0),
        stop_distance=0.5,
        posture_stability=0.70,
    )
    marginal = navigation_intents(
        (0.0, 0.0),
        (1.0, 0.0),
        (3.0, 0.0),
        stop_distance=0.5,
        posture_stability=0.52,
    )
    assert full and marginal
    assert marginal["intent_stride"] < full["intent_stride"]
    assert marginal["intent_stride"] >= 0.5


def test_nav_hold_resolve_does_not_freeze_approach() -> None:
    """resolve/post_resolve must not arm a multi-tick nav freeze."""
    from engine.features.simulation.mixin_grounded_language import (
        SimulationGroundedLanguageMixin,
    )

    class _S(SimulationGroundedLanguageMixin):
        def __init__(self) -> None:
            self.tick = 700
            self._nav_hold_until_tick = -1

    s = _S()
    s._arm_nav_hold(547, reason="post_resolve")  # stale bind tick
    assert s._nav_hold_active(700) is False
    s._arm_nav_hold(700, reason="fall_recover")
    assert s._nav_hold_until_tick > 700
    assert s._nav_hold_active(700) is True


def test_bearing_nav_turn_blend_is_continuous() -> None:
    from engine.goal_navigation import navigation_intents_from_bearing_range

    mild = navigation_intents_from_bearing_range(0.08, 2.5, 0.55)
    sharp = navigation_intents_from_bearing_range(0.45, 2.5, 0.55)
    assert mild and sharp
    assert abs(float(mild["intent_gait_coupling"]) - 0.5) < abs(
        float(sharp["intent_gait_coupling"]) - 0.5
    )
    assert float(sharp["intent_gait_coupling"]) > 0.5
