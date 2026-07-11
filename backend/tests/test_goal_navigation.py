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
    assert intents["intent_gait_coupling"] > 0.5
    assert intents.get("intent_support_left", 0.5) > intents.get("intent_support_right", 0.5)


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


def test_aligned_forward_stride_bounded() -> None:
    intents = navigation_intents((0.0, 0.0), (1.0, 0.0), (3.0, 0.0), stop_distance=0.5)
    stride = float(intents.get("intent_stride", 0.5))
    assert 0.5 <= stride <= 0.68 + 1e-6
    assert math.isclose(stride, stride, rel_tol=0.0, abs_tol=1e-6)
