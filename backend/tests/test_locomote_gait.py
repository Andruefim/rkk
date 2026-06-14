"""Alternating support for anthropomorphic gait."""
from __future__ import annotations

from engine.locomote_gait import alternating_support_from_swings


def test_alternating_support_opposes_swings() -> None:
    sup_l, sup_r = alternating_support_from_swings(1.0, 0.0, amp=0.22)
    assert sup_l < 0.5
    assert sup_r > 0.5
    assert abs((sup_l + sup_r) - 1.0) < 0.02


def test_alternating_support_swapped_when_right_swings() -> None:
    sup_l, sup_r = alternating_support_from_swings(0.0, 1.0, amp=0.22)
    assert sup_l > 0.5
    assert sup_r < 0.5
