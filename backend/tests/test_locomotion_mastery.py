"""Tests for locomotion mastery gates (Sprint 5.1 / 8.0)."""
from __future__ import annotations

from engine.locomotion_mastery import LocomotionEval, is_locomotion_mastered


def test_locomotion_not_mastered_on_place_march() -> None:
    metrics = {
        "com_x_vel_ema": 0.0004,
        "pe_fwd_ema": -0.95,
        "com_x_displacement": 0.08,
        "ticks_in_step3": 2000,
        "coupling_motor": 0.88,
        "fall_rate": 0.0,
    }
    assert not is_locomotion_mastered(metrics)


def test_locomotion_mastered_when_criteria_met() -> None:
    metrics = {
        "com_x_vel_ema": 0.012,
        "pe_fwd_ema": -0.2,
        "com_x_displacement": 0.8,
        "ticks_in_step3": 1000,
        "coupling_motor": 0.76,
        "fall_rate": 0.0,
    }
    assert is_locomotion_mastered(metrics)


def test_locomotion_eval_window() -> None:
    ev = LocomotionEval(window=20)
    for i in range(25):
        ev.record_tick(
            {
                "com_x_vel_ema": 0.01,
                "pe_fwd_ema": -0.2,
                "fall_rate": 0.0,
                "support_asymmetry": 0.15,
                "com_x": i * 0.02,
            }
        )
    r = ev.evaluate()
    assert r.passed
