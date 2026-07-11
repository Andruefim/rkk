"""Tests for physical manipulation verification."""
from __future__ import annotations

from engine.manipulation_verify import ManipulationEpisode, verify_manipulation
from engine.object_resolver import ResolvedObject


def _chair_at(x: float, y: float) -> ResolvedObject:
    return ResolvedObject(
        ref="manip_chair_front",
        obj_id="manip_chair_front",
        body_id=42,
        semantic="chair",
        position=(x, y, 0.4),
        mass=5.5,
        movable=True,
        source="test",
    )


def test_verify_success_on_displacement() -> None:
    ep = ManipulationEpisode.begin(_chair_at(1.0, 0.0), min_displacement_m=0.12)
    out = verify_manipulation(ep, (1.25, 0.0))
    assert out["success"] is True
    assert out["moved_enough"] is True
    assert out["displacement_m"] >= 0.12
    assert out["reason"] == "displacement_ok"


def test_verify_fail_when_not_moved() -> None:
    ep = ManipulationEpisode.begin(_chair_at(1.0, 0.0), min_displacement_m=0.12)
    out = verify_manipulation(ep, (1.02, 0.01))
    assert out["success"] is False
    assert out["reason"] == "insufficient_displacement"


def test_verify_direction_projection() -> None:
    ep = ManipulationEpisode.begin(
        _chair_at(0.0, 0.0),
        min_displacement_m=0.12,
        requested_direction=(1.0, 0.0),
    )
    ok = verify_manipulation(ep, (0.2, 0.0))
    bad = verify_manipulation(ep, (-0.2, 0.0))
    assert ok["success"] is True
    assert bad["success"] is False
    assert bad["reason"] == "wrong_direction"


def test_intent_cannot_cause_success() -> None:
    ep = ManipulationEpisode.begin(_chair_at(1.0, 0.0), min_displacement_m=0.12)
    out = verify_manipulation(
        ep,
        (1.01, 0.0),
        intent_signals={"intent_grasp": 0.95, "intent_reach_right": 0.9},
        pe_success=True,
    )
    assert out["success"] is False
    assert out["intent_high"] is True
    assert out["pe_success_flag"] is True
    assert out["intent_could_not_succeed"] is True
