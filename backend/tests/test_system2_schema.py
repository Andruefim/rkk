"""Smoke tests for System2 schema / student."""
from __future__ import annotations

from engine.system2.schema import GoalSpec, System2Proposal, proposal_from_dict
from engine.system2.student import choose_macro_from_obs
from engine.system2.validate import validate_proposal


def test_proposal_from_dict():
    p = proposal_from_dict(
        {
            "macro": "recover_posture",
            "goal": {"com_z_min": 0.55},
            "intent_deltas": {"intent_torso_forward": 0.04},
        }
    )
    assert p is not None
    assert p.normalized_macro() == "RECOVER_POSTURE"
    assert p.goal.com_z_min == 0.55
    assert "intent_torso_forward" in p.intent_deltas


def test_student_recover_low_com():
    m = choose_macro_from_obs({"com_z": 0.35, "posture_stability": 0.5})
    assert m == "RECOVER_POSTURE"


def test_validate_proposal_filters_intent_by_graph():
    p = System2Proposal(
        macro="EXPLORE",
        goal=GoalSpec(),
        intent_deltas={
            "intent_stride": 0.05,
            "intent_phantom": 0.99,
            "bad_key": 1.0,
        },
    )
    allowed = frozenset({"intent_stride", "com_z"})
    out = validate_proposal(p, allowed_intent_keys=allowed)
    assert out is not None
    assert set(out.intent_deltas.keys()) == {"intent_stride"}
    assert abs(out.intent_deltas["intent_stride"] - 0.05) < 1e-6


def test_validate_proposal_clips_goal():
    p = proposal_from_dict(
        {
            "macro": "idle",
            "goal": {"com_z_min": 0.01, "posture_stability_min": 0.99},
        }
    )
    out = validate_proposal(p)
    assert out is not None
    assert out.goal.com_z_min == 0.05
    assert out.goal.posture_stability_min == 0.95
