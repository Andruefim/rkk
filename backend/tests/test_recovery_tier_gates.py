"""Tiered fallen_override exit gates."""
from __future__ import annotations

from engine.system2.schema import EpisodeSuccessSpec
from engine.system2.success_predicates import (
    evaluate_override_recovery_exit,
    override_recovered_tier1_ok,
    override_recovered_tier2_ok,
)


def test_tier1_progress_not_tier2_stand():
    obs0 = {
        "com_z": 0.08,
        "posture_stability": 0.0,
        "foot_contact_l": 0.75,
        "foot_contact_r": 0.72,
    }
    obs1 = {
        "com_z": 0.14,
        "posture_stability": 0.08,
        "foot_contact_l": 0.76,
        "foot_contact_r": 0.74,
    }
    t1, d1 = override_recovered_tier1_ok(obs1, obs0)
    t2, d2 = override_recovered_tier2_ok(obs1, obs0)
    assert t1
    assert not t2
    assert d2.get("override_exit_block") == "posture_low"


def test_tier2_stand_after_lift():
    obs0 = {
        "com_z": 0.30,
        "posture_stability": 0.35,
        "foot_contact_l": 0.70,
        "foot_contact_r": 0.68,
    }
    obs1 = {
        "com_z": 0.42,
        "posture_stability": 0.48,
        "foot_contact_l": 0.75,
        "foot_contact_r": 0.73,
    }
    t2, _ = override_recovered_tier2_ok(obs1, obs0)
    assert t2


def test_tier1_prone_with_real_lift():
    obs0 = {
        "com_z": 0.08,
        "posture_stability": 0.0,
        "foot_contact_l": 0.75,
        "foot_contact_r": 0.72,
    }
    obs1 = {
        "com_z": 0.14,
        "posture_stability": 0.08,
        "foot_contact_l": 0.76,
        "foot_contact_r": 0.74,
    }
    t1, _ = override_recovered_tier1_ok(obs1, obs0)
    assert t1


def test_evaluate_exit_tier_order():
    obs0 = {"com_z": 0.30, "posture_stability": 0.35, "foot_contact_l": 0.7, "foot_contact_r": 0.7}
    obs1 = {"com_z": 0.42, "posture_stability": 0.48, "foot_contact_l": 0.75, "foot_contact_r": 0.73}
    tier, ok, diag = evaluate_override_recovery_exit(
        obs1, obs0, EpisodeSuccessSpec(), macro="RECOVER_POSTURE"
    )
    assert tier == 2
    assert diag.get("recover_tier") == 2
