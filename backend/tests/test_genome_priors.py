"""Genome priors: stand program, walk CPG, physics drive."""
from __future__ import annotations

from engine.genome.priors import (
    REFLEX_TABLE,
    STAND_PROGRAM,
    WALK_PROGRAM,
    WALK_PHASE_JOINTS,
    apply_reflexes,
    genome_walk_eligible,
    genome_walk_innate_enabled,
    walk_burst_pairs,
    walk_intents_at_tick,
    walk_leg_joints_at_tick,
    walk_phase_index,
)


def test_stand_program_moderate_torso():
    for p in STAND_PROGRAM:
        tf = p["intents"]["intent_torso_forward"]
        assert 0.45 <= tf <= 0.58


def test_walk_alternates_support():
    for p in WALK_PROGRAM:
        name = str(p.get("phase", ""))
        sup_l = p["intents"]["intent_support_left"]
        sup_r = p["intents"]["intent_support_right"]
        if "left" in name and "swing" not in name:
            assert sup_l > sup_r
        if "right" in name and "swing" not in name:
            assert sup_r > sup_l


def test_walk_joints_per_phase():
    for p in WALK_PROGRAM:
        phase = str(p["phase"])
        jt = WALK_PHASE_JOINTS.get(phase, {})
        assert "lhip" in jt and "rknee" in jt


def test_walk_burst_intents_only():
    pairs = dict(walk_burst_pairs(10))
    assert "intent_stride" in pairs
    assert "lhip" not in pairs


def test_genome_walk_eligible_innate_when_stable():
    obs = {
        "com_z": 0.55,
        "posture_stability": 0.62,
        "foot_contact_l": 0.55,
        "foot_contact_r": 0.55,
    }
    assert genome_walk_innate_enabled()
    assert genome_walk_eligible(
        obs, goal_walk=False, is_fallen=False, fixed_root=False
    )


def test_genome_walk_eligible_when_forced():
    obs = {
        "com_z": 0.55,
        "posture_stability": 0.55,
        "foot_contact_l": 0.5,
        "foot_contact_r": 0.5,
    }
    import os

    os.environ["RKK_GENOME_WALK_FORCE"] = "1"
    try:
        assert genome_walk_eligible(
            obs, goal_walk=False, is_fallen=False, fixed_root=False
        )
    finally:
        os.environ.pop("RKK_GENOME_WALK_FORCE", None)


def test_reflex_low_com_z_reduces_forward_lean():
    rules = [
        r
        for r in REFLEX_TABLE
        if r["sensor"] == "com_z"
        and r["cmp"] == "lt"
        and r["target"] == "intent_torso_forward"
    ]
    assert rules and all(r["delta"] < 0 for r in rules)


def test_walk_phase_cycles():
    assert walk_phase_index(0, cycle_ticks=40) == 0
    a = walk_intents_at_tick(0, cycle_ticks=40)
    b = walk_intents_at_tick(20, cycle_ticks=40)
    assert a["intent_support_left"] != b["intent_support_left"]
    assert walk_leg_joints_at_tick(0)["lhip"] != walk_leg_joints_at_tick(20)["lhip"]
