"""Genome priors: stand program, walk CPG, reflexes."""
from __future__ import annotations

from engine.genome.priors import (
    REFLEX_TABLE,
    STAND_PROGRAM,
    WALK_PROGRAM,
    apply_reflexes,
    compute_walk_residuals,
    get_stand_program,
    get_walk_program,
    walk_intents_at_tick,
    walk_phase_index,
)


def test_stand_program_phases_and_moderate_torso():
    prog = get_stand_program()
    assert len(prog) >= 5
    assert sum(int(p["ticks"]) for p in prog) >= 180
    phases = {p.get("phase") for p in prog}
    assert "tuck" in phases
    assert "release_stand" in phases
    for p in prog:
        tf = p["intents"]["intent_torso_forward"]
        assert 0.45 <= tf <= 0.58, f"torso too aggressive in {p.get('phase')}: {tf}"


def test_walk_program_alternates_support():
    prog = get_walk_program()
    assert len(prog) == 8
    left_stance = [p for p in prog if "left" in str(p.get("phase", "")) and "swing" not in str(p.get("phase", ""))]
    right_stance = [p for p in prog if "right" in str(p.get("phase", "")) and "swing" not in str(p.get("phase", ""))]
    assert left_stance and right_stance
    for p in left_stance:
        assert p["intents"]["intent_support_left"] > p["intents"]["intent_support_right"]
    for p in right_stance:
        assert p["intents"]["intent_support_right"] > p["intents"]["intent_support_left"]


def test_walk_phase_cycles():
    assert walk_phase_index(0, cycle_ticks=40) == 0
    assert walk_phase_index(39, cycle_ticks=40) == len(WALK_PROGRAM) - 1
    intents_a = walk_intents_at_tick(0, cycle_ticks=40)
    intents_b = walk_intents_at_tick(20, cycle_ticks=40)
    assert intents_a["intent_support_left"] != intents_b["intent_support_left"]


def test_reflex_low_com_z_reduces_forward_lean():
    low_com_rules = [
        r for r in REFLEX_TABLE
        if r["sensor"] == "com_z" and r["cmp"] == "lt" and r["target"] == "intent_torso_forward"
    ]
    assert low_com_rules
    assert all(r["delta"] < 0 for r in low_com_rules)


def test_apply_reflexes_clip():
    ms = {"intent_stop_recover": 0.5, "intent_torso_forward": 0.5}
    obs = {"com_z": 0.2, "torso_pitch": 0.8}
    out = apply_reflexes(obs, ms)
    assert out["intent_stop_recover"] > 0.5
    assert out["intent_torso_forward"] < 0.5


def test_compute_walk_residuals_bounded():
    cur = {k: 0.5 for k in STAND_PROGRAM[0]["intents"]}
    res = compute_walk_residuals(cur, tick=5, gain=0.2)
    assert res
    assert all(abs(v) <= 0.18 for v in res.values())
