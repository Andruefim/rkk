"""Skill library leg alternation."""
from __future__ import annotations

from engine.skill_library import SkillLibrary


def test_select_skill_alternates_legs_after_l_dominance() -> None:
    lib = SkillLibrary()
    for sk in lib.skills:
        if sk.name == "step_forward_L":
            sk.uses = 50
            sk.success_rate = 1.0
        if sk.name == "step_forward_R":
            sk.uses = 0
            sk.success_rate = 0.5
    st = {
        "com_z": 0.55,
        "posture_stability": 0.85,
        "foot_contact_l": 0.7,
        "foot_contact_r": 0.7,
        "gait_phase_l": 0.6,
        "gait_phase_r": 0.2,
    }
    picked = lib.select_skill(st, goal="walk")
    assert picked is not None
    assert picked.name == "step_forward_R"
