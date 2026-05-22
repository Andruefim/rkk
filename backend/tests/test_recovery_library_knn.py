"""RecoveryLibrary k-NN + bootstrap seed."""
from __future__ import annotations

from engine.system2.recovery_library import RecoveryLibrary


def test_library_bootstrap_lookup():
    lib = RecoveryLibrary()
    obs = {
        "phys_com_z": 0.45,
        "phys_posture_stability": 0.5,
        "phys_foot_contact_l": 0.55,
        "phys_foot_contact_r": 0.55,
        "phys_target_dist": 0.95,
    }
    hit = lib.lookup(obs)
    assert hit is not None
    steps, _, _, skill = hit
    assert len(steps) >= 2
    assert skill in ("recovery_fallback_seed", "recovery_scripted_seed")


def test_library_add_and_lookup_near():
    lib = RecoveryLibrary()
    obs0 = {
        "phys_com_z": 0.20,
        "phys_posture_stability": 0.15,
        "phys_foot_contact_l": 0.80,
        "phys_foot_contact_r": 0.78,
        "phys_target_dist": 0.95,
    }
    custom = [
        {"ticks": 30, "intent_deltas": {"intent_torso_forward": 0.07}},
        {"ticks": 25, "intent_deltas": {"intent_stop_recover": 0.05}},
    ]
    lib.add_success(obs0, custom, skill_id="test_skill")
    hit = lib.lookup(
        {
            "phys_com_z": 0.21,
            "phys_posture_stability": 0.16,
            "phys_foot_contact_l": 0.79,
            "phys_foot_contact_r": 0.77,
            "phys_target_dist": 0.95,
        }
    )
    assert hit is not None
    steps, _, _, skill = hit
    assert steps[0]["intent_deltas"].get("intent_torso_forward") == 0.07
    assert skill == "test_skill"
