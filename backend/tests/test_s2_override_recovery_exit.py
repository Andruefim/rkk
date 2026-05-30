"""S2 fallen override must evaluate recovery exit while override is active."""
from __future__ import annotations

from engine.system2.controller import System2Controller
from engine.system2.success_predicates import evaluate_override_recovery_exit
from engine.system2.schema import EpisodeSuccessSpec


def test_override_recovery_exit_tier2():
    obs0 = {"com_z": 0.12, "posture_stability": 0.15, "phys_com_z": 0.12}
    obs1 = {
        "com_z": 0.82,
        "posture_stability": 0.88,
        "phys_com_z": 0.82,
        "foot_contact_l": 0.9,
        "foot_contact_r": 0.9,
    }
    tier, ok, diag = evaluate_override_recovery_exit(
        obs1, obs0, EpisodeSuccessSpec(skill_id="learned_recovery"), macro="RECOVER_POSTURE"
    )
    assert tier == 2
    assert ok
    assert diag.get("override_posture_gate")


def test_controller_evaluates_exit_while_fallen_flag_true():
    ctrl = System2Controller()
    ctrl._s2_override_active = True
    ctrl._override_start_obs_f = {"com_z": 0.12, "posture_stability": 0.15}
    obs_f = {
        "com_z": 0.84,
        "posture_stability": 0.90,
        "foot_contact_l": 0.85,
        "foot_contact_r": 0.85,
    }
    ok, pe_diag, note = ctrl._override_episode_eval(obs_f, fallen=False)
    assert note in ("recovered", "recovered_tier1")
    assert ok
    assert pe_diag.get("recover_tier", 0) >= 1
