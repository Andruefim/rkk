"""Stable locomote gate tests."""
from __future__ import annotations

from engine.locomote_gate import stable_locomote_ready
from engine.skill_library import SkillLibrary
from engine.system2.student import choose_macro_from_obs


def test_stable_locomote_ready_variant_com_z() -> None:
    obs = {
        "posture_stability": 0.95,
        "com_z": 0.44,
        "foot_contact_l": 0.65,
        "foot_contact_r": 0.64,
    }
    assert stable_locomote_ready(obs)


def test_stable_locomote_not_ready_low_posture() -> None:
    obs = {"posture_stability": 0.7, "com_z": 0.5, "foot_contact_l": 0.7, "foot_contact_r": 0.7}
    assert not stable_locomote_ready(obs)


def test_student_chooses_locomote_when_stable() -> None:
    macro = choose_macro_from_obs(
        {
            "posture_stability": 0.95,
            "com_z": 0.44,
            "foot_contact_l": 0.65,
            "foot_contact_r": 0.64,
        }
    )
    assert macro == "LOCOMOTE_DELIVERY"


def test_hold_stance_not_selected_for_walk_goal() -> None:
    lib = SkillLibrary()
    st = {
        "com_z": 0.44,
        "posture_stability": 0.95,
        "foot_contact_l": 0.65,
        "foot_contact_r": 0.64,
    }
    sk = lib.select_skill(st, "walk")
    assert sk is not None
    assert sk.name.startswith("step_forward")


def test_sticky_floor_raises_stride_from_skill_reset() -> None:
    from engine.neuro_symbolic.motor_sync import enforce_sticky_locomote_priors

    class _IC:
        macro_hint = "LOCOMOTE_DELIVERY"

    class _Agent:
        graph = type("G", (), {"nodes": {"intent_stride": 0.5, "intent_gait_coupling": 0.88}})()
        env = type("E", (), {"_motor_state": {"intent_stride": 0.5, "intent_gait_coupling": 0.88}})()

    class _Sim:
        agent = _Agent()
        _intention_state = _IC()

    sticky = enforce_sticky_locomote_priors(_Sim())
    assert sticky.get("intent_stride", 0) >= 0.64
    assert abs(sticky.get("intent_gait_coupling", 0) - 0.78) < 0.01
    assert _Sim.agent.env._motor_state["intent_stride"] >= 0.64


def test_step_forward_stride_at_least_064() -> None:
    lib = SkillLibrary()
    step_r = next(s for s in lib.skills if s.name == "step_forward_R")
    strides = [
        float(v)
        for step in step_r.action_sequence
        for k, v in (step if isinstance(step, list) else [(step[0], step[1])])
        if k == "intent_stride"
    ]
    assert strides
    assert max(strides) >= 0.64
