"""Wave 2: PE-based episode success, detector_id, curriculum spec bridge."""
from __future__ import annotations

from unittest.mock import MagicMock

from engine.llm_curriculum import CurriculumStage
from engine.system2.schema import EpisodeSuccessSpec, merge_episode_success_specs
from engine.system2.success_predicates import (
    build_s2_detector_id,
    curriculum_stage_to_spec,
    evaluate_macro_success,
    prediction_error_total,
    resolve_max_prediction_error,
    should_attach_curriculum_pe_spec,
)


def test_homeostatic_veto_blocks_success():
    from engine.system2.success_predicates import homeostatic_veto

    ok, reason = homeostatic_veto({"intero_stress": 0.99, "intero_energy": 0.5})
    assert ok is False
    assert reason == "intero_stress"


def test_evaluate_macro_success_pe_threshold():
    spec = EpisodeSuccessSpec(
        expected_state={"com_z": 0.5},
        max_prediction_error=0.05,
        skill_id="t",
    )
    ok, diag = evaluate_macro_success(
        {"com_z": 0.52, "intero_stress": 0.1, "intero_energy": 0.5},
        spec,
        macro="EXPLORE",
    )
    assert ok is True
    assert diag["pe_total"] <= diag["max_pe"]


def test_evaluate_macro_success_veto_wins_over_low_pe():
    spec = EpisodeSuccessSpec(
        expected_state={"com_z": 0.5},
        max_prediction_error=10.0,
    )
    ok, diag = evaluate_macro_success(
        {"com_z": 0.5, "intero_stress": 0.99, "intero_energy": 0.5},
        spec,
        macro="IDLE",
    )
    assert ok is False
    assert diag["homeo_veto"] is True


def test_prediction_error_missing_key_uses_penalty(monkeypatch):
    monkeypatch.setenv("RKK_S2_PE_MISSING_KEY_PENALTY", "2.0")
    pe = prediction_error_total(
        {"com_z": 0.5},
        {"com_z": 0.5, "posture_stability": 0.7},
    )
    assert pe >= 2.0


def test_resolve_max_prediction_error_llm_vs_fallback():
    assert resolve_max_prediction_error(0.5, n_keys=3, macro="IDLE", skill_id=None) == 0.5
    fb = resolve_max_prediction_error(None, n_keys=2, macro="IDLE", skill_id=None)
    assert 0.05 < fb < 2.0


def test_build_s2_detector_id_sorted_keys():
    d1 = build_s2_detector_id("EXPLORE", "sk", {"b": 1.0, "a": 0.0})
    d2 = build_s2_detector_id("EXPLORE", "sk", {"a": 0.0, "b": 1.0})
    assert d1 == d2
    assert d1.startswith("system2:sk:a_")


def test_merge_episode_success_specs_curriculum_fills_empty():
    llm = EpisodeSuccessSpec(expected_state={"com_z": 0.55}, max_prediction_error=0.1)
    cur = EpisodeSuccessSpec(
        expected_state={"posture_stability": 0.6},
        max_prediction_error=0.2,
        skill_id="stage1",
    )
    m = merge_episode_success_specs(llm, cur)
    assert "com_z" in m.expected_state and "posture_stability" in m.expected_state
    assert m.max_prediction_error == 0.1

    only_cur = merge_episode_success_specs(EpisodeSuccessSpec(), cur)
    assert only_cur.expected_state == cur.expected_state


def test_curriculum_stage_to_spec():
    st = CurriculumStage(
        stage_id=1,
        name="n",
        description="d",
        intent_targets={},
        advance_conditions={"posture_stability": 0.5},
        s2_expected_state={"com_z": 0.5},
        s2_max_prediction_error=0.2,
        s2_skill_id="skill_a",
    )
    sp = curriculum_stage_to_spec(st)
    assert sp.skill_id == "skill_a"
    assert sp.expected_state.get("com_z") == 0.5


def test_should_attach_curriculum_pe_only_idle():
    gov = EpisodeSuccessSpec(expected_state={"posture_stability": 0.7})
    assert should_attach_curriculum_pe_spec("IDLE", gov) is True
    assert should_attach_curriculum_pe_spec("EXPLORE", gov) is False
    assert should_attach_curriculum_pe_spec("LOCOMOTE_DELIVERY", gov) is False
    assert should_attach_curriculum_pe_spec("RECOVER_POSTURE", gov) is False
    assert should_attach_curriculum_pe_spec("IDLE", EpisodeSuccessSpec()) is False


def test_curriculum_stage_mock_for_controller_merge():
    stage = MagicMock()
    stage.s2_expected_state = {"foot_contact_l": 0.8}
    stage.s2_max_prediction_error = 0.25
    stage.s2_skill_id = "mock"
    sp = curriculum_stage_to_spec(stage)
    assert sp.expected_state["foot_contact_l"] == 0.8
