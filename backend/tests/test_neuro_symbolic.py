"""Neuro-Symbolic Layer 3 bridge + Layer 4 engine."""
from __future__ import annotations

import os
from unittest import mock

import torch

from engine.neuro_symbolic.bridge import NeuroSymbolicBridge, neuro_symbolic_enabled
from engine.neuro_symbolic.engine import SymbolicCognitiveEngine
from engine.neuro_symbolic.planner import macro_to_goal, plan_to_goal
from engine.neuro_symbolic.predicates import (
    PATH_BLOCKED_ARM_RAW,
    PATH_BLOCKED_CLEAR_RAW,
    PATH_BLOCKED_FORWARD_MAX,
    PathBlockedHysteresis,
    compute_path_blocked_confidence,
    compute_path_blocked_raw,
    embodied_var_id,
    ground_humanoid_state,
    path_forward_blocked,
)


def test_neuro_symbolic_enabled_default() -> None:
    with mock.patch.dict(os.environ, {"RKK_NEURO_SYMBOLIC": "1"}, clear=False):
        assert neuro_symbolic_enabled()


def test_ground_humanoid_standing() -> None:
    st = ground_humanoid_state(
        {
            "posture_stability": 0.85,
            "com_z": 0.55,
            "intent_stride": 0.62,
            "self_goal_active": 0.8,
        }
    )
    assert st.best("IsStable") > 0.7
    assert st.best("StrideHigh") > 0.3


def test_plan_locomote_from_stable() -> None:
    st = ground_humanoid_state(
        {
            "posture_stability": 0.9,
            "com_z": 0.55,
            "intent_stride": 0.5,
            "foot_contact_l": 0.7,
            "foot_contact_r": 0.7,
            "self_goal_active": 0.85,
        }
    )
    path = plan_to_goal(st, {"StrideHigh": 0.5})
    assert path
    assert path[0].name in ("StepForward", "ApproachTarget", "ApproachObject")


def test_bridge_priors_for_locomote() -> None:
    bridge = NeuroSymbolicBridge()
    ctx = bridge.priors_for_active_inference(
        "LOCOMOTE_DELIVERY",
        {"posture_stability": 0.88, "com_z": 0.52, "self_goal_active": 0.9},
        {"intent_stride": 0.5, "intent_gait_coupling": 0.5},
    )
    assert ctx.plan_steps
    assert ctx.motor_priors.get("intent_stride", 0) > 0.55


def test_engine_veto_fallen_stride() -> None:
    eng = SymbolicCognitiveEngine()
    st = ground_humanoid_state({"posture_stability": 0.1, "com_z": 0.2, "intent_stride": 0.8})
    veto = eng.check_fuzzy_safety(st)
    assert not veto.allowed


def test_human_proximity_veto() -> None:
    eng = SymbolicCognitiveEngine()
    eng.set_distance_to_human(0.5)
    veto = eng.check_human_proximity({"distance_to_human": 0.5})
    assert not veto.allowed
    assert veto.hard_veto
    eng.set_distance_to_human(0.95)
    assert eng.check_human_proximity({"distance_to_human": 0.95}).allowed


def test_motor_sync_locomote_defaults() -> None:
    from engine.neuro_symbolic.motor_sync import collect_motor_targets

    class _IC:
        macro_hint = "LOCOMOTE_DELIVERY"
        primary = None
        intent_residuals = {}

    class _Agent:
        graph = type("G", (), {"nodes": {"intent_stride": 0.5}})()

    class _Sim:
        agent = _Agent()
        _ns_last_ctx = {"motor_priors": {"intent_stride": 0.64}}
        _intention_state = _IC()
        _system2_last = {}

    t = collect_motor_targets(_Sim())
    assert t.get("intent_stride", 0) >= 0.64


def test_motor_sync_human_veto_zeros_intent() -> None:
    from engine.neuro_symbolic.engine import SymbolicCognitiveEngine
    from engine.neuro_symbolic.motor_sync import sync_ns_motor_every_tick

    class _Agent:
        graph = type("G", (), {"nodes": {"intent_stride": 0.64}})()
        env = type("E", (), {"_motor_state": {"intent_stride": 0.64}})()

    eng = SymbolicCognitiveEngine()
    eng.set_distance_to_human(0.5)

    class _Sim:
        agent = _Agent()
        _ns_engine = eng
        _ns_bridge = None
        _intention_state = None
        _system2_last = {}
        _ns_last_ctx = {"motor_priors": {"intent_stride": 0.64}}

        def _graph_vec_cached(self):
            return {}

    sync_ns_motor_every_tick(_Sim())
    assert _Sim.agent.graph.nodes["intent_stride"] == 0.5
    assert _Sim.agent.env._motor_state["intent_stride"] == 0.5


def test_embodied_var_filter() -> None:
    assert embodied_var_id("intent_stride")
    assert embodied_var_id("target_dist")
    assert not embodied_var_id("l1_left_leg_lhip_std")
    assert not embodied_var_id("gait_phase_r")


def test_path_blocked_from_forward_pe() -> None:
    conf = compute_path_blocked_confidence(
        {"intent_stride": 0.64, "hai_pe_fwd_ema": -0.95, "visual_depth": 0.1},
        {"active_skill": "step_forward_L", "actual_delta": 0.0},
    )
    assert conf >= 0.65


def test_path_blocked_not_from_locomote_macro_alone() -> None:
    """Turn / LOCOMOTE without step_forward must not latch PathBlocked on PE alone."""
    conf = compute_path_blocked_raw(
        {"hai_pe_fwd_ema": -0.95, "intent_stride": 0.64},
        {"macro_hint": "LOCOMOTE_DELIVERY", "actual_delta": 0.0, "ns_plan_head": "Turn"},
    )
    assert conf < PATH_BLOCKED_FORWARD_MAX


def test_path_blocked_ignored_during_turn_plan() -> None:
    conf = compute_path_blocked_raw(
        {"hai_pe_fwd_ema": -0.95},
        {"ns_plan_head": "Turn", "actual_delta": 0.0},
    )
    assert conf == 0.0


def test_path_blocked_hysteresis_arms_and_clears() -> None:
    h = PathBlockedHysteresis()
    mid = h.update(0.5)
    assert mid <= PATH_BLOCKED_FORWARD_MAX
    high = h.update(0.7)
    assert high >= PATH_BLOCKED_ARM_RAW
    hold = h.update(0.5)
    assert hold > PATH_BLOCKED_FORWARD_MAX
    low = h.update(0.2)
    assert low < PATH_BLOCKED_CLEAR_RAW


def test_path_blocked_sigmoid_above_dead_zone() -> None:
    conf = compute_path_blocked_confidence(
        {"hai_pe_fwd_ema": -0.91, "visual_depth": 0.12},
        {"active_skill": "step_forward_L", "actual_delta": 0.0},
    )
    assert conf >= 0.65
    assert path_forward_blocked(conf)


def test_fallback_turn_not_step_forward_when_blocked() -> None:
    bridge = NeuroSymbolicBridge()
    st = ground_humanoid_state(
        {
            "posture_stability": 0.9,
            "com_z": 0.55,
            "foot_contact_l": 0.8,
            "foot_contact_r": 0.8,
            "hai_pe_fwd_ema": -0.91,
            "intent_stride": 0.64,
            "visual_depth": 0.1,
        },
        context={
            "active_skill": "step_forward_L",
            "actual_delta": 0.0,
            "_path_blocked_hysteresis": bridge._path_blocked_hyst,
        },
    )
    assert st.best("PathBlocked") > PATH_BLOCKED_FORWARD_MAX
    act = bridge._fallback_action("LOCOMOTE_DELIVERY", st)
    assert act is not None
    assert act.name == "Turn"


def test_path_blocked_from_visual_depth() -> None:
    conf = compute_path_blocked_confidence(
        {"visual_depth": 0.15},
        {"actual_delta": 0.0},
    )
    assert conf >= 0.5


def test_ground_humanoid_path_blocked_fact() -> None:
    h = PathBlockedHysteresis()
    st = ground_humanoid_state(
        {
            "posture_stability": 0.9,
            "com_z": 0.55,
            "hai_pe_fwd_ema": -0.92,
            "visual_depth": 0.1,
        },
        {"intent_stride": 0.64},
        context={
            "active_skill": "step_forward_R",
            "actual_delta": 0.0,
            "_path_blocked_hysteresis": h,
        },
    )
    assert st.best("PathBlocked") >= 0.65


def test_plan_turn_when_path_blocked() -> None:
    h = PathBlockedHysteresis()
    st = ground_humanoid_state(
        {
            "posture_stability": 0.9,
            "com_z": 0.55,
            "intent_stride": 0.64,
            "foot_contact_l": 0.8,
            "foot_contact_r": 0.8,
            "self_goal_active": 0.85,
            "hai_pe_fwd_ema": -0.95,
            "visual_depth": 0.1,
        },
        context={
            "active_skill": "step_forward_L",
            "actual_delta": 0.0,
            "_path_blocked_hysteresis": h,
        },
    )
    assert st.best("PathBlocked") > 0.55
    path = plan_to_goal(st, {"StrideHigh": 0.5})
    assert path
    assert path[0].name == "Turn"


def test_bridge_replan_invalidates_forward_plan() -> None:
    bridge = NeuroSymbolicBridge()
    ctx = bridge.priors_for_active_inference(
        "LOCOMOTE_DELIVERY",
        {
            "posture_stability": 0.9,
            "com_z": 0.52,
            "foot_contact_l": 0.8,
            "foot_contact_r": 0.8,
            "self_goal_active": 0.9,
            "hai_pe_fwd_ema": -0.95,
            "intent_stride": 0.64,
            "visual_depth": 0.1,
        },
        {"intent_stride": 0.64},
    )
    # Without step_forward skill + physical cue, should not stay on Turn forever
    assert ctx.plan_steps
    assert ctx.facts.get("PathBlocked", 0) <= PATH_BLOCKED_FORWARD_MAX or ctx.plan_steps[0] == "Turn"
