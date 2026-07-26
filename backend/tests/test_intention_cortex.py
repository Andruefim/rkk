"""Intention Cortex — long-horizon goal stack and executive projection."""
from __future__ import annotations

import os
from unittest import mock

import torch

from engine.causal_graph import CausalGraph
from engine.curriculum_graph import CurriculumGraph, CurriculumNode
from engine.intention_cortex import (
    IntentionCortex,
    SubGoal,
    intention_cortex_enabled,
)


class _FakeEnv:
    preset = "humanoid"

    def __init__(self) -> None:
        self._patch: dict[str, float] = {}

    def apply_self_state_patch(self, patch: dict[str, float]) -> None:
        self._patch.update(patch)


class _FakeAgent:
    def __init__(self) -> None:
        self.env = _FakeEnv()
        self.graph = CausalGraph(torch.device("cpu"))
        ids = [
            "target_dist",
            "posture_stability",
            "self_goal_active",
            "self_goal_target_dist",
            "self_attention",
            "intent_stride",
        ]
        self.graph.rebind_variables(ids, {k: 0.5 for k in ids})
        self._w_meta = None


class _FakeSim:
    def __init__(self) -> None:
        self.agent = _FakeAgent()
        self.tick = 500
        self.current_world = "humanoid"
        self._physical_curriculum = None
        self._hierarchical_graph = None


def test_intention_cortex_enabled_default() -> None:
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("RKK_INTENTION_CORTEX", None)
        assert intention_cortex_enabled()


def test_stride_macro_when_stable_posture_low_com_z() -> None:
    ic = IntentionCortex()
    ic._stack = [
        SubGoal(
            subgoal_id="walk_fast",
            var_id="intent_stride",
            target_val=0.76,
            intent_targets={
                "intent_stride": 0.76,
                "intent_torso_forward": 0.7,
            },
            tick_start=1800,
            tick_deadline=4183,
            source="curriculum_pending",
            priority=0.55,
            status="active",
        )
    ]
    sim = _FakeSim()
    obs = {"posture_stability": 0.96, "com_z": 0.444, "phys_com_z": 0.444}
    ctx = ic.tick_pre_control(sim, tick=1805, obs=obs, fallen=False)
    assert ctx.macro_hint == "LOCOMOTE_DELIVERY"


def test_project_intent_motor_raises_stride_toward_curriculum() -> None:
    ic = IntentionCortex()
    ic._stack = [
        SubGoal(
            subgoal_id="walk_fast",
            var_id="intent_stride",
            target_val=0.76,
            intent_targets={
                "intent_stride": 0.76,
                "intent_torso_forward": 0.7,
                "intent_gait_coupling": 0.92,
            },
            tick_start=100,
            tick_deadline=900,
            source="curriculum_active",
            priority=0.9,
            status="active",
        )
    ]
    sim = _FakeSim()
    import numpy as np

    sim.agent.env._motor_state = {
        "intent_stride": 0.5,
        "intent_torso_forward": 0.5,
        "intent_gait_coupling": 0.5,
    }
    applied: list[dict[str, float]] = []

    def _apply(residuals: dict[str, float]) -> None:
        applied.append(dict(residuals))
        for k, dv in residuals.items():
            sim.agent.env._motor_state[k] = float(
                np.clip(sim.agent.env._motor_state.get(k, 0.5) + dv, 0.05, 0.95)
            )

    sim.agent.env.apply_motor_intent_residuals = _apply
    obs = {"posture_stability": 0.9, "com_z": 0.44}
    ctx = ic.tick_pre_control(sim, tick=200, obs=obs, fallen=False)
    assert ctx.macro_hint == "LOCOMOTE_DELIVERY"
    assert applied, "motor residuals should be applied each tick"
    assert float(sim.agent.graph.nodes["intent_stride"]) > 0.52
    assert float(sim.agent.env._motor_state["intent_stride"]) > 0.5


def test_stack_projects_self_goal_to_graph() -> None:
    ic = IntentionCortex()
    ic._stack = [
        SubGoal(
            subgoal_id="walk_1",
            var_id="target_dist",
            target_val=0.32,
            intent_targets={"intent_stride": 0.58},
            tick_start=400,
            tick_deadline=900,
            source="curriculum_active",
            priority=0.9,
            status="active",
        )
    ]
    sim = _FakeSim()
    obs = {"posture_stability": 0.62, "com_z": 0.55, "target_dist": 0.7}
    ctx = ic.tick_pre_control(sim, tick=500, obs=obs, fallen=False)
    assert ctx.macro_hint == "LOCOMOTE_DELIVERY"
    assert float(sim.agent.graph.nodes["self_goal_active"]) > 0.7
    assert float(sim.agent.graph.nodes["self_goal_target_dist"]) < 0.4
    assert ctx.stack_depth == 1


def test_curriculum_pending_chain() -> None:
    cg = CurriculumGraph()
    cg._nodes = {
        "a": CurriculumNode(
            node_id="a",
            var_id="posture_stability",
            prerequisites=[],
            intent_targets={"posture_stability": 0.7},
            status="pending",
        ),
        "b": CurriculumNode(
            node_id="b",
            var_id="target_dist",
            prerequisites=["a"],
            intent_targets={"intent_stride": 0.6},
            status="pending",
        ),
    }
    chain = cg.pending_chain("humanoid", 4)
    assert len(chain) == 1
    cg._nodes["a"].status = "completed"
    chain2 = cg.pending_chain("humanoid", 4)
    assert len(chain2) == 1
    assert chain2[0].node_id == "b"


def test_persist_roundtrip() -> None:
    ic = IntentionCortex()
    ic._stack = [
        SubGoal(
            subgoal_id="x",
            var_id="target_dist",
            target_val=0.4,
            tick_deadline=1000,
        )
    ]
    ic._narrative_lines = ["t1: walk"]
    data = ic.to_dict()
    ic2 = IntentionCortex()
    ic2.load_dict(data)
    assert len(ic2._stack) == 1
    assert ic2._stack[0].var_id == "target_dist"
    assert ic2._narrative_lines == ["t1: walk"]


def test_coupling_subgoal_deprioritized() -> None:
    ic = IntentionCortex()
    stack = [
        SubGoal(
            subgoal_id="coupling",
            var_id="phys_intent_gait_coupling",
            target_val=0.74,
            intent_targets={"phys_intent_gait_coupling": 0.74},
            priority=0.9,
        ),
        SubGoal(
            subgoal_id="walk",
            var_id="intent_stride",
            target_val=0.64,
            intent_targets={"intent_stride": 0.64},
            priority=0.55,
        ),
    ]
    ordered = ic._prioritize_embodied_subgoals(stack)
    assert ordered[0].var_id == "intent_stride"


def test_stable_gate_locomote_when_stack_empty() -> None:
    ic = IntentionCortex()
    ic._stack = []
    sim = _FakeSim()
    obs = {
        "posture_stability": 0.95,
        "com_z": 0.44,
        "foot_contact_l": 0.65,
        "foot_contact_r": 0.64,
    }
    ctx = ic.tick_pre_control(sim, tick=900, obs=obs, fallen=False)
    assert ctx.macro_hint == "LOCOMOTE_DELIVERY"
    assert "intent_stride" in ctx.intent_residuals


def test_human_command_subgoal_macro_is_idle() -> None:
    """absorb_human_task pushes human_command; _macro_for_subgoal must not LOCOMOTE."""
    ic = IntentionCortex()
    sg = SubGoal(
        subgoal_id="human_1",
        var_id="target_dist",
        target_val=0.55,
        intent_targets={"intent_stride": 0.62},
        source="human_command",
        priority=0.92,
        status="active",
    )
    obs = {"posture_stability": 0.95, "com_z": 0.5}
    assert ic._macro_for_subgoal(sg, obs, fallen=False) == "IDLE"


def test_human_executive_suppresses_locomote_and_explore() -> None:
    """Active human task → curriculum LOCOMOTE/EXPLORE subgoals become IDLE."""

    class _Task:
        status = "active"
        expected_state = {"target_dist": 0.55}

    class _Binding:
        active_task = _Task()

    class _Sim(_FakeSim):
        def __init__(self) -> None:
            super().__init__()
            self._task_binding = _Binding()
            self._motor_arbiter = None

    ic = IntentionCortex()
    sim = _Sim()
    obs = {"posture_stability": 0.95, "com_z": 0.5}

    walk = SubGoal(
        subgoal_id="walk",
        var_id="intent_stride",
        target_val=0.7,
        intent_targets={"intent_stride": 0.7},
        source="curriculum_active",
        status="active",
    )
    reach = SubGoal(
        subgoal_id="reach",
        var_id="intent_reach_left",
        target_val=0.6,
        intent_targets={"intent_reach_left": 0.6},
        source="curriculum_active",
        status="active",
    )
    assert ic._macro_for_subgoal(walk, obs, fallen=False) == "LOCOMOTE_DELIVERY"
    assert ic._macro_for_subgoal(reach, obs, fallen=False) == "EXPLORE"
    assert ic._macro_for_subgoal(walk, obs, fallen=False, sim=sim) == "IDLE"
    assert ic._macro_for_subgoal(reach, obs, fallen=False, sim=sim) == "IDLE"

    # Empty stack + stable gate must not invent LOCOMOTE during human task.
    ic._stack = []
    ctx = ic.tick_pre_control(sim, tick=100, obs=obs, fallen=False)
    assert ctx.macro_hint == "IDLE"


def test_unrecognized_subgoal_fallback_is_idle() -> None:
    ic = IntentionCortex()
    sg = SubGoal(
        subgoal_id="weird",
        var_id="some_unknown_var",
        target_val=0.5,
        intent_targets={},
        source="curriculum_pending",
        status="active",
    )
    obs = {"posture_stability": 0.95, "com_z": 0.5}
    assert ic._macro_for_subgoal(sg, obs, fallen=False) == "IDLE"


def test_merge_deliberation_does_not_resurrect_locomote_during_human_task() -> None:
    from engine.intention_cortex import IntentionContext

    class _Latest:
        macro_hint = "LOCOMOTE_DELIVERY"
        expected_state = {}
        graph_patch = {}
        intent_residuals = {}
        narrative = ""

    class _Delib:
        def latest(self, max_age_ticks: int = 120):
            return _Latest()

    class _Task:
        status = "active"
        expected_state = {"target_dist": 0.5}

    class _Binding:
        active_task = _Task()

    class _Sim(_FakeSim):
        def __init__(self) -> None:
            super().__init__()
            self._task_binding = _Binding()
            self._deliberation = _Delib()
            self._motor_arbiter = None

    ic = IntentionCortex()
    ctx = IntentionContext(macro_hint="IDLE")
    out = ic._merge_deliberation(_Sim(), ctx, tick=10)
    assert out.macro_hint == "IDLE"