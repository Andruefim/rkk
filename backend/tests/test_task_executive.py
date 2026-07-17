"""Tests for stage-gated task executive motor filtering."""
from __future__ import annotations

from engine.task_executive import (
    filter_motor_targets_for_stage,
    human_task_suppresses_autonomous_locomotion,
    human_task_suppresses_s2_locomote,
    intent_allowed_for_stage,
    motor_for_stage,
)
from engine.task_goal import GoalPredicate, TaskGoal


def test_reach_blocked_during_approach() -> None:
    assert not intent_allowed_for_stage("intent_reach_left", "approach")
    assert intent_allowed_for_stage("intent_reach_left", "reach_contact")
    out = filter_motor_targets_for_stage(
        {"intent_reach_left": 0.62, "intent_grasp": 0.45, "intent_stride": 0.62},
        "approach",
    )
    assert out == {}


def test_motor_for_stage_approach_only_stride() -> None:
    goal = TaskGoal(
        text="touch",
        predicates=[
            GoalPredicate(kind="reduce_distance", target_value=0.55),
            GoalPredicate(kind="contact", target_value=1.0),
        ],
    )
    m = motor_for_stage(
        goal,
        "approach",
        agent_xy=(0.0, 0.0),
        target_xy=(1.0, 0.0),
        agent_forward=(1.0, 0.0),
    )
    assert "intent_stride" in m
    assert "intent_reach_left" not in m


def test_motor_for_stage_reach_contact() -> None:
    goal = TaskGoal(
        text="touch",
        predicates=[
            GoalPredicate(kind="reduce_distance", target_value=0.55),
            GoalPredicate(kind="contact", target_value=1.0),
        ],
    )
    m = motor_for_stage(
        goal,
        "reach_contact",
        agent_xy=(0.0, 0.0),
        target_xy=(0.5, 0.2),
        agent_forward=(1.0, 0.0),
    )
    assert "intent_reach_left" in m or "intent_reach_right" in m
    assert "intent_stride" not in m


def test_human_task_suppresses_locomotion_on_approach() -> None:
    class _Node:
        kind = "approach"

    class _Tree:
        active_node = _Node()

    class _Task:
        status = "active"

    class _Binding:
        active_task = _Task()

    class _Sim:
        _task_binding = _Binding()
        _task_tree_ctrl = _Tree()
        _motor_arbiter = None

    assert human_task_suppresses_autonomous_locomotion(_Sim())
    assert human_task_suppresses_s2_locomote(_Sim())
