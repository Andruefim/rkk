"""Tests for task-conditioned observations and manipulation control."""
from __future__ import annotations

from engine.manipulation_control import manipulation_intents
from engine.success_predicates import evaluate_goal
from engine.task_goal import GoalPredicate, TaskGoal
from engine.task_observation import (
    TASK_CONTACT,
    TASK_TARGET_DIST,
    build_task_observations,
    inject_task_observations,
    nav_stop_m,
    reach_start_m,
    task_observation_keys_for_goal,
)


def test_task_observation_keys_for_contact_goal() -> None:
    goal = TaskGoal(
        text="touch",
        predicates=[
            GoalPredicate(kind="reduce_distance", target_value=nav_stop_m()),
            GoalPredicate(kind="contact", target_value=1.0),
        ],
    )
    keys = task_observation_keys_for_goal(goal)
    assert TASK_TARGET_DIST in keys
    assert TASK_CONTACT in keys


def test_build_task_observations_distance_and_contact() -> None:
    obs = build_task_observations(
        agent_xy=(0.0, 0.0),
        target_xy=(1.0, 0.0),
        contact=1.0,
    )
    assert obs[TASK_TARGET_DIST] == 1.0
    assert obs[TASK_CONTACT] == 1.0


def test_evaluate_goal_uses_task_obs_keys() -> None:
    goal = TaskGoal(
        text="touch",
        predicates=[
            GoalPredicate(kind="reduce_distance", target_value=nav_stop_m(), tolerance=0.25),
            GoalPredicate(kind="contact", target_value=1.0, tolerance=0.5),
        ],
    )
    obs = inject_task_observations(
        {"com_x": 0.5},
        build_task_observations(
            agent_xy=(0.0, 0.0),
            target_xy=(0.4, 0.0),
            contact=1.0,
        ),
    )
    ok, score, _ = evaluate_goal(goal, obs, ctx=None)
    assert ok is True
    assert score >= 0.85


def test_manipulation_intents_geometry_side() -> None:
    intents = manipulation_intents(
        (0.0, 0.0),
        (1.0, 0.0),
        (0.5, 0.5),
        dist=reach_start_m() * 0.5,
    )
    assert intents
    assert "intent_reach_left" in intents or "intent_reach_right" in intents
    assert "intent_grasp" in intents


def test_manipulation_intents_empty_when_too_far() -> None:
    intents = manipulation_intents(
        (0.0, 0.0),
        (1.0, 0.0),
        (5.0, 0.0),
        dist=3.0,
    )
    assert intents == {}
