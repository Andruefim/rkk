"""Tests for event-driven hierarchical task tree (wave 1)."""
from __future__ import annotations

import pytest

from engine.task_goal import GoalPredicate, TaskGoal
from engine.task_tree import (
    DECOMPOSE_GENERIC,
    DECOMPOSE_MANIPULATE,
    DECOMPOSE_RECOVER,
    TaskTreeController,
)


@pytest.fixture
def ctrl() -> TaskTreeController:
    return TaskTreeController()


def test_bind_goal_contact_decomposition(ctrl: TaskTreeController) -> None:
    goal = TaskGoal(
        text="touch ball",
        predicates=[
            GoalPredicate(kind="reduce_distance", target_value=0.9, tolerance=0.25),
            GoalPredicate(kind="contact", target_value=1.0, tolerance=0.5),
        ],
        diagnostics={"needs_target": True},
    )
    tree = ctrl.bind_goal(goal, tick=1, needs_target=True, target_ref="ball")
    kinds = [tree.nodes[c].kind for c in tree.nodes[tree.root_id].children]
    assert kinds == ["resolve_target", "approach", "reach_contact"]
    approach = tree.nodes[tree.nodes[tree.root_id].children[1]]
    assert approach.expected_state.get("stop_distance") == 0.9


def test_bind_goal_displace_decomposition(ctrl: TaskTreeController) -> None:
    goal = TaskGoal(
        text="push box",
        predicates=[
            GoalPredicate(kind="reduce_distance", target_value=0.9),
            GoalPredicate(kind="displace", target_value=0.12),
        ],
        diagnostics={"needs_target": True},
    )
    tree = ctrl.bind_goal(goal, tick=2, needs_target=True)
    kinds = [tree.nodes[c].kind for c in tree.nodes[tree.root_id].children]
    assert kinds == [
        "resolve_target",
        "approach",
        "reach_target",
        "push_target",
        "verify_target",
    ]


def test_decomposition_manipulate(ctrl: TaskTreeController) -> None:
    tree = ctrl.bind_command(
        "push the box",
        tick=10,
        command_kind="manipulate_object",
        target_ref="slot_2",
        expected_state={"target_dist": 0.3},
    )
    root = tree.nodes[tree.root_id]
    kinds = [tree.nodes[cid].kind for cid in root.children]
    assert kinds == list(DECOMPOSE_MANIPULATE)
    assert ctrl.active_node is not None
    assert ctrl.active_node.kind == "resolve_target"
    assert ctrl.active_node.motor_targets == {}
    reach = tree.nodes[root.children[2]]
    assert "intent_reach_right" in reach.motor_targets
    assert "intent_grasp" in reach.motor_targets
    push = tree.nodes[root.children[3]]
    assert "intent_lean_forward" in push.motor_targets
    verify = tree.nodes[root.children[4]]
    assert verify.motor_targets == {}
    assert verify.expected_state == {"target_dist": 0.3}


def test_decomposition_recover_and_generic(ctrl: TaskTreeController) -> None:
    rec = ctrl.bind_command("stand up", tick=1, command_kind="recover")
    rec_kinds = [rec.nodes[c].kind for c in rec.nodes[rec.root_id].children]
    assert rec_kinds == list(DECOMPOSE_RECOVER)
    assert rec.nodes[rec.root_id].children[0]
    recover_step = rec.nodes[rec.nodes[rec.root_id].children[0]]
    assert "intent_stop_recover" in recover_step.motor_targets

    gen = ctrl.bind_command("explore", tick=2, command_kind="generic")
    gen_kinds = [gen.nodes[c].kind for c in gen.nodes[gen.root_id].children]
    assert gen_kinds == list(DECOMPOSE_GENERIC)


def test_deterministic_progression(ctrl: TaskTreeController) -> None:
    ctrl.bind_command("push", tick=0, command_kind="manipulate_object")
    root_children = ctrl.tree.nodes[ctrl.tree.root_id].children
    expected_kinds = [
        ctrl.tree.nodes[cid].kind for cid in root_children
    ]

    for i, kind in enumerate(expected_kinds):
        node = ctrl.active_node
        assert node is not None
        assert node.kind == kind
        if kind.startswith("verify"):
            assert node.status == "verifying"
        else:
            assert node.status == "active"
        ctrl.complete_active(tick=i + 1, diagnostics={"pe_total": 0.05})

    assert ctrl.tree.root_status == "done"
    assert ctrl.active_node is None
    assert ctrl.is_active is False


def test_retry_bound(monkeypatch: pytest.MonkeyPatch, ctrl: TaskTreeController) -> None:
    monkeypatch.setenv("RKK_TASK_REPLAN_MAX", "2")
    ctrl.bind_command("push", tick=0, command_kind="manipulate_object")
    node_id = ctrl.tree.active_node_id
    assert node_id is not None

    ctrl.fail_active(tick=1, reason="miss", retryable=True)
    assert ctrl.active_node is not None
    assert ctrl.active_node.id == node_id
    assert ctrl.active_node.attempts == 2

    ctrl.fail_active(tick=2, reason="miss", retryable=True)
    assert ctrl.active_node.attempts == 3

    ctrl.fail_active(tick=3, reason="miss", retryable=True)
    assert ctrl.tree.root_status == "failed"
    assert ctrl.active_node is None


def test_preemption_cancels_old_tree(ctrl: TaskTreeController) -> None:
    first = ctrl.bind_command("first", tick=1, command_kind="generic")
    first_id = first.session_id
    second = ctrl.bind_command("second", tick=5, command_kind="recover")
    assert second.session_id != first_id
    assert first.root_status == "cancelled"
    assert ctrl.is_active is True
    assert ctrl.tree.command_text == "second"


def test_clear_pulse_and_acknowledge(ctrl: TaskTreeController) -> None:
    ctrl.bind_command("task", tick=1, command_kind="generic")
    cleared_tree = ctrl.clear(tick=9)
    assert cleared_tree is not None
    snap = ctrl.snapshot(tick=9)
    assert snap["cleared"] is True
    assert snap["active"] is False

    snap2 = ctrl.snapshot(tick=10)
    assert snap2["cleared"] is True

    ctrl.acknowledge_clear()
    snap3 = ctrl.snapshot(tick=11)
    assert snap3["cleared"] is False
    assert snap3["active"] is False
    assert snap3["session_id"] is None
    assert ctrl.tree is None


def test_clear_preserves_completed_status_and_frontend_nodes_shape(
    ctrl: TaskTreeController,
) -> None:
    ctrl.bind_command("task", tick=1, command_kind="generic")
    while ctrl.is_active:
        ctrl.complete_active(tick=2)
    assert ctrl.tree is not None
    assert ctrl.tree.root_status == "done"

    ctrl.clear(tick=3)
    snap = ctrl.snapshot(tick=3)
    assert snap["root_status"] == "done"
    assert isinstance(snap["nodes"], list)
    assert len(snap["nodes"]) == 1
    assert snap["nodes"][0]["status"] == "done"


def test_retry_resets_stage_deadline(
    ctrl: TaskTreeController, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RKK_TASK_DEADLINE_TICKS", "20")
    ctrl.bind_command("push", tick=0, command_kind="manipulate_object")
    node = ctrl.active_node
    assert node is not None
    assert node.tick_deadline == 20

    ctrl.fail_active(tick=21, reason="timeout", retryable=True)
    retried = ctrl.active_node
    assert retried is not None
    assert retried.tick_started == 21
    assert retried.tick_deadline == 41


def test_serialization_roundtrip(ctrl: TaskTreeController) -> None:
    ctrl.bind_command(
        "move box",
        tick=3,
        command_kind="manipulate_object",
        target_ref="slot_1",
        expected_state={"slot_1": 0.8},
    )
    ctrl.complete_active(tick=4)
    payload = ctrl.to_dict()
    restored = TaskTreeController.from_dict(payload)
    assert restored.tree is not None
    assert restored.tree.session_id == ctrl.tree.session_id
    assert restored.tree.active_node_id == ctrl.tree.active_node_id
    assert restored.tree.root_status == ctrl.tree.root_status
    assert len(restored.tree.nodes) == len(ctrl.tree.nodes)
    snap_a = ctrl.snapshot(tick=5)
    snap_b = restored.snapshot(tick=5)
    assert snap_a["progress"] == snap_b["progress"]
    assert snap_a["current_node_id"] == snap_b["current_node_id"]


def test_progress_increases(ctrl: TaskTreeController) -> None:
    ctrl.bind_command("task", tick=0, command_kind="manipulate_object")
    n_steps = len(ctrl.tree.nodes[ctrl.tree.root_id].children)
    snap0 = ctrl.snapshot(tick=0)
    assert snap0["progress"] == 0.0

    for i in range(1, n_steps):
        ctrl.complete_active(tick=i)
        snap = ctrl.snapshot(tick=i)
        assert snap["progress"] == pytest.approx(i / n_steps, abs=0.01)

    ctrl.complete_active(tick=n_steps)
    assert ctrl.snapshot(tick=n_steps)["progress"] == 1.0


def test_motor_targets_helper(ctrl: TaskTreeController) -> None:
    ctrl.bind_command("push", tick=0, command_kind="manipulate_object")
    # resolve_target has no motor
    assert ctrl.motor_targets() == {}
    ctrl.complete_active(tick=1)
    # approach_target
    mt = ctrl.motor_targets()
    assert "intent_stride" in mt
    assert "intent_torso_forward" in mt
