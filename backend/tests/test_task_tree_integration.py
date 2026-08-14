"""Integration: task tree + command/tick loop (mock env, no PyBullet)."""
from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from engine.object_resolver import ResolvedObject
from engine.task_goal import GoalPredicate, TaskGoal
from engine.task_tree import DECOMPOSE_MANIPULATE, TaskTreeController
from engine.verbal_action import SpeechType
from tests.conftest import AgiLoopSim, _default_humanoid_obs
from tests.test_agi_human_command_loop import _patch_fallback_embed


def _chair_scene() -> dict:
    return {
        "registry": [
            {
                "ref": "manip_chair_front",
                "id": "manip_chair_front",
                "body_id": 9001,
                "semantic": "chair",
                "movable": True,
                "mass": 5.5,
                "x": 1.0,
                "y": 0.0,
                "z": 0.4,
                "source": "test",
            }
        ]
    }


def _displace_goal(text: str) -> TaskGoal:
    return TaskGoal(
        text=text,
        predicates=[
            GoalPredicate(kind="reduce_distance", target_value=0.9, tolerance=0.25),
            GoalPredicate(kind="displace", target_value=0.12, tolerance=0.05),
        ],
        diagnostics={"needs_target": True},
    )


def _bind_displace_chair_task(sim: AgiLoopSim, text: str = "push the object") -> None:
    """Bind a displace TaskGoal against the mock chair (bypasses flaky embed grounding)."""
    from engine.manipulation_verify import ManipulationEpisode

    goal = _displace_goal(text)
    sim._task_tree_kind = "goal"
    sim._task_goal = goal
    tt = sim._ensure_task_tree()
    tt.bind_goal(goal, sim.tick, needs_target=True)
    resolved = ResolvedObject(
        ref="manip_chair_front",
        obj_id="manip_chair_front",
        body_id=9001,
        semantic="chair",
        position=(1.0, 0.0, 0.4),
        mass=5.5,
        movable=True,
        source="test",
    )
    sim._apply_goal_target_ref(goal, resolved)
    sim._manip_resolved = resolved
    sim._manip_episode = ManipulationEpisode.begin(resolved, requested_direction=(1.0, 0.0))
    tt.complete_active(sim.tick)


def test_sandbox_scene_extras_reads_from_sim(agi_loop_sim: AgiLoopSim) -> None:
    """Scene extras must come from base._sim, not HumanoidEnvironment base."""
    sim = agi_loop_sim
    ball_scene = {
        "ball": {
            "x": 1.5,
            "y": 0.0,
            "z": 0.2,
            "body_id": 100,
            "movable": True,
            "semantic": "ball",
            "ref": "ball",
        },
    }
    sim.agent.env.set_scene_extras(ball_scene)
    assert not hasattr(sim.agent.env, "get_sandbox_scene_extras")
    assert sim._sandbox_scene_extras() == ball_scene


def test_manip_command_creates_predicate_tree_and_resolves(
    agi_loop_sim: AgiLoopSim, monkeypatch: pytest.MonkeyPatch
) -> None:
    sim = agi_loop_sim
    _patch_fallback_embed(sim)
    sim.agent.env.set_scene_extras(_chair_scene())
    sim.agent.env._obs["com_x"] = 0.0
    sim.agent.env._obs["com_y"] = 0.0
    monkeypatch.setattr(
        sim,
        "_ground_command_goal",
        lambda text, _gl: _displace_goal(text),
    )

    out = sim.handle_human_command("push the object")

    assert out["ok"] is True
    tree = out.get("task_tree") or {}
    assert tree.get("active") is True
    assert out.get("manipulation_target") == "manip_chair_front"
    tt = sim._ensure_task_tree()
    root = tt.tree
    assert root is not None
    kinds = [root.nodes[c].kind for c in root.nodes[root.root_id].children]
    assert kinds[0] == "resolve_target"
    assert "approach" in kinds
    assert "push_target" in kinds
    assert tt.active_node is not None
    assert tt.active_node.kind == "approach"


def test_manip_no_target_fails_and_reports(agi_loop_sim: AgiLoopSim) -> None:
    sim = agi_loop_sim
    _patch_fallback_embed(sim)
    sim.agent.env.set_scene_extras({"registry": []})
    verbal = SimpleNamespace(
        _messages=[],
        _on_message=[],
        total_messages=0,
        _last_report_tick=-1,
    )
    sim._verbal = verbal

    out = sim.handle_human_command("передвинь стул")

    assert out["ok"] is True
    assert out.get("task_binding") is False
    assert len(verbal._messages) == 1
    assert "не вижу" in verbal._messages[0].text.lower() or "не могу" in verbal._messages[0].text.lower()
    assert verbal._messages[0].speech_type == SpeechType.REPORT


def test_manip_motor_stage_progresses(agi_loop_sim: AgiLoopSim) -> None:
    sim = agi_loop_sim
    sim.agent.env.set_scene_extras(_chair_scene())
    sim.agent.env._obs["com_x"] = 0.0
    sim.agent.env._obs["com_y"] = 0.0
    _bind_displace_chair_task(sim)
    tt = sim._ensure_task_tree()
    assert tt.active_node is not None
    assert tt.active_node.kind == "approach"

    sim.agent.env._obs["com_x"] = 0.95
    sim.agent.env._obs["com_y"] = 0.0
    for tick in range(101, 104):
        sim.tick = tick
        sim._tick_human_task(fallen=False)
    assert tt.active_node is not None
    assert tt.active_node.kind == "reach_target"

    for tick in range(104, 106):
        sim.tick = tick
        sim._tick_human_task(fallen=False)
    assert tt.active_node is not None
    assert tt.active_node.kind == "push_target"


def test_manip_physical_verify_required_not_intent(agi_loop_sim: AgiLoopSim) -> None:
    sim = agi_loop_sim
    sim.agent.env.set_scene_extras(_chair_scene())
    _bind_displace_chair_task(sim)
    tt = sim._ensure_task_tree()
    assert tt.active_node is not None
    assert tt.active_node.kind == "approach"
    tt.complete_active(sim.tick)
    tt.complete_active(sim.tick + 1)
    assert tt.active_node is not None
    assert tt.active_node.kind == "push_target"

    sim.set_obs(
        {
            **_default_humanoid_obs(),
            "intent_grasp": 0.95,
            "intent_reach_right": 0.95,
            "com_x": 0.0,
            "com_y": 0.0,
        }
    )
    sim.tick = 200
    sim._tick_human_task(fallen=False)
    assert tt.is_active
    assert tt.active_node is not None
    assert tt.active_node.kind == "push_target"


def test_manip_full_success_reports_and_clears(agi_loop_sim: AgiLoopSim) -> None:
    sim = agi_loop_sim
    sim.agent.env.set_scene_extras(_chair_scene())
    verbal = SimpleNamespace(
        _messages=[],
        _on_message=[],
        total_messages=0,
        _last_report_tick=-1,
    )
    sim._verbal = verbal
    sim.grounded_lang_generate = lambda obs=None: "Готово."  # type: ignore[method-assign]

    _bind_displace_chair_task(sim)
    out = {"manipulation_target": "manip_chair_front"}
    tt = sim._ensure_task_tree()
    assert out["manipulation_target"] == "manip_chair_front"
    assert tt.tree is not None
    assert all(
        node.target_ref == "manip_chair_front"
        for node in tt.tree.nodes.values()
    )

    near_obs = {**sim._obs, "com_x": 0.95, "com_y": 0.0}
    sim.set_obs(near_obs)
    with mock.patch("engine.verbal_action.ollama_chat_speech_enabled", return_value=False):
        for tick in range(101, 130):
            sim.tick = tick
            sim._tick_human_task(fallen=False)
            if getattr(sim, "_task_tree_cleared_pending_ack", False):
                break

    assert tt.tree is not None
    assert tt.tree.root_status == "done"
    assert tt.snapshot(sim.tick)["cleared"] is True
    assert len(verbal._messages) == 1
    assert verbal._messages[0].speech_type == SpeechType.REPORT
    assert sim._manip_diag["verify"]["success"] is True
    assert sim._system2.working_memory.read("human_task_active") == 0.0

    sim.tick += 1
    sim._tick_human_task(fallen=False)
    assert tt.tree is None
    assert len(verbal._messages) == 1


def test_contact_goal_navigation_and_done(agi_loop_sim: AgiLoopSim) -> None:
    sim = agi_loop_sim
    _patch_fallback_embed(sim)
    sim.agent.env.set_scene_extras(
        {
            "ball": {
                "x": 1.0,
                "y": 0.0,
                "z": 0.25,
                "body_id": sim.agent.env._ball_body_id,
                "movable": True,
            }
        }
    )
    sim.agent.env._ball_pos = [1.0, 0.0, 0.25]
    verbal = SimpleNamespace(
        _messages=[],
        _on_message=[],
        total_messages=0,
        _last_report_tick=-1,
    )
    sim._verbal = verbal
    sim.grounded_lang_generate = lambda obs=None: "Готово."  # type: ignore[method-assign]

    from engine.task_observation import nav_stop_m, reach_start_m

    near = nav_stop_m()
    goal = TaskGoal(
        text="touch ball",
        target_ref="ball",
        predicates=[
            GoalPredicate(kind="reduce_distance", target_ref="ball", target_value=near),
            GoalPredicate(kind="contact", target_ref="ball", target_value=1.0),
        ],
        diagnostics={"needs_target": True},
    )
    tt = sim._ensure_task_tree()
    tt.bind_goal(goal, sim.tick, needs_target=True, target_ref="ball")
    sim._task_tree_kind = "goal"
    sim._task_goal = goal
    sim._manip_resolved = ResolvedObject(
        ref="ball",
        obj_id="ball",
        body_id=int(sim.agent.env._ball_body_id),
        semantic="ball",
        position=(1.0, 0.0, 0.25),
        mass=0.3,
        movable=True,
        source="ball",
    )
    tt.complete_active(sim.tick)  # resolve_target

    sim.agent.env._obs["com_x"] = 0.0
    sim.agent.env._obs["com_y"] = 0.0
    sim.tick = 110
    sim._tick_human_task(fallen=False)
    nav_intents = [
        mi
        for mi in sim._motor_arbiter._intents
        if getattr(mi, "source", "") == "navigation"
    ]
    assert nav_intents

    sim.agent.env._obs["com_x"] = 0.98
    sim.agent.env._obs["com_y"] = 0.0
    sim.agent.env._contact_flag = True
    for tick in range(111, 130):
        sim.tick = tick
        sim._tick_human_task(fallen=False)
        if tt.tree is not None and tt.tree.root_status == "done":
            break

    assert tt.tree is not None
    assert tt.active_node is None or tt.tree.root_status == "done"
    # Advance through reach_contact → verify_goal with contact satisfied.
    if tt.tree.root_status != "done":
        sim.agent.env._obs["com_x"] = near * 0.85
        for tick in range(130, 145):
            sim.tick = tick
            sim._tick_human_task(fallen=False)
            if tt.tree.root_status == "done":
                break

    assert tt.tree.root_status == "done"
    sim._maybe_finalize_task_tree(sim.tick)
    assert len(verbal._messages) == 1
    assert verbal._messages[0].speech_type == SpeechType.REPORT


def test_push_timeout_with_stationary_target_fails(
    agi_loop_sim: AgiLoopSim, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RKK_TASK_REPLAN_MAX", "0")
    sim = agi_loop_sim
    sim.agent.env.set_scene_extras(_chair_scene())
    _bind_displace_chair_task(sim)
    tt = sim._ensure_task_tree()
    tt.complete_active(sim.tick)  # approach
    tt.complete_active(sim.tick)  # reach
    assert tt.active_node is not None
    assert tt.active_node.kind == "push_target"
    tt.active_node.tick_deadline = sim.tick

    monkeypatch.setattr(
        sim.agent.env,
        "apply_manipulation_push",
        lambda *_args, **_kwargs: {"applied": False, "reason": "blocked"},
    )
    sim.tick += 1
    sim._tick_human_task(fallen=False)

    assert tt.tree is not None
    assert tt.tree.root_status == "failed"
    assert tt.snapshot(sim.tick)["cleared"] is True


def test_outcome_affect_once_and_bounded(agi_loop_sim: AgiLoopSim) -> None:
    sim = agi_loop_sim
    _patch_fallback_embed(sim)
    sim.agent.env.set_scene_extras(_chair_scene())
    sim.handle_human_command("move chair")
    tt = sim._ensure_task_tree()
    while tt.is_active and tt.active_node is not None:
        tt.complete_active(sim.tick)
        sim.tick += 1
    before_e = float(sim.agent.env._intero_state["intero_energy"])
    before_s = float(sim.agent.env._intero_state["intero_stress"])
    sim._maybe_finalize_task_tree(sim.tick)
    after_e = float(sim.agent.env._intero_state["intero_energy"])
    after_s = float(sim.agent.env._intero_state["intero_stress"])
    assert after_e - before_e <= 0.06
    assert after_s - before_s <= 0.0
    sim._maybe_finalize_task_tree(sim.tick + 1)
    assert float(sim.agent.env._intero_state["intero_energy"]) == after_e


def test_completion_clears_tb_tree_wm_and_one_report(
    agi_loop_sim: AgiLoopSim, monkeypatch: pytest.MonkeyPatch
) -> None:
    sim = agi_loop_sim
    _patch_fallback_embed(sim)
    monkeypatch.setattr(sim, "_ground_command_goal", lambda *_a, **_k: None)
    verbal = SimpleNamespace(
        _messages=[],
        _on_message=[],
        total_messages=0,
        _last_report_tick=-1,
    )
    sim._verbal = verbal
    sim.grounded_lang_generate = lambda obs=None: "Готово."  # type: ignore[method-assign]

    sim.handle_human_command("подойди ближе")
    tb = sim._ensure_task_binding()
    task = tb.active_task
    assert task is not None
    task.expected_state = {"target_dist": 0.35, "posture_stability": 0.75}
    task.max_prediction_error = 0.25
    if task.goal is not None:
        task.goal.wm_trusted = True
    match_obs = dict(sim._obs)
    for k, tgt in task.expected_state.items():
        match_obs[k] = float(tgt)
    sim.set_obs(match_obs)

    with mock.patch("engine.verbal_action.ollama_chat_speech_enabled", return_value=False):
        sim.tick = task.tick_started + 2
        sim._tick_human_task(fallen=False)
        sim.tick = task.tick_started + 3
        sim._tick_human_task(fallen=False)

    assert len(verbal._messages) == 1
    assert verbal._messages[0].speech_type == SpeechType.REPORT
    wm = sim._system2.working_memory
    assert wm.read("human_task_active") == 0.0
    assert tb.active_task is None
    tt = sim._ensure_task_tree()
    assert not tt.is_active


def test_generic_tree_verifies_task_binding_once_per_tick(
    agi_loop_sim: AgiLoopSim,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sim = agi_loop_sim
    _patch_fallback_embed(sim)
    monkeypatch.setattr(sim, "_ground_command_goal", lambda *_a, **_k: None)
    sim.handle_human_command("подойди ближе")
    tb = sim._ensure_task_binding()

    with mock.patch.object(tb, "tick_verify", wraps=tb.tick_verify) as verify:
        sim.tick += 1
        sim._tick_human_task(fallen=False)

    assert verify.call_count == 1


@pytest.mark.parametrize("val,expect", [("0", False), ("1", True)])
def test_task_tree_enabled_flag(monkeypatch: pytest.MonkeyPatch, val: str, expect: bool) -> None:
    monkeypatch.setenv("RKK_TASK_BINDING", "1")
    monkeypatch.setenv("RKK_TASK_TREE", val)
    from engine.task_tree import task_tree_enabled

    assert task_tree_enabled() is expect


def test_second_command_survives_tick_during_bind(
    agi_loop_sim: AgiLoopSim,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preempt + slow bind_goal must not REPORT fail or wipe the new tree.

    Live bug: handle_human_command cancelled the first tree, then LLM decompose
    yielded; a sim tick finalized the preempted tree as «Не удалось», armed
    pending_ack, and the next tick acknowledge_clear()'d the replacement
    before vision resolve ran.
    """
    sim = agi_loop_sim
    _patch_fallback_embed(sim)
    monkeypatch.setenv("RKK_TASK_RESOLVE", "vision")
    goal2 = TaskGoal(
        text="подойди к цилиндрическому объекту перед тобой",
        predicates=[
            GoalPredicate(kind="reduce_distance", target_value=0.55, tolerance=0.25),
        ],
        diagnostics={"needs_target": True},
    )
    monkeypatch.setattr(sim, "_ground_command_goal", lambda *_a, **_k: goal2)

    first = TaskGoal(
        text="подойди к конусовидному объекту перед тобой",
        predicates=[
            GoalPredicate(kind="reduce_distance", target_value=0.55, tolerance=0.25),
        ],
        diagnostics={"needs_target": True},
    )
    sim._task_tree_kind = "goal"
    sim._task_goal = first
    tt = sim._ensure_task_tree()
    tt.bind_goal(first, sim.tick, needs_target=True)
    tt.complete_active(sim.tick)
    assert tt.active_node is not None
    assert tt.active_node.kind == "approach"

    orig_bind = TaskTreeController.bind_goal

    def racing_bind(self, *args, **kwargs):
        sim.tick += 1
        sim._tick_human_task(fallen=False)
        return orig_bind(self, *args, **kwargs)

    monkeypatch.setattr(TaskTreeController, "bind_goal", racing_bind)

    verbal = SimpleNamespace(
        _messages=[],
        _on_message=[],
        total_messages=0,
        _last_report_tick=-1,
    )
    sim._verbal = verbal

    with mock.patch("engine.verbal_action.ollama_chat_speech_enabled", return_value=False):
        out = sim.handle_human_command("подойди к цилиндрическому объекту перед тобой")
        sim.tick += 1
        sim._tick_human_task(fallen=False)

    assert out.get("ok") is not False
    tt = sim._ensure_task_tree()
    assert tt.is_active
    assert tt.active_node is not None
    assert tt.active_node.kind == "resolve_target"
    assert sim._task_tree_kind == "goal"
    assert getattr(sim, "_deferred_vision_resolve", None)
    fail_texts = [str(getattr(m, "text", "")) for m in verbal._messages]
    assert not any("Не удалось" in t for t in fail_texts)

