"""Tests for human command task binding (PE-verifiable goals)."""
from __future__ import annotations

import numpy as np
import torch

from engine.causal_graph import CausalGraph
from engine.grounded_language import (
    motor_interventions_for_command,
    phrase_for_human_task,
    state_phrase_for_speech,
)
from engine.system2.schema import filter_expected_state_raw
from engine.system2.wm_planner import task_from_planning_context
from engine.task_binding import HumanTask, TaskBindingController, task_observation_keys
from engine.task_goal import GoalPredicate, TaskGoal


class _InterventionGraph:
    """Minimal graph: do(intent_stride) shifts posture_stability; free rollout decays it."""

    def __init__(self) -> None:
        self.nodes = {
            "intent_stride": 0.5,
            "intent_stop_recover": 0.5,
            "intent_torso_forward": 0.5,
            "posture_stability": 0.55,
            "target_dist": 0.7,
        }

    def propagate_from(self, base: dict[str, float], variable: str, value: float) -> dict[str, float]:
        out = dict(base)
        out[str(variable)] = float(value)
        if variable == "intent_stride":
            out["posture_stability"] = float(np.clip(0.35 + 0.55 * float(value), 0.05, 0.95))
        if variable == "intent_stop_recover":
            out["posture_stability"] = float(np.clip(0.4 + 0.45 * float(value), 0.05, 0.95))
            out["com_z"] = float(np.clip(0.35 + 0.35 * float(value), 0.05, 0.95))
        return out

    def rollout_step_free(self, base: dict[str, float]) -> dict[str, float]:
        out = dict(base)
        out["posture_stability"] = float(out.get("posture_stability", 0.5)) * 0.98
        out["target_dist"] = float(out.get("target_dist", 0.5)) * 0.995
        return out

def test_filter_expected_state_includes_slots() -> None:
    raw = {"slot_0": 0.7, "slot_3": 0.2, "posture_stability": 0.5, "bogus": 1.0}
    out = filter_expected_state_raw(raw, obs_keys=list(raw.keys()))
    assert "slot_0" in out
    assert "slot_3" in out
    assert "posture_stability" in out
    assert "bogus" not in out


def test_imagine_expected_state_rollout() -> None:
    g = CausalGraph(device=torch.device("cpu"))
    for i in range(6):
        g.set_node(f"slot_{i}", 0.5)
    g.set_node("posture_stability", 0.6)
    g.set_node("target_dist", 0.8)
    obs = {"slot_0": 0.5, "posture_stability": 0.6, "target_dist": 0.8}
    tb = TaskBindingController()
    expected, _, _ = tb.imagine_expected_state(g, obs, horizon=4)
    assert expected
    assert all(k in obs or k.startswith("slot_") or k in ("posture_stability", "target_dist") for k in expected)


def test_task_verify_success_when_close() -> None:
    tb = TaskBindingController()
    expected = {"target_dist": 0.3, "posture_stability": 0.7}
    task = HumanTask(
        text="подойди",
        expected_state=expected,
        max_prediction_error=0.5,
        tick_started=10,
        tick_deadline=1000,
    )
    obs = {"target_dist": 0.32, "posture_stability": 0.68, "intero_energy": 0.9, "intero_stress": 0.1}
    ok, pe, _ = tb.verify(obs, task)
    assert pe < 0.5
    assert ok


def test_state_phrase_standing_not_fallen() -> None:
    from engine.grounded_language import state_phrase_for_speech

    obs = {
        "com_z": 0.55,
        "posture_stability": 0.28,
        "intent_stride": 0.58,
    }
    assert state_phrase_for_speech(obs, fallen=False) in ("Иду вперёд", "Стабилен")
    assert state_phrase_for_speech(obs, fallen=False) != "Я упал"


def test_state_phrase_confirmed_fallen() -> None:
    from engine.grounded_language import state_phrase_for_speech

    obs = {"com_z": 0.15, "posture_stability": 0.2}
    assert state_phrase_for_speech(obs, fallen=True) == "Я упал"


def test_task_observation_keys_cap() -> None:
    obs = {f"slot_{i}": 0.5 for i in range(20)}
    obs["com_z"] = 0.5
    keys = task_observation_keys(obs)
    assert "com_z" in keys
    assert len(keys) <= 48


def test_imagine_intervention_differs_from_free_rollout() -> None:
    g = _InterventionGraph()
    obs = {"posture_stability": 0.55, "target_dist": 0.7, "intent_stride": 0.5}
    tb = TaskBindingController()
    free, _, _ = tb.imagine_expected_state(g, obs, horizon=6, text="")
    locomote, _, _ = tb.imagine_expected_state(g, obs, horizon=6, text="Иди вперёд")
    recover, _, _ = tb.imagine_expected_state(g, obs, horizon=6, text="Встань, ты упал")
    assert locomote != free
    assert recover != free
    assert locomote.get("posture_stability", 0) > free.get("posture_stability", 0)


def test_imagine_predicate_goal_not_keyword_fallback() -> None:
    g = _InterventionGraph()
    obs = {"posture_stability": 0.55, "target_dist": 0.7, "intent_stride": 0.5}
    goal = TaskGoal(
        text="иди вперёд",
        predicates=[
            GoalPredicate(
                kind="state_key",
                key="intent_stride",
                target_value=0.66,
                tolerance=0.15,
            ),
        ],
    )
    tb = TaskBindingController()
    _, _, diag = tb.imagine_expected_state(g, obs, horizon=4, text="иди вперёд", goal=goal)
    assert "intent_stride" in diag.get("interventions", [])


def test_imagine_keyword_fallback_without_goal() -> None:
    g = _InterventionGraph()
    obs = {"posture_stability": 0.55, "target_dist": 0.7, "intent_stride": 0.5}
    tb = TaskBindingController()
    _, _, diag = tb.imagine_expected_state(g, obs, horizon=4, text="Иди вперёд")
    assert "intent_stride" in diag.get("interventions", [])


def test_motor_interventions_for_command() -> None:
    loco = motor_interventions_for_command("Иди вперёд")
    rec = motor_interventions_for_command("Встань, ты упал")
    assert "intent_stride" in loco
    assert "intent_stop_recover" in rec


def test_state_phrase_human_task_semantic() -> None:
    obs = {"com_z": 0.55, "posture_stability": 0.6}
    assert state_phrase_for_speech(obs, human_task_text="Встань, ты упал") == "Встань, ты упал"
    assert state_phrase_for_speech(obs, human_task_text="Иди вперёд") == "Иду вперёд"
    assert phrase_for_human_task("Повернись") == "Повернись"


def test_task_from_planning_context_human_recover() -> None:
    ctx = {
        "human_task_active": True,
        "human_task_text": "Встань, ты упал",
        "expected_state": {"posture_stability": 0.8, "com_z": 0.6},
        "fallen": True,
    }
    t = task_from_planning_context(ctx, {})
    assert t.macro == "RECOVER_POSTURE"
    assert t.skill_id == "human_command"
    assert t.expected_state["posture_stability"] == 0.8


def test_task_from_planning_context_human_keeps_idle() -> None:
    """Human task must not invent LOCOMOTE/EXPLORE from NL tags."""
    for text in ("подойди ближе", "осмотрись", "иди вперёд"):
        ctx = {
            "human_task_active": True,
            "human_task_text": text,
            "macro": "IDLE",
            "expected_state": {"target_dist": 0.55},
            "skill_id": "human_command",
        }
        t = task_from_planning_context(ctx, {})
        assert t.macro == "IDLE", text
        # Stale LOCOMOTE/EXPLORE in ctx must also clamp.
        ctx["macro"] = "LOCOMOTE_DELIVERY"
        assert task_from_planning_context(ctx, {}).macro == "IDLE"
        ctx["macro"] = "EXPLORE"
        assert task_from_planning_context(ctx, {}).macro == "IDLE"
