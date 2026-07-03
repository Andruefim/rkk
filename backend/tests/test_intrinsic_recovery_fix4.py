"""Fix 4: recovery goals via GoalImagination + get_target_priors."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from engine.intristic_objective import (
    GoalImagination,
    IntrinsicObjective,
    _recovery_goal_candidates,
)


class _GraphStub:
    def __init__(self, node_ids: list[str]):
        self._node_ids = node_ids
        self.nodes = {n: 0.5 for n in node_ids}


def test_get_target_priors_recovery_when_fallen():
    obj = IntrinsicObjective.__new__(IntrinsicObjective)
    obj._posture_hist = __import__("collections").deque(maxlen=100)

    priors = IntrinsicObjective.get_target_priors(
        obj,
        {"posture_stability": 0.25},
    )
    assert priors == {
        "intent_stop_recover": 0.80,
        "intent_torso_forward": 0.55,
    }


def test_get_target_priors_no_recovery_when_stable():
    obj = IntrinsicObjective.__new__(IntrinsicObjective)
    obj._posture_hist = __import__("collections").deque(maxlen=100)
    obj.goal_imagination = GoalImagination(__import__("torch").device("cpu"))
    obj.goal_imagination._current_goal = None

    priors = IntrinsicObjective.get_target_priors(
        obj,
        {"posture_stability": 0.85},
    )
    assert priors == {}


def test_recovery_goal_candidates_resolves_phys_prefix():
    g = _GraphStub(["phys_intent_stop_recover", "phys_intent_torso_forward"])
    cands = _recovery_goal_candidates(g)
    assert ("phys_intent_stop_recover", 0.80) in cands
    assert ("phys_intent_torso_forward", 0.55) in cands


def test_generate_goal_fallen_uses_recovery_fallback():
    gi = GoalImagination(__import__("torch").device("cpu"))
    graph = _GraphStub(["intent_stop_recover", "intent_torso_forward", "intent_stride"])
    env = MagicMock()
    env.observe.return_value = {"posture_stability": 0.2, "intent_stride": 0.5}

    with patch("engine.hypothesis_testing.eig_for_action", return_value=0.0):
        goal = gi.generate_goal(
            graph=graph,
            agent_env=env,
            n_interventions=100,
            causal_surprise=MagicMock(),
        )

    assert goal is not None
    assert goal["target_var"] == "intent_stop_recover"
    assert goal["target_val"] == 0.80


def test_generate_goal_fallen_picks_max_eig_recovery():
    gi = GoalImagination(__import__("torch").device("cpu"))
    graph = _GraphStub(["intent_stop_recover", "intent_torso_forward"])
    env = MagicMock()
    env.observe.return_value = {"posture_stability": 0.15}

    def _eig(_g, _obs, action, return_best=False):
        # find torso_forward candidate
        for var, val in action:
            if "torso_forward" in var:
                return 0.9, var, val
        return 0.1, action[0][0], action[0][1]

    with patch("engine.hypothesis_testing.eig_for_action", side_effect=_eig):
        goal = gi.generate_goal(
            graph=graph,
            agent_env=env,
            n_interventions=50,
            causal_surprise=MagicMock(),
        )

    assert goal is not None
    assert goal["target_var"] == "intent_torso_forward"
    assert goal["target_val"] == 0.55
