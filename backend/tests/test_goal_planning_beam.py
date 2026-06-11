"""Beam planning horizons and imagination defaults."""
from __future__ import annotations

import os

import pytest

from engine.goal_planning import (
    beam_search_first_action,
    imagination_steps_default,
    imagination_steps_fallen,
    plan_depth,
    plan_depth_max,
)


class _FakeGraph:
    def __init__(self) -> None:
        self._node_ids = ["target_dist", "intent_stride"]
        self.nodes = {"target_dist": 0.8, "intent_stride": 0.5}

    def propagate_from_batch(self, base: dict, actions: list) -> list[dict]:
        out = []
        for var, val in actions:
            s = dict(base)
            s[var] = float(val)
            if var == "intent_stride":
                s["target_dist"] = max(0.1, float(base.get("target_dist", 0.8)) - 0.08 * (val - 0.5))
            out.append(s)
        return out

    def propagate_from_multi_batch(self, bases: list[dict], actions: list) -> list[dict]:
        return [
            self.propagate_from_batch(b, [(v, x)])[0] for b, (v, x) in zip(bases, actions)
        ]

    def rollout_step_free_batch(self, states: list[dict]) -> list[dict]:
        out = []
        for s in states:
            n = dict(s)
            n["target_dist"] = max(0.05, float(s.get("target_dist", 0.5)) - 0.02)
            out.append(n)
        return out


class _FakeAgent:
    def __init__(self) -> None:
        self.graph = _FakeGraph()
        self._imagination_horizon = 2

    def _batch_rollout_imagination_states(
        self, base, actions, *, row_bases=None, horizon=None
    ):
        h = 2 if horizon is None else horizon
        if row_bases is None:
            states = self.graph.propagate_from_batch(dict(base), actions)
        else:
            states = self.graph.propagate_from_multi_batch(row_bases, actions)
        for _ in range(h):
            states = self.graph.rollout_step_free_batch(states)
        return states


def test_plan_depth_defaults_extended(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RKK_PLAN_DEPTH", raising=False)
    monkeypatch.delenv("RKK_PLAN_DEPTH_MAX", raising=False)
    assert plan_depth() == 5
    assert plan_depth_max() >= 12


def test_imagination_defaults_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RKK_IMAGINATION_STEPS", raising=False)
    monkeypatch.delenv("RKK_IMAGINATION_STEPS_FALLEN", raising=False)
    assert imagination_steps_default() == 12
    assert imagination_steps_fallen() == 6


def test_beam_search_picks_better_first_action() -> None:
    agent = _FakeAgent()
    state0 = dict(agent.graph.nodes)
    actions = [("intent_stride", 0.38), ("intent_stride", 0.62)]

    def score(_s0, _v, val, sfin):
        return -float(sfin.get("target_dist", 1.0))

    best, _sc = beam_search_first_action(
        agent,
        state0=state0,
        actions=actions,
        depth=2,
        beam_k=2,
        rollout_horizon=2,
        score_fn=score,
        maximize=True,
    )
    assert best is not None
    assert best[0] == "intent_stride"
    assert best[1] == 0.62
