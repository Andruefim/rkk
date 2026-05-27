"""System2 async LLM plan timeout → student fallback."""
from __future__ import annotations

from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import pytest

from engine.system2.controller import System2Controller, _system2_llm_timeout_ticks


def test_system2_llm_timeout_ticks_default():
    assert _system2_llm_timeout_ticks() >= 16


def test_maybe_timeout_clears_inflight_and_replans(monkeypatch):
    monkeypatch.setenv("RKK_SYSTEM2_LLM_TIMEOUT_TICKS", "10")
    ctrl = System2Controller()
    fut: Future = Future()
    ctrl._llm_future = fut
    ctrl._llm_submit_tick = 100
    ctrl._active_macro = "LOCOMOTE_DELIVERY"
    ctrl._macro_until_tick = 200
    ctrl._last_source = "student"

    agent = MagicMock()
    agent.graph._node_ids = ["intent_stride", "com_z"]
    base = MagicMock()
    graph = agent.graph
    obs_f = {"com_z": 0.6, "posture_stability": 0.7}

    with patch.object(ctrl, "_apply_planning_step", return_value={"enabled": True, "macro": "EXPLORE", "source": "student"}) as apply:
        out = ctrl._maybe_timeout_system2_llm(120, agent, obs_f, base, graph, frozenset(graph._node_ids), None)

    assert out is not None
    assert out.get("llm_timeout") is True
    assert ctrl._llm_future is None
    apply.assert_called_once()
    assert apply.call_args[0][7] is None  # proposal_llm
