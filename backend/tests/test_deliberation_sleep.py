"""Deliberation worker + sleep lesson annotation fixes."""
from __future__ import annotations

import os
from unittest import mock

import torch

from engine.deliberation_worker import (
    DeliberationResult,
    DeliberationService,
    deliberation_enabled,
    deliberation_plan_value_levels,
)
from engine.sleep_consolidation import SleepController, SleepLessonAnnotation


def test_sleep_lesson_has_intent_adjustments() -> None:
    ann = SleepLessonAnnotation(tick=1, timestamp=0.0, mode="lesson")
    assert hasattr(ann, "intent_adjustments")
    assert ann.intent_adjustments == {}


def test_apply_lesson_without_intent_adjustments_crash() -> None:
    sc = SleepController()
    ann = SleepLessonAnnotation(
        tick=10,
        timestamp=0.0,
        mode="lesson",
        primary_concepts=["balance"],
    )

    class _Timescale:
        def set_intent(self, _lvl, var, val):
            pass

    class _Sim:
        agent = type("A", (), {"graph": type("G", (), {"_node_ids": [], "nodes": {}})()})()
        _inner_voice = None
        _timescale = _Timescale()

    sc._apply_lesson(10, _Sim(), ann)


def test_deliberation_plan_values_compact() -> None:
    with mock.patch.dict(
        os.environ, {"RKK_DELIBERATION_PLAN_VALUES": "0.42,0.62"}, clear=False
    ):
        levels = deliberation_plan_value_levels()
        assert levels == [0.42, 0.62]


def test_deliberation_enabled_default() -> None:
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("RKK_DELIBERATION_ENABLED", None)
        assert deliberation_enabled()


def test_deliberation_enqueue_coalesces_while_busy() -> None:
    class _Graph:
        _node_ids = ["intent_stride"]
        nodes = {"intent_stride": 0.5}

    class _Env:
        preset = "humanoid_variant"

    class _Agent:
        env = _Env()
        graph = _Graph()

    class _Sim:
        agent = _Agent()
        tick = 100
        _sim_step_lock = __import__("threading").Lock()

    svc = DeliberationService(_Sim())
    svc._busy = True
    assert svc.enqueue(tick=10, macro="IDLE", intention_ctx=None) is True
    assert svc.enqueue(tick=20, macro="LOCOMOTE_DELIVERY", intention_ctx=None) is True
    with svc._coalesce_lock:
        assert svc._coalesced is not None
        assert svc._coalesced.tick == 20
        assert svc._coalesced.macro == "LOCOMOTE_DELIVERY"


def test_deliberation_result_roundtrip() -> None:
    r = DeliberationResult(
        tick=100,
        macro_hint="LOCOMOTE_DELIVERY",
        first_action=("intent_stride", 0.58),
        score=0.42,
        stale=False,
    )
    d = r.to_dict()
    assert d["macro_hint"] == "LOCOMOTE_DELIVERY"
    assert d["first_action"]["variable"] == "intent_stride"
