"""Tests for engine.tick_profiler."""
from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _enable_profiler(monkeypatch):
    monkeypatch.setenv("RKK_TICK_PROFILE", "1")
    monkeypatch.setenv("RKK_TICK_PROFILE_WINDOW", "50")
    monkeypatch.setenv("RKK_TICK_PROFILE_REPORT_EVERY", "10000")
    from engine import tick_profiler as tp

    tp._profiler = None
    yield
    tp._profiler = None


def test_span_records_and_ranks():
    from engine.tick_profiler import get_tick_profiler, tick_profile

    p = get_tick_profiler()
    p.begin_tick(1)
    with tick_profile("agent.train_step"):
        pass
    p.record("agent.train_step", 120.0)
    p.record("agent.observe", 5.0)
    p.end_tick()

    ranked = p.ranked(top_n=10)
    names = [r["name"] for r in ranked]
    assert "agent.train_step" in names
    train = next(r for r in ranked if r["name"] == "agent.train_step")
    assert train["avg_ms"] >= 5.0


def test_snapshot_shape():
    from engine.tick_profiler import get_tick_profiler, profile_snapshot

    p = get_tick_profiler()
    p.begin_tick(2)
    p.record("sim.wall", 50.0)
    p.end_tick()
    snap = profile_snapshot()
    assert snap["enabled"] is True
    assert "ranked" in snap
    assert snap["effective_hz"] > 0


def test_disabled_when_env_off(monkeypatch):
    monkeypatch.setenv("RKK_TICK_PROFILE", "0")
    from engine import tick_profiler as tp
    from engine.tick_profiler import TickProfiler, profile_snapshot

    tp._profiler = None
    assert TickProfiler.enabled() is False
    assert profile_snapshot() == {"enabled": False}
