"""Tests for session log clear + AI windowed analyzer."""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from engine.task_log_analyzer import (
    analyze_tick_window,
    maybe_analyze_task_logs,
    reset_analyzer_state_for_tests,
    task_log_ai_enabled,
    task_log_ai_every,
)
from engine.task_logger import (
    ai_analysis_path,
    append_ai_analysis,
    clear_session_logs,
    read_ai_analysis_text,
    read_task_log_events,
    task_log_event,
)


@pytest.fixture
def task_log_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RKK_TASK_LOG", "1")
    monkeypatch.setenv("RKK_TASK_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("RKK_TASK_LOG_AI", "1")
    monkeypatch.setenv("RKK_TASK_LOG_AI_EVERY", "100")
    reset_analyzer_state_for_tests()
    yield tmp_path
    reset_analyzer_state_for_tests()


def test_clear_session_logs_removes_runtime_files(task_log_tmp: Path) -> None:
    task_log_event("command_received", tick=1, text="go")
    append_ai_analysis(1, 100, "first window")
    (task_log_tmp / "live_uv_candidates.jsonl").write_text("{}\n", encoding="utf-8")
    (task_log_tmp / "system2_distill.jsonl").write_text("{}\n", encoding="utf-8")
    cleared = clear_session_logs()
    assert "task_log.jsonl" in cleared
    assert "task_log.txt" in cleared
    assert "ai_task_analysis.txt" in cleared
    assert not (task_log_tmp / "task_log.jsonl").exists()
    assert not ai_analysis_path().exists()


def test_read_task_log_events_filters_by_tick(task_log_tmp: Path) -> None:
    for t in (10, 50, 120, 150):
        task_log_event("task_progress", tick=t, node_kind="approach")
    rows = read_task_log_events(tick_lo=100, tick_hi=150)
    ticks = [r["tick"] for r in rows]
    assert ticks == [120, 150]


def test_append_and_read_ai_analysis(task_log_tmp: Path) -> None:
    append_ai_analysis(0, 100, "стоит на месте")
    append_ai_analysis(101, 200, "пошёл к цели")
    text = read_ai_analysis_text()
    assert "[0-100]:" in text
    assert "стоит на месте" in text
    assert "[101-200]:" in text


def test_analyze_tick_window_uses_ollama_and_writes(
    task_log_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_log_event(
        "command_received",
        tick=5,
        text="подойди к объекту",
    )
    task_log_event(
        "task_progress",
        tick=50,
        task_nav_active=1.0,
        vision_range_m=1.2,
        macro_hint="IDLE",
    )

    def _fake_ollama(prompt: str) -> str:
        assert "подойди" in prompt or "command_received" in prompt
        assert "[0-100]" in prompt or "0-100" in prompt
        return "Команда получена; nav активен; range≈1.2; macro IDLE."

    monkeypatch.setattr(
        "engine.task_log_analyzer._call_ollama",
        _fake_ollama,
    )
    out = analyze_tick_window(0, 100)
    assert "nav" in out.lower() or "IDLE" in out
    saved = ai_analysis_path().read_text(encoding="utf-8")
    assert "[0-100]:" in saved
    assert out in saved


def test_maybe_analyze_triggers_on_boundary(
    task_log_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert task_log_ai_enabled()
    assert task_log_ai_every() == 100
    calls: list[tuple[int, int]] = []

    def _fake_analyze(lo: int, hi: int) -> str:
        calls.append((lo, hi))
        append_ai_analysis(lo, hi, f"ok {lo}-{hi}")
        return "ok"

    monkeypatch.setattr(
        "engine.task_log_analyzer.analyze_tick_window",
        _fake_analyze,
    )
    # Empty window: no Ollama / no job.
    assert maybe_analyze_task_logs(50) is False
    assert maybe_analyze_task_logs(100) is False
    assert calls == []

    reset_analyzer_state_for_tests()
    task_log_event("command_received", tick=42, text="иди")
    assert maybe_analyze_task_logs(100) is True
    import time

    for _ in range(50):
        if calls:
            break
        time.sleep(0.02)
    assert calls == [(0, 100)]


def test_analyze_skips_ollama_when_no_events(
    task_log_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = {"n": 0}

    def _fake_ollama(prompt: str) -> str:
        called["n"] += 1
        return "should not run"

    monkeypatch.setattr("engine.task_log_analyzer._call_ollama", _fake_ollama)
    out = analyze_tick_window(0, 100)
    assert out == ""
    assert called["n"] == 0
    assert not ai_analysis_path().exists()
