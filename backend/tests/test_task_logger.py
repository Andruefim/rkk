"""Tests for human-task file logger."""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from engine.task_logger import (
    _MAX_BYTES,
    summarize_expected_state,
    task_log_dir,
    task_log_event,
    task_log_enabled,
)


@pytest.fixture
def task_log_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RKK_TASK_LOG", "1")
    monkeypatch.setenv("RKK_TASK_LOG_DIR", str(tmp_path))
    yield tmp_path


def test_task_log_disabled_by_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RKK_TASK_LOG", "0")
    monkeypatch.setenv("RKK_TASK_LOG_DIR", str(tmp_path))
    task_log_event("command_received", tick=1, text="hello")
    assert not (tmp_path / "task_log.jsonl").exists()


def test_task_log_enabled_default_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RKK_TASK_LOG", raising=False)
    assert task_log_enabled() is True


def test_task_log_event_writes_jsonl_and_txt(task_log_tmp: Path) -> None:
    task_log_event(
        "command_received",
        tick=42,
        text="move chair",
        command_kind="manipulate",
        tag="MANIPULATE",
    )
    jsonl = task_log_tmp / "task_log.jsonl"
    txt = task_log_tmp / "task_log.txt"
    assert jsonl.is_file()
    assert txt.is_file()

    row = json.loads(jsonl.read_text(encoding="utf-8").strip())
    assert row["event"] == "command_received"
    assert row["tick"] == 42
    assert row["text"] == "move chair"
    assert row["command_kind"] == "manipulate"
    assert "ts" in row

    human = txt.read_text(encoding="utf-8").strip()
    assert "event=command_received" in human
    assert "tick=42" in human


def test_task_log_dir_from_env(task_log_tmp: Path) -> None:
    assert task_log_dir() == task_log_tmp


def test_summarize_expected_state() -> None:
    summary = summarize_expected_state(
        {"com_x": 0.0, "com_y": 0.5, "posture_stability": 0.8, "slot_0": 0.01}
    )
    assert summary["n_expected_keys"] == 4
    assert summary["n_nonzero"] == 3
    assert "posture_stability" in summary["top5"]


def test_task_log_rotation(task_log_tmp: Path) -> None:
    jsonl = task_log_tmp / "task_log.jsonl"
    txt = task_log_tmp / "task_log.txt"
    payload = "x" * (_MAX_BYTES + 1024)
    jsonl.write_text(payload + "\n", encoding="utf-8")
    txt.write_text(payload + "\n", encoding="utf-8")

    task_log_event("task_progress", tick=1, node_kind="execute_goal")

    assert jsonl.is_file()
    assert (task_log_tmp / "task_log.jsonl.1").is_file()
    line = jsonl.read_text(encoding="utf-8").strip()
    row = json.loads(line)
    assert row["event"] == "task_progress"


def test_task_log_swallows_internal_errors(
    task_log_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with mock.patch(
        "engine.task_logger._json_safe",
        side_effect=RuntimeError("boom"),
    ):
        task_log_event("command_received", tick=1, text="safe")


def test_task_log_creates_log_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nested = tmp_path / "nested" / "logs"
    monkeypatch.setenv("RKK_TASK_LOG", "1")
    monkeypatch.setenv("RKK_TASK_LOG_DIR", str(nested))
    task_log_event("command_received", tick=0, text="init")
    assert (nested / "task_log.jsonl").is_file()
