"""Tests for neural / latent / WM diagnostic logger."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_neural_log_writes_jsonl_and_txt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RKK_NEURAL_LOG", "1")
    monkeypatch.setenv("RKK_NEURAL_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("RKK_NEURAL_LOG_EVERY", "1")

    from engine.neural_logger import neural_log_event, summarize_latent, summarize_slot_table

    neural_log_event(
        "vision",
        "resolve",
        tick=10,
        force=True,
        reason="ok",
        candidates=summarize_slot_table(
            [
                {
                    "slot_id": "slot_0",
                    "label": "ball",
                    "uv_valid": False,
                    "u": 0.4,
                    "v": 0.5,
                    "activation": 0.2,
                    "match_score": 0.4,
                    "vector": [1.0, 0.0, 0.0],
                }
            ]
        ),
        latent=summarize_latent([1.0, 0.0, 0.0]),
    )

    jsonl = tmp_path / "neural_log.jsonl"
    txt = tmp_path / "neural_log.txt"
    assert jsonl.is_file()
    assert txt.is_file()
    row = json.loads(jsonl.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert row["channel"] == "vision"
    assert row["event"] == "resolve"
    assert row["tick"] == 10
    assert row["candidates"][0]["slot_id"] == "slot_0"
    assert row["latent"]["dim"] == 3
    assert "ch=vision" in txt.read_text(encoding="utf-8")


def test_neural_log_throttle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RKK_NEURAL_LOG", "1")
    monkeypatch.setenv("RKK_NEURAL_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("RKK_NEURAL_LOG_EVERY", "5")

    from engine import neural_logger

    neural_logger._last_tick_by_channel.clear()
    neural_logger.neural_log_event("owm", "track", tick=100, bearing=0.1)
    neural_logger.neural_log_event("owm", "track", tick=101, bearing=0.2)  # throttled
    neural_logger.neural_log_event("owm", "track", tick=106, bearing=0.3)

    lines = (tmp_path / "neural_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["bearing"] == 0.1
    assert json.loads(lines[1])["bearing"] == 0.3


def test_clear_session_logs_includes_neural(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RKK_TASK_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("RKK_NEURAL_LOG_DIR", str(tmp_path))
    (tmp_path / "neural_log.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "neural_log.txt").write_text("x\n", encoding="utf-8")
    from engine.task_logger import clear_session_logs

    cleared = clear_session_logs()
    assert "neural_log.jsonl" in cleared
    assert not (tmp_path / "neural_log.jsonl").exists()
