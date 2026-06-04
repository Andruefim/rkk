"""
Integration helpers: JSONL parse, gate snapshot/result, scorecard schema asserts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_TRANSFER_FIELDS = frozenset(
    {
        "eval_kind",
        "success_rate",
        "fallen_frac",
        "ticks_to_recover",
        "train_stage",
        "eval_stage",
        "fixed_root",
        "curriculum_step",
        "scope_phase",
    }
)


def parse_jsonl(path: Path | str, *, last_n: int = 1) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if last_n <= 0:
        return rows
    return rows[-last_n:]


def load_gate_snapshot_meta(snapshot_path: Path | str) -> dict[str, Any] | None:
    p = Path(snapshot_path)
    meta = p.with_suffix(".meta.json")
    if not meta.is_file():
        return None
    try:
        return json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_gate_result(path: Path | str) -> dict[str, Any] | None:
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def assert_transfer_row_schema(row: dict[str, Any]) -> None:
    missing = REQUIRED_TRANSFER_FIELDS - set(row.keys())
    assert not missing, f"missing transfer eval fields: {sorted(missing)}"
    assert row.get("eval_kind") == "within_run_transfer"


def assert_scorecard_schema(card: dict[str, Any]) -> None:
    assert "worlds" in card and isinstance(card["worlds"], dict)
    assert "thresholds" in card and isinstance(card["thresholds"], dict)
    assert "a1_max" in card["thresholds"]
    assert "a4_max" in card["thresholds"]
