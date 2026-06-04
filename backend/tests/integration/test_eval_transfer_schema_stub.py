"""Phase 0 stub: transfer eval JSONL schema without full PyBullet run."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import assert_transfer_row_schema  # noqa: E402


def test_transfer_row_schema_stub() -> None:
    row = {
        "eval_kind": "within_run_transfer",
        "success_rate": 0.42,
        "fallen_frac": 0.12,
        "ticks_to_recover": None,
        "train_stage": "fixed_root",
        "eval_stage": "curriculum_step_2",
        "fixed_root": False,
        "curriculum_step": 2,
        "scope_phase": 1,
    }
    assert_transfer_row_schema(row)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "transfer_eval.jsonl"
        p.write_text(json.dumps(row) + "\n", encoding="utf-8")
        loaded = json.loads(p.read_text(encoding="utf-8").strip())
        assert_transfer_row_schema(loaded)
