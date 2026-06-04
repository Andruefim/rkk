"""Track A Phase 0: curriculum eval gate metrics and result I/O."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from engine.curriculum_eval_gate import (
    evaluate_gate_metrics,
    load_gate_result,
    write_gate_result,
)


class CurriculumEvalGateTests(unittest.TestCase):
    def test_evaluate_gate_pass_fail(self) -> None:
        ok = evaluate_gate_metrics(
            fallen_frac=0.2,
            success_rate=0.45,
            quality=0.4,
        )
        self.assertTrue(ok["passed"])
        bad = evaluate_gate_metrics(
            fallen_frac=0.9,
            success_rate=0.1,
        )
        self.assertFalse(bad["passed"])
        self.assertIn("thresholds", bad)

    def test_gate_result_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "eval_gate_result.json"
            write_gate_result({"passed": True, "fallen_frac": 0.1}, p)
            loaded = load_gate_result(p)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertTrue(loaded["passed"])
            raw = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(raw["fallen_frac"], 0.1)
