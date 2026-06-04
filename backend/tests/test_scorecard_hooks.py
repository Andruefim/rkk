"""Track A Phase 0: autonomy scorecard schema hooks."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from engine.scorecard.autonomy_scorecard import build_scorecard, write_scorecard


class ScorecardHooksTests(unittest.TestCase):
    def test_build_scorecard_worlds_and_thresholds(self) -> None:
        card = build_scorecard(
            {"discovery_new_frac": 0.55},
            worlds=["humanoid", "grid_nav"],
        )
        self.assertIn("worlds", card)
        self.assertIn("thresholds", card)
        self.assertIn("a1_max", card["thresholds"])
        self.assertIn("humanoid", card["worlds"])
        self.assertIn("grid_nav", card["worlds"])
        self.assertIn("pass_core_embodied", card)

    def test_write_scorecard_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "autonomy_scorecard.json"
            write_scorecard(build_scorecard(), p)
            data = json.loads(p.read_text(encoding="utf-8"))
            self.assertIn("worlds", data)
