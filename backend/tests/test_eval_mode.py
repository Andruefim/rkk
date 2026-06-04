"""Track A Phase 0: RKK_EVAL_MODE suppresses WM train and distill."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from engine.causal_graph import CausalGraph
from engine.eval_mode import cross_env_allow_wm_train, eval_mode_enabled
from engine.system2.controller import System2Controller
from engine.system2.distill_log import distill_enabled


class EvalModeTests(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("RKK_EVAL_MODE", None)
        os.environ.pop("RKK_CROSS_ENV_ALLOW_WM_TRAIN", None)

    def test_eval_mode_env_flag(self) -> None:
        os.environ["RKK_EVAL_MODE"] = "1"
        self.assertTrue(eval_mode_enabled())
        os.environ["RKK_EVAL_MODE"] = "0"
        self.assertFalse(eval_mode_enabled())

    def test_graph_train_step_skipped_in_eval_mode(self) -> None:
        os.environ["RKK_EVAL_MODE"] = "1"
        g = CausalGraph(device=__import__("torch").device("cpu"))
        g.rebind_variables(["a", "b", "c"], {"a": 0.5, "b": 0.5, "c": 0.5})
        before = int(getattr(g, "_wm_train_calls", 0))
        out = g.train_step()
        after = int(getattr(g, "_wm_train_calls", 0))
        self.assertIsNone(out)
        self.assertEqual(before, after)

    def test_distill_append_skipped_in_eval_mode(self) -> None:
        os.environ["RKK_EVAL_MODE"] = "1"
        self.assertTrue(distill_enabled())
        ctrl = System2Controller()
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "distill.jsonl"
            with mock.patch.dict(
                os.environ,
                {"RKK_SYSTEM2_DISTILL_LOG": str(log)},
            ):
                ctrl._append_distill(
                    tick=1,
                    macro="IDLE",
                    source="test",
                    success=True,
                    delta={},
                )
            self.assertFalse(log.exists())

    def test_cross_env_wm_train_blocked(self) -> None:
        os.environ["RKK_CROSS_ENV_ALLOW_WM_TRAIN"] = "0"
        self.assertFalse(cross_env_allow_wm_train())
        g = CausalGraph(device=__import__("torch").device("cpu"))
        g.rebind_variables(["a", "b"], {"a": 0.5, "b": 0.5})
        before = int(getattr(g, "_wm_train_calls", 0))
        self.assertIsNone(g.train_step())
        self.assertEqual(before, int(getattr(g, "_wm_train_calls", 0)))
