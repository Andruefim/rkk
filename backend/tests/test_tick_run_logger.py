"""Tick run JSONL logger."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock


class TickRunLoggerTests(unittest.TestCase):
    def test_writes_run_start_and_tick(self) -> None:
        from engine import tick_run_logger as trl

        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "run.jsonl"
            os.environ["RKK_TICK_RUN_LOG"] = "1"
            os.environ["RKK_TICK_RUN_LOG_PATH"] = str(log_path)
            os.environ["RKK_TICK_RUN_LOG_EVERY"] = "1"
            trl._LOGGER = None

            sim = MagicMock()
            sim.tick = 1
            sim.current_world = "humanoid"
            sim._fixed_root_active = False
            sim._visual_mode = True
            sim._fall_count = 0
            sim._tick_log_prev_edges = 0
            sim._dr_window = [0.1]
            sim.phase = 1
            sim.events = []
            sim._sleep_ctrl = None
            sim._locomotion_controller = None
            sim._system2_last = {
                "enabled": True,
                "macro": "LOCOMOTE_DELIVERY",
                "source": "student",
                "until": 120,
                "idle": False,
            }
            sim.agent = MagicMock()
            sim.agent.graph._node_ids = ["a", "b"]
            sim.agent._last_step_timings = {"total_ms": 12.5, "observe": 1.0}
            sim.agent.env = MagicMock()
            sim.agent.env._motor_state = {"intent_lean_forward": 0.7}
            sim.agent.env.cpg_owns_legs = True
            sim.agent.snapshot.return_value = {
                "edge_count": 10,
                "discovery_rate": 0.05,
                "phi": 0.2,
            }

            trl.record_sim_tick(
                sim,
                result={
                    "variable": "phys_intent_lean_forward",
                    "value": 0.71,
                    "blocked": False,
                    "prediction_error": 0.02,
                    "compression_delta": 0.0,
                },
                snap={
                    "phi": 0.2,
                    "compression_gain": -0.01,
                    "alpha_mean": 0.08,
                    "h_W": 0.01,
                    "edge_count": 10,
                    "total_interventions": 5,
                    "total_blocked": 0,
                    "discovery_rate": 0.05,
                    "fall_count": 0,
                    "value_layer": {"block_rate": 0.0, "vl_phase": "warmup"},
                    "progressive_scope": {"phase": 0, "mastery_quality": 0.0},
                    "trajectory": {"segments": 1},
                    "system1": {"mean_loss": 0.1},
                },
                inner_ms=45.0,
                obs={"posture_stability": 0.4, "com_z": 0.5},
                fallen=False,
                posture=0.4,
            )
            trl.finalize_tick_run_log(sim)
            trl._LOGGER = None

            lines = log_path.read_text(encoding="utf-8").strip().split("\n")
            self.assertGreaterEqual(len(lines), 3)
            start = json.loads(lines[0])
            self.assertEqual(start["type"], "run_start")
            tick = json.loads(lines[1])
            self.assertEqual(tick["type"], "tick")
            self.assertEqual(tick["tick"], 1)
            self.assertEqual(tick["action"]["variable"], "phys_intent_lean_forward")
            self.assertIn("system2", tick)
            self.assertEqual(tick["system2"]["macro"], "LOCOMOTE_DELIVERY")
            self.assertEqual(tick["system2"]["source"], "student")
            self.assertIn("perf", tick)
            end = json.loads(lines[-1])
            self.assertEqual(end["type"], "run_end")

            del os.environ["RKK_TICK_RUN_LOG"]
            del os.environ["RKK_TICK_RUN_LOG_PATH"]
            del os.environ["RKK_TICK_RUN_LOG_EVERY"]


if __name__ == "__main__":
    unittest.main()
