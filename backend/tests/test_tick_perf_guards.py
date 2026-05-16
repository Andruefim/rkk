"""Guards for tick-path performance helpers."""
from __future__ import annotations

import os
import unittest

import numpy as np

from engine.llm_curriculum import CurriculumScheduler, CurriculumStage


class TestSupportBiasRangeMetric(unittest.TestCase):
    def test_range_uses_window_not_instant(self):
        stage = CurriculumStage(
            stage_id=1,
            name="weight_shift",
            description="",
            advance_conditions={"support_bias_range": 0.20},
            min_ticks=1,
        )
        sched = CurriculumScheduler()
        sched._stages = [stage]
        sched._current_idx = 0
        for bias in (0.50, 0.52, 0.48, 0.70, 0.72, 0.68):
            obs = {"support_bias": bias, "posture_stability": 0.8}
            sched.tick(1, obs, fallen=False)
        recent = list(sched._metrics_history)[-20:]
        biases = [float(m["support_bias"]) for m in recent]
        window_range = max(biases) - min(biases)
        self.assertGreaterEqual(window_range, 0.20)

    def test_instant_bias_deviation_insufficient_for_weight_shift(self):
        """Старый баг: abs(bias-0.5) на одном тике не отражает качание веса."""
        stage = CurriculumStage(
            stage_id=1,
            name="weight_shift",
            description="",
            advance_conditions={"support_bias_range": 0.28},
            min_ticks=1,
        )
        sched = CurriculumScheduler()
        sched._stages = [stage]
        sched._current_idx = 0
        for bias in (0.50, 0.51, 0.49, 0.52, 0.48):
            sched.tick(1, {"support_bias": bias, "posture_stability": 0.8}, fallen=False)
        recent = list(sched._metrics_history)[-20:]
        mean_metrics = {
            k: float(np.mean([m.get(k, 0.0) for m in recent]))
            for k in recent[0].keys()
        }
        biases = [float(m.get("support_bias", 0.5)) for m in recent]
        mean_metrics["support_bias_range"] = float(max(biases) - min(biases))
        self.assertLess(mean_metrics["support_bias_range"], 0.28)


if __name__ == "__main__":
    unittest.main()
