"""Track A Phase 0: curriculum tags in trajectory segments."""
from __future__ import annotations

import unittest

from engine.eval_mode import aggregate_segment_tags
from engine.trajectory_contrastive import TrajectoryCollector


class TrajectoryBufferTagsTests(unittest.TestCase):
    def test_finalize_includes_buffer_tags(self) -> None:
        col = TrajectoryCollector()
        col.segment_len = 4
        col.overlap = 1
        tags = [
            {"fixed_root": True, "fallen": False, "curriculum_step": 1, "scope_phase": 0},
            {"fixed_root": True, "fallen": False, "curriculum_step": 1, "scope_phase": 0},
            {"fixed_root": False, "fallen": True, "curriculum_step": 2, "scope_phase": 0},
            {"fixed_root": False, "fallen": False, "curriculum_step": 2, "scope_phase": 1},
        ]
        seg = None
        for i, t in enumerate(tags):
            seg = col.tick(
                obs={"a": 0.5, "posture_stability": 0.7},
                action=("intent_stride", 0.5),
                is_fallen=bool(t["fallen"]),
                node_ids=["a"],
                engine_tick=i,
                curriculum_tags=t,
            )
        assert seg is not None
        self.assertIn("fallen_frac", seg.outcome)
        self.assertIn("fixed_root_frac", seg.outcome)
        self.assertIn("dominant_stage", seg.outcome)
        self.assertEqual(seg.outcome["fallen_frac"], 0.25)
        self.assertEqual(seg.outcome["fixed_root_frac"], 0.5)

    def test_aggregate_segment_tags(self) -> None:
        agg = aggregate_segment_tags(
            [
                {"fixed_root": True, "fallen": False, "scope_phase": 0},
                {"fixed_root": False, "fallen": False, "scope_phase": 1},
            ]
        )
        self.assertEqual(agg["fallen_frac"], 0.0)
        self.assertEqual(agg["fixed_root_frac"], 0.5)
