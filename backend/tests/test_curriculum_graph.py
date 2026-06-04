"""Track G Phase 5: CurriculumGraph DAG."""
from __future__ import annotations

import os
import unittest
from unittest import mock

from engine.curriculum_graph import CurriculumGraph, CurriculumNode
from engine.goal_generator import GoalCandidate
from engine.physical_curriculum import PhysicalCurriculum


class CurriculumGraphTests(unittest.TestCase):
    def test_seed_from_physical_curriculum(self) -> None:
        with mock.patch.dict(os.environ, {"RKK_CURRICULUM_GRAPH_HUMAN_SEED": "1"}):
            cg = CurriculumGraph()
            n = cg.seed_from_physical_curriculum(PhysicalCurriculum())
            self.assertGreater(n, 0)
            self.assertGreater(len(cg._nodes), 0)

    def test_generated_node_and_complete(self) -> None:
        cg = CurriculumGraph()
        cand = GoalCandidate(var_id="intent_stride", score=1.2, tick_proposed=50)
        node = cg.add_generated_node(cand, tick=50)
        self.assertIsNotNone(node)
        assert node is not None
        self.assertEqual(node.source, "generated")
        active = cg.activate_next(60, world_id="humanoid")
        self.assertIsNotNone(active)
        assert active is not None
        newly = cg.mark_completed(active.node_id, success_rate=0.55, tick=300)
        self.assertIsInstance(newly, list)
        snap = cg.snapshot()
        completed = snap.get("completed_generated") or []
        if completed:
            self.assertEqual(completed[-1].get("source"), "generated")

    def test_persistence_roundtrip(self) -> None:
        cg = CurriculumGraph()
        cg._nodes["n1"] = CurriculumNode(
            node_id="n1",
            var_id="posture_stability",
            intent_targets={"posture_stability": 0.7},
            status="completed",
            source="generated",
            success_rate=0.5,
        )
        data = cg.to_dict()
        cg2 = CurriculumGraph()
        cg2.load_dict(data)
        self.assertIn("n1", cg2._nodes)
        self.assertEqual(cg2._nodes["n1"].source, "generated")

    def test_goal_transfer(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "RKK_GOAL_TRANSFER_ENABLED": "1",
                "RKK_GOAL_TRANSFER_MIN_SUCCESS": "0.40",
            },
        ):
            cg = CurriculumGraph()
            cg._nodes["h1"] = CurriculumNode(
                node_id="h1",
                var_id="intent_stride",
                world_id="humanoid",
                status="completed",
                success_rate=0.55,
                intent_targets={"intent_stride": 0.6},
            )
            xfer = cg.transfer_goals_to_world("humanoid", "humanoid_variant")
            self.assertGreaterEqual(len(xfer), 1)
            self.assertEqual(xfer[0].source, "transferred")
