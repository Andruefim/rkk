"""Track C5: promote_to_universal_concept across worlds."""
from __future__ import annotations

import os
import unittest
from unittest import mock

import torch

from engine.causal_graph import CausalGraph
from engine.genome.learned_roles import (
    learned_roles,
    promote_min_worlds,
    promote_to_universal_concept,
    reset_learned_roles,
)
from engine.latent_confounder import LatentRecord


class LearnedRolesPromoteTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_learned_roles()

    def test_promote_after_two_worlds(self) -> None:
        g = CausalGraph(device=torch.device("cpu"))
        rec = LatentRecord(
            node_id="latent_X_posture_abc",
            k_states=2,
            role_cluster="posture",
            ttl_passed=True,
        )
        g.rebind_variables(
            ["torso_pitch", "posture_stability", rec.node_id],
            {"torso_pitch": 0.5, "posture_stability": 0.5, rec.node_id: 0.5},
        )
        rec.edge_pairs = [(rec.node_id, "torso_pitch", 0.35)]
        with mock.patch.dict(
            os.environ,
            {"RKK_C5_ENABLED": "1", "RKK_PROMOTE_MIN_WORLDS": "2"},
        ):
            e1 = promote_to_universal_concept(rec, g, world_id="humanoid", force=True)
            self.assertIsNotNone(e1)
            assert e1 is not None
            self.assertEqual(e1.worlds, ["humanoid"])
            e2 = promote_to_universal_concept(rec, g, world_id="humanoid_variant", force=True)
            assert e2 is not None
            self.assertGreaterEqual(len(e2.worlds), promote_min_worlds())
            self.assertIn(e2.role_type, learned_roles)

    def tearDown(self) -> None:
        reset_learned_roles()


if __name__ == "__main__":
    unittest.main()
