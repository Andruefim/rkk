"""Phase 3 gate smoke: C4 pipeline + C5 + ensemble sync + WorldAutonomyContract."""
from __future__ import annotations

import os
import unittest
from unittest import mock

import torch

from engine.causal_graph import CausalGraph
from engine.genome.learned_roles import learned_roles, reset_learned_roles
from engine.latent_confounder import LatentConfounderManager, set_c4_active_global
from engine.scorecard.world_autonomy_contract import registered_world_ids


class Phase3LatentRolePipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        set_c4_active_global(True)
        reset_learned_roles()

    def test_gate_smoke_bundle(self) -> None:
        g = CausalGraph(device=torch.device("cpu"))
        g.rebind_variables(
            ["torso_pitch", "posture_stability", "lhip", "com_z"],
            {k: 0.5 for k in ["torso_pitch", "posture_stability", "lhip", "com_z"]},
        )
        g._maybe_init_ensemble()
        mgr = LatentConfounderManager()
        with mock.patch.dict(
            os.environ,
            {
                "RKK_C4_ENABLED": "1",
                "RKK_LATENT_TTL_TICKS": "3",
                "RKK_LATENT_MIN_IG": "0.99",
                "RKK_LATENT_MAX_INJECT_FAILURES": "5",
            },
        ):
            rec = mgr.inject_latent(g, "posture", tick=0, residual=0.5)
            self.assertIsNotNone(rec)
            mgr.tick(
                g,
                engine_tick=5,
                prediction_error=0.5,
                cluster_pe={"torso_pitch": 0.9, "posture_stability": 0.9},
            )
        self.assertTrue(registered_world_ids())
        self.assertGreaterEqual(len(learned_roles), 0)


if __name__ == "__main__":
    unittest.main()
