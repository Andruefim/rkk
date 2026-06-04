"""Track D Phase 3: WorldAutonomyContract registry and humanoid A1/A4 mapping."""
from __future__ import annotations

import unittest

from engine.scorecard.autonomy_scorecard import build_scorecard, default_thresholds
from engine.scorecard.world_autonomy_contract import (
    extract_a1_a4,
    get_contract,
    registered_world_ids,
)


class WorldAutonomyContractTests(unittest.TestCase):
    def test_registered_worlds(self) -> None:
        worlds = registered_world_ids()
        for wid in ("humanoid", "cartpole", "grid_nav"):
            self.assertIn(wid, worlds)
        self.assertIsNotNone(get_contract("humanoid"))

    def test_humanoid_a1_a4_from_system2(self) -> None:
        snap = {
            "tick": 1000,
            "system2": {
                "script_override_frac_post_warmup": 0.12,
                "emergency_override_frac_post_warmup": 0.08,
            },
        }
        a1, a4, applicable = extract_a1_a4("humanoid", snap)
        self.assertTrue(applicable)
        self.assertAlmostEqual(a1, 0.12, places=4)
        self.assertAlmostEqual(a4, 0.08, places=4)
        th = default_thresholds()
        card = build_scorecard(snap, worlds=["humanoid"])
        h = card["worlds"]["humanoid"]
        self.assertTrue(h["a1_pass"])
        self.assertTrue(h["a4_pass"])
        self.assertLess(a1, th["a1_max"])
        self.assertLess(a4, th["a4_max"])

    def test_humanoid_maps_frozen_phase2_probes(self) -> None:
        contract = get_contract("humanoid")
        assert contract is not None
        self.assertEqual(contract.a1_probe_key, "s2_override_frac")
        self.assertEqual(contract.a4_probe_key, "fallen_override_frac_post_800")

    def test_stub_world_metrics_present(self) -> None:
        card = build_scorecard({}, worlds=["cartpole", "grid_nav"])
        self.assertIn("cartpole", card["worlds"])
        self.assertIn("a1_probe", card["worlds"]["cartpole"])
        self.assertIn("a4_probe", card["worlds"]["grid_nav"])


if __name__ == "__main__":
    unittest.main()
