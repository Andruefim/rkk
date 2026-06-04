"""Track A Phase 0: post-FR alpha decay and WM LR window."""
from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from engine.causal_graph import CausalGraph, Edge
from engine.graph_ensemble import WeightedGraphEnsemble
from engine.post_fr import (
    apply_post_fr_alpha_decay,
    apply_post_fr_ensemble_entropy_boost,
    post_fr_wm_lr_active,
    post_fr_wm_lr_scale,
)


class PostFrTests(unittest.TestCase):
    def test_alpha_decay_on_motor_edges(self) -> None:
        motor = Edge("intent_stride", "posture_stability", 0.3, 0.9)
        other = Edge("slot_a", "slot_b", 0.2, 0.8)
        stub = SimpleNamespace(edges=[motor, other], _invalidate_cache=lambda: None)
        with mock.patch.dict(os.environ, {"RKK_POST_FR_ALPHA_DECAY": "0.4"}):
            n = apply_post_fr_alpha_decay(stub)
        self.assertEqual(n, 1)
        self.assertAlmostEqual(motor.alpha_trust, 0.5, places=3)
        self.assertAlmostEqual(other.alpha_trust, 0.8, places=3)

    def test_ensemble_entropy_boost(self) -> None:
        g = CausalGraph(device=torch.device("cpu"))
        g.rebind_variables(["a", "b"], {"a": 0.5, "b": 0.5})
        g._ensemble = WeightedGraphEnsemble(2, torch.device("cpu"), n=3)
        h0 = g._ensemble.entropy()
        with mock.patch.dict(os.environ, {"RKK_POST_FR_ENSEMBLE_ENT_BOOST": "0.5"}):
            h1 = apply_post_fr_ensemble_entropy_boost(g)
        self.assertIsNotNone(h1)
        assert h1 is not None
        self.assertGreaterEqual(h1, h0)

    def test_wm_lr_window(self) -> None:
        sim = SimpleNamespace(tick=100, _post_fr_last_release_tick=80)
        with mock.patch.dict(
            os.environ,
            {"RKK_POST_FR_WM_LR_MULT": "2.5", "RKK_POST_FR_WM_LR_TICKS": "450"},
        ):
            self.assertAlmostEqual(post_fr_wm_lr_scale(sim), 2.5, places=3)
            self.assertTrue(post_fr_wm_lr_active(sim))
        sim.tick = 600
        with mock.patch.dict(os.environ, {"RKK_POST_FR_WM_LR_TICKS": "450"}):
            self.assertAlmostEqual(post_fr_wm_lr_scale(sim), 1.0, places=3)
