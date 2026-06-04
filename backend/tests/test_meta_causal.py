"""Track F Phase 5: W_meta ensemble and do-calculus."""
from __future__ import annotations

import os
import unittest
from unittest import mock

import torch

from engine.meta_causal import (
    META_INPUTS,
    WMetaEnsemble,
    build_meta_observation,
    meta_causal_enabled,
    meta_do_safe,
)
from engine.causal_graph import CausalGraph


class MetaCausalTests(unittest.TestCase):
    def test_do_learning_rate_predicts_finite_pe(self) -> None:
        device = torch.device("cpu")
        w = WMetaEnsemble(device)
        obs = build_meta_observation(
            type("A", (), {"graph": CausalGraph(device), "_last_notears_loss": None, "_last_result": {}})(),
            tick=50,
            curriculum_step=2,
            success_rate=0.7,
        )
        with mock.patch.dict(os.environ, {"RKK_META_CAUSAL_ENABLED": "1"}):
            self.assertTrue(meta_causal_enabled())
        result = w.do_intervention("learning_rate_eff", 0.75, obs)
        self.assertIn(result.variable, META_INPUTS)
        self.assertTrue(0.0 <= result.meta_prediction_error <= 1.0)
        self.assertTrue(0.0 <= result.predicted_success <= 1.0)

    def test_meta_nodes_effect_observable_after_observations(self) -> None:
        device = torch.device("cpu")
        w = WMetaEnsemble(device)
        from engine.meta_causal import MetaObservation

        for i in range(12):
            w.observe(
                MetaObservation(
                    learning_rate_eff=0.2 + 0.05 * i,
                    exploration_rate=0.1 * i,
                    curriculum_phase=0.1 * i,
                    wm_lr_mult=1.0,
                    success_rate=0.3 + 0.04 * i,
                ),
                tick=i * 50,
            )
        effects = w.effect_observable()
        for key in META_INPUTS:
            self.assertIn(f"{key}_effect", effects)

    def test_meta_do_safe_default(self) -> None:
        self.assertTrue(meta_do_safe())

    def test_roundtrip_persistence(self) -> None:
        device = torch.device("cpu")
        w = WMetaEnsemble(device)
        w.do_intervention("exploration_rate", 0.5, build_meta_observation(
            type("A", (), {"graph": CausalGraph(device), "_last_notears_loss": None, "_last_result": {}})(),
            tick=1,
        ))
        data = w.to_dict()
        w2 = WMetaEnsemble(device)
        w2.load_dict(data)
        self.assertEqual(w.W_stack.shape, w2.W_stack.shape)
