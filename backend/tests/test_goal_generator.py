"""Track G Phase 5: GoalGenerator + CausalNoveltyScore."""
from __future__ import annotations

import os
import unittest
from unittest import mock

import torch

from engine.causal_graph import CausalGraph
from engine.goal_generator import GoalGenerator, causal_novelty_score
from engine.meta_causal import WMetaEnsemble
class GoalGeneratorTests(unittest.TestCase):
    def _graph(self) -> CausalGraph:
        g = CausalGraph(device=torch.device("cpu"))
        ids = ["torso_pitch", "posture_stability", "intent_stride", "lhip"]
        g.rebind_variables(ids, {k: 0.5 for k in ids})
        g.set_env_preset("humanoid")
        return g

    def test_causal_novelty_score_keys(self) -> None:
        g = self._graph()
        scores = causal_novelty_score(g, g.role_type_map())
        self.assertGreaterEqual(len(scores), 4)
        for vid in g._node_ids:
            self.assertIn(vid, scores)

    def test_saturation_guard_blocks_repeat(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "RKK_GOAL_GEN_ENABLED": "1",
                "RKK_GOAL_DIVERSITY_WINDOW": "10",
                "RKK_GOAL_COOLDOWN_MAX": "3",
                "RKK_GOAL_SATURATION_FRAC": "0.50",
                "RKK_GOAL_WMETA_MIN_SUCCESS": "0.0",
                "RKK_GOAL_MAX_ACTIVE": "10",
            },
        ):
            gen = GoalGenerator()
            g = self._graph()
            w = WMetaEnsemble(torch.device("cpu"))
            top_var = max(causal_novelty_score(g).items(), key=lambda x: -x[1])[0]
            for _ in range(6):
                gen._recent.append(top_var)
            cand = gen.propose(g, w, tick=100)
            if cand is not None:
                self.assertNotEqual(cand.var_id, top_var)

    def test_w_meta_filter_rejects_low_success(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "RKK_META_CAUSAL_ENABLED": "1",
                "RKK_GOAL_WMETA_MIN_SUCCESS": "0.99",
                "RKK_GOAL_MAX_ACTIVE": "3",
            },
        ):
            gen = GoalGenerator()
            g = self._graph()
            w = WMetaEnsemble(torch.device("cpu"))
            cand = gen.propose(g, w, tick=200)
            self.assertIsNone(cand)
            blocked = [b for b in gen._blocked_log if b.get("reason") == "w_meta_reject"]
            self.assertGreater(len(blocked), 0)

    def test_persistence_roundtrip(self) -> None:
        gen = GoalGenerator()
        g = self._graph()
        w = WMetaEnsemble(torch.device("cpu"))
        with mock.patch.dict(os.environ, {"RKK_GOAL_WMETA_MIN_SUCCESS": "0.0"}):
            gen.propose(g, w, tick=1)
        data = gen.to_dict()
        gen2 = GoalGenerator()
        gen2.load_dict(data)
        self.assertEqual(len(gen2._active), len(gen._active))
