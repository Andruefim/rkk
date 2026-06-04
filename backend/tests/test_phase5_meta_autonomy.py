"""Phase 5 gate smoke: W_meta, GoalGenerator, CurriculumGraph, scorecard schema."""
from __future__ import annotations

import os
import unittest
from unittest import mock

import torch

from engine.causal_graph import CausalGraph
from engine.curriculum_graph import CurriculumGraph
from engine.goal_generator import GoalGenerator
from engine.meta_causal import WMetaEnsemble, MetaObservation
from engine.scorecard.autonomy_scorecard import build_scorecard


class Phase5MetaAutonomyGateTests(unittest.TestCase):
    def test_w_meta_lr_explore_curriculum_effects(self) -> None:
        device = torch.device("cpu")
        w = WMetaEnsemble(device)
        for i in range(20):
            w.observe(
                MetaObservation(
                    learning_rate_eff=0.1 + 0.02 * i,
                    exploration_rate=0.05 * i,
                    curriculum_phase=0.05 * i,
                    wm_lr_mult=1.0,
                    success_rate=0.2 + 0.03 * i,
                ),
                tick=i * 50,
            )
        effects = w.effect_observable()
        self.assertIn("learning_rate_eff_effect", effects)
        self.assertIn("exploration_rate_effect", effects)
        self.assertIn("curriculum_phase_effect", effects)

    def test_goal_generator_curriculum_graph_roundtrip(self) -> None:
        gen = GoalGenerator()
        cg = CurriculumGraph()
        g = CausalGraph(device=torch.device("cpu"))
        g.rebind_variables(["a", "b"], {"a": 0.5, "b": 0.5})
        gen.load_dict(gen.to_dict())
        cg.load_dict(cg.to_dict())
        self.assertIsInstance(gen._active, list)
        self.assertIsInstance(cg._nodes, dict)

    def test_saturation_guard_not_same_goal_three_times(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "RKK_GOAL_DIVERSITY_WINDOW": "10",
                "RKK_GOAL_COOLDOWN_MAX": "3",
                "RKK_GOAL_SATURATION_FRAC": "0.50",
                "RKK_GOAL_WMETA_MIN_SUCCESS": "0.0",
                "RKK_GOAL_MAX_ACTIVE": "10",
            },
        ):
            gen = GoalGenerator()
            g = CausalGraph(device=torch.device("cpu"))
            g.rebind_variables(["v1", "v2", "v3"], {"v1": 0.5, "v2": 0.5, "v3": 0.5})
            w = WMetaEnsemble(torch.device("cpu"))
            proposals = []
            for t in range(5):
                c = gen.propose(g, w, tick=t * 200)
                if c:
                    proposals.append(c.var_id)
            if len(proposals) >= 4:
                window = proposals[-10:]
                from collections import Counter
                counts = Counter(window)
                top, n = counts.most_common(1)[0]
                self.assertLessEqual(n, 3, f"{top} proposed {n} times in window")

    def test_scorecard_phase5_schema(self) -> None:
        card = build_scorecard(
            {
                "discovery_new_frac": 0.65,
                "meta_prediction_error": 0.10,
                "phase5": {
                    "goal_generator": {
                        "autonomous_goals_crossworld_pass": True,
                    },
                },
            },
            worlds=["humanoid"],
        )
        self.assertIn("meta_prediction_error", card)
        self.assertIn("autonomous_goals_crossworld_pass", card)
        self.assertIn("pass_agi_extended", card)
        self.assertIn("thresholds", card)
