"""Phase 6b gate smoke: EWC stable-edge Fisher, CausalHealthMonitor, continual metrics."""
from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np
import torch

from engine.causal_graph import CausalGraph
from engine.causal_health_monitor import CausalHealthMonitor
from engine.elastic_role_protector import ElasticRoleProtector, ewc_enabled
from engine.features.humanoid.constants import VAR_NAMES
from engine.scorecard.autonomy_scorecard import build_scorecard


def _graph_with_stable_edges() -> CausalGraph:
    g = CausalGraph(torch.device("cpu"))
    g.set_env_preset("humanoid")
    for i, nid in enumerate(VAR_NAMES[:40]):
        g.set_node(nid, 0.4 + 0.01 * i)
    g._rebuild_core()
    ids = list(g._node_ids)
    for t in range(30):
        obs = {nid: float(0.4 + 0.01 * (t % 7)) for nid in ids}
        g.record_observation(obs)
    for i in range(len(ids)):
        for j in range(len(ids)):
            if i != j and (i + j) % 5 == 0:
                fr, to = ids[i], ids[j]
                g.set_edge(fr, to, 0.12, alpha=0.7)
                g._edge_age[(fr, to)] = 250
    g.tick_edge_ages()
    return g


class Phase6bContinualTests(unittest.TestCase):
    def test_ewc_stable_edges_and_recompute(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "RKK_EWC_ENABLED": "1",
                "RKK_EWC_STABLE_AGE_MIN": "200",
                "RKK_EWC_GRAPH_CHANGE_THRESH": "0.20",
            },
        ):
            self.assertTrue(ewc_enabled())
            g = _graph_with_stable_edges()
            prot = ElasticRoleProtector()
            fisher = prot.compute_fisher(g)
            self.assertGreater(prot._stable_edge_count, 0)
            self.assertEqual(float(fisher.sum()), float(fisher.sum()))

            prot.anchor_weights(g)
            with torch.no_grad():
                W = g._core.W_masked()[: g._d, : g._d].clone()
                fisher = prot._fisher
                assert fisher is not None
                nz = (fisher > 0).nonzero(as_tuple=False)
                self.assertGreater(nz.shape[0], 0)
                i, j = int(nz[0, 0]), int(nz[0, 1])
                W[i, j] = W[i, j] + 0.05
            pen = prot.ewc_penalty(W)
            self.assertGreater(float(pen.item()), 0.0)

            prot._last_hash = "old_hash"
            self.assertTrue(prot.should_recompute(g))
            prot.maybe_update(g, world_switch=True)
            self.assertGreater(prot._ewc_recompute_count, 0)

    def test_three_world_switch_forgetting_logged(self) -> None:
        with mock.patch.dict(os.environ, {"RKK_EWC_ENABLED": "1"}):
            prot = ElasticRoleProtector()
            g = _graph_with_stable_edges()
            prot.on_world_switch(g, "humanoid")
            prot.update_forgetting_ratio(0.8, 0.3)
            m = prot.metrics()
            self.assertIn("continual_forgetting_ratio", m)
            self.assertGreaterEqual(m["continual_forgetting_ratio"], 0.5)

    def test_health_monitor_detects_degradation(self) -> None:
        mon = CausalHealthMonitor()
        healthy = [{"discovery_new_frac": 0.7, "graph_ensemble": {"entropy": 0.5}, "meta_prediction_error": 0.05}]
        degraded = [
            {
                "discovery_new_frac": 0.1,
                "graph_ensemble": {"entropy": 0.05},
                "meta_prediction_error": 0.35,
                "cross_env_success_rate_200": 0.2,
            }
        ]
        runs = 0
        hits = 0
        for _ in range(3):
            mon._baseline_cross_sr = 0.8
            rep = mon.diagnose(healthy + degraded)
            runs += 1
            if rep.degraded:
                hits += 1
        self.assertGreaterEqual(hits / runs, 0.7)

    def test_scorecard_nonphys_a1_a4_fields(self) -> None:
        card = build_scorecard(
            {
                "pathfinder_override_frac": 0.05,
                "stuck_override_active": 0.0,
                "rule_engine_bailout_frac": 0.04,
                "constraint_violation_override": 0.0,
            },
            worlds=["grid_nav", "symbolic_control"],
        )
        for wid in ("grid_nav", "symbolic_control"):
            self.assertIn("a1_probe", card["worlds"][wid])
            self.assertIn("a4_probe", card["worlds"][wid])
