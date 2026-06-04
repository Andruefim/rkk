"""Track C4/C4b: latent confounder inject, EM, TTL, ensemble sync."""
from __future__ import annotations

import os
import unittest
from unittest import mock

import torch

from engine.causal_graph import CausalGraph
from engine.latent_confounder import (
    LatentConfounderManager,
    LatentRecord,
    c4_active_global,
    collect_language_context,
    language_state_prior,
    set_c4_active_global,
)
from engine.role_types import ROLE_POSTURE, ROLE_PROPRIOCEPTIVE


class LatentConfounderTests(unittest.TestCase):
    def setUp(self) -> None:
        set_c4_active_global(True)

    def _graph_with_roles(self) -> CausalGraph:
        g = CausalGraph(device=torch.device("cpu"))
        ids = ["torso_pitch", "posture_stability", "lhip", "rhip", "com_z"]
        vals = {k: 0.5 for k in ids}
        g.rebind_variables(ids, vals)
        g.set_env_preset("humanoid")
        g._maybe_init_ensemble()
        return g

    def test_inject_and_em_value(self) -> None:
        g = self._graph_with_roles()
        mgr = LatentConfounderManager()
        with mock.patch.dict(
            os.environ,
            {"RKK_C4_ENABLED": "1", "RKK_LATENT_EM_WINDOW": "8"},
        ):
            rec = mgr.inject_latent(g, ROLE_POSTURE, tick=10, residual=0.5, k_states=2)
        self.assertIsNotNone(rec)
        assert rec is not None
        obs_high = {nid: 0.85 for nid in rec.target_nodes}
        obs_high[rec.node_id] = 0.5
        v0 = mgr.infer_latent_value(obs_high, rec, lang_text="")
        obs_low = {nid: 0.15 for nid in rec.target_nodes}
        v1 = mgr.infer_latent_value(obs_low, rec, lang_text="")
        self.assertIn(v0, (0, 1))
        self.assertIn(v1, (0, 1))

    def test_language_prior_shifts_posterior(self) -> None:
        rec = LatentRecord(
            node_id="latent_X_posture_ab12",
            k_states=2,
            target_nodes=["torso_pitch", "posture_stability"],
        )
        obs = {"torso_pitch": 0.8, "posture_stability": 0.82}
        with mock.patch.dict(
            os.environ,
            {
                "RKK_LATENT_LANG_PRIOR_WEIGHT": "0.10",
                "RKK_LATENT_LANG_PRIOR_MIN_CORR": "0.25",
            },
        ):
            off = language_state_prior(
                "torso posture stability high",
                rec.target_nodes,
                obs,
                2,
            )
        self.assertGreater(float(off.max()), 0.0)
        with mock.patch.dict(os.environ, {"RKK_LATENT_LANG_PRIOR_WEIGHT": "0"}):
            off0 = language_state_prior("torso", rec.target_nodes, obs, 2)
        self.assertAlmostEqual(float(off0.sum()), 0.0)

    def test_ttl_prune_and_c4_disable_after_failures(self) -> None:
        g = self._graph_with_roles()
        mgr = LatentConfounderManager()
        with mock.patch.dict(
            os.environ,
            {
                "RKK_C4_ENABLED": "1",
                "RKK_LATENT_TTL_TICKS": "5",
                "RKK_LATENT_MIN_IG": "0.99",
                "RKK_LATENT_MAX_INJECT_FAILURES": "2",
            },
        ):
            rec = mgr.inject_latent(g, ROLE_PROPRIOCEPTIVE, tick=0, residual=0.4)
        self.assertIsNotNone(rec)
        d0 = g._d
        with mock.patch.dict(
            os.environ,
            {
                "RKK_C4_ENABLED": "1",
                "RKK_LATENT_TTL_TICKS": "5",
                "RKK_LATENT_MIN_IG": "0.99",
                "RKK_LATENT_MAX_INJECT_FAILURES": "1",
                "RKK_LATENT_K_RETRY": "0",
            },
        ):
            mgr.tick(
                g,
                engine_tick=10,
                prediction_error=0.4,
                obs={"lhip": 0.5, "rhip": 0.5},
                cluster_pe={"lhip": 0.5, "rhip": 0.5},
            )
        self.assertLessEqual(g._d, d0)
        self.assertFalse(mgr.c4_active)

    def test_ensemble_latent_edges_synced(self) -> None:
        g = self._graph_with_roles()
        assert g._ensemble is not None
        mgr = LatentConfounderManager()
        with mock.patch.dict(os.environ, {"RKK_C4_ENABLED": "1"}):
            rec = mgr.inject_latent(g, ROLE_POSTURE, tick=1, residual=0.5)
        assert rec is not None
        nids = g._node_ids
        stacks = []
        for fr, to, w in rec.edge_pairs:
            stacks.append(
                [float(g._ensemble.W_stack[k, nids.index(fr), nids.index(to)].item()) for k in range(g._ensemble.n)]
            )
        for row in stacks:
            self.assertEqual(len(set(round(x, 4) for x in row)), 1)

    def test_collect_language_context_empty_sim(self) -> None:
        self.assertEqual(collect_language_context(None), "")


class LatentConfounderGlobalFlagTests(unittest.TestCase):
    def test_global_flag(self) -> None:
        set_c4_active_global(False)
        self.assertFalse(c4_active_global())
        set_c4_active_global(True)


if __name__ == "__main__":
    unittest.main()
