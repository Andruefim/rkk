"""Regression hooks for Strong AI architecture plan (phases C, E, F, G, I)."""
from __future__ import annotations

import os
import unittest


class PlanHookTests(unittest.TestCase):
    def test_predicted_efference_veto_triggers(self) -> None:
        os.environ["RKK_VALUE_VETO"] = "1"
        from engine.value_layer import efference_predicted_veto

        pred_bad = {"com_z": 0.05, "posture_stability": 0.5}
        b, msg = efference_predicted_veto("intent_stride", pred_bad, fixed_root=False)
        self.assertTrue(b)
        self.assertIn("efference", msg)

        pred_ok = {"com_z": 0.72, "posture_stability": 0.62}
        self.assertFalse(
            efference_predicted_veto("intent_stride", pred_ok, fixed_root=False)[0]
        )

    def test_predicted_body_critical_threshold(self) -> None:
        from engine.value_layer import predicted_body_critical

        self.assertTrue(predicted_body_critical({"com_z": 0.1}))
        self.assertFalse(predicted_body_critical({"com_z": 0.8, "posture_stability": 0.7}))

    def test_pearl_context_task_embedding_shape(self) -> None:
        from engine.context_posterior import RollingObservationPosterior

        cp = RollingObservationPosterior(["a", "b"], k=4)
        cp.push({"a": 0.5, "b": 0.5}, {"g": -9.81, "mu": 0.8})
        te = cp.task_embedding()
        self.assertGreater(len(te), 2)

    def test_phase_c_tick_doc_mentions_contract(self) -> None:
        from engine.features.simulation.mixin_tick import SimulationTickMixin

        doc = (SimulationTickMixin.__doc__ or "") + (
            SimulationTickMixin.tick_step.__doc__ or ""
        )
        self.assertIn("Phase C", doc)

    def test_sz_head_on_causal_gnn_core(self) -> None:
        import torch
        from engine.causal_gnn import CausalGNNCore

        c = CausalGNNCore(6, torch.device("cpu"))
        self.assertTrue(hasattr(c, "sz_head_z"))


if __name__ == "__main__":
    unittest.main()
