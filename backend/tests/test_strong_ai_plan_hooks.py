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

    def test_context_posterior_remap_preserves_aligned_components(self) -> None:
        from engine.context_posterior import RollingObservationPosterior

        cp = RollingObservationPosterior(["a", "b"], k=4)
        cp.push({"a": 0.2, "b": 0.3})
        cp.remap_node_ids(["a", "b", "c"])
        cp.push({"a": 0.4, "b": 0.5, "c": 0.9})
        mz = cp.mean_z()
        self.assertEqual(len(mz), 3)
        self.assertAlmostEqual(float(mz[0]), 0.3)

    def test_phase_c_tick_doc_mentions_contract(self) -> None:
        from engine.features.simulation.mixin_tick import SimulationTickMixin

        doc = SimulationTickMixin.__doc__ or ""
        ts = getattr(SimulationTickMixin, "tick_step", None)
        if ts is not None and getattr(ts, "__doc__", None):
            doc = doc + (ts.__doc__ or "")
        self.assertIn("Phase C", doc)

    def test_sz_head_outputs_differ_across_inputs(self) -> None:
        import torch
        from engine.causal_gnn import CausalGNNCore

        c = CausalGNNCore(6, torch.device("cpu"))
        hdim = 6 * c.hidden
        fh1 = torch.randn(4, hdim)
        fh2 = fh1 + 2.0
        z1 = c.sz_head_z(fh1)
        z2 = c.sz_head_z(fh2)
        self.assertFalse(torch.allclose(z1, z2, atol=1e-5))

    def test_modality_single_source_vestibular(self) -> None:
        from engine.precision_channels import modality_of_node
        from engine.precision_groups import modality_group_for_var

        self.assertEqual(
            modality_of_node("vestibular_gz"),
            modality_group_for_var("vestibular_gz"),
        )
        self.assertEqual(modality_group_for_var("vestibular_gz"), "vestibular")

    def test_floor_friction_is_sandbox_not_vestibular(self) -> None:
        from engine.precision_groups import modality_group_for_var

        self.assertEqual(modality_group_for_var("floor_friction"), "sandbox")

    def test_sleep_mocap_dreams_env_toggle(self) -> None:
        from engine import sleep_consolidation as sc

        key = "RKK_SLEEP_MOCAP_DREAMS"
        prev = os.environ.get(key)
        try:
            os.environ[key] = "0"
            self.assertFalse(sc.mocap_dreams_enabled())
            os.environ[key] = "1"
            self.assertTrue(sc.mocap_dreams_enabled())
        finally:
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev


if __name__ == "__main__":
    unittest.main()
