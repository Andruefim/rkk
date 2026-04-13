"""Simulation mixin: GNN → visual cortex."""
from __future__ import annotations

from engine.features.simulation.mixin_imports import *


class SimulationVisionPredictorMixin:
    def _feed_gnn_prediction_to_visual(self):
        """Передаём текущий GNN-прогноз в visual env для predictive coding."""
        if self._visual_env is None or self.agent.graph._core is None:
            return
        try:
            current_obs = self._visual_env.observe()
            node_ids    = self.agent.graph._node_ids
            slot_ids    = [f"slot_{k}" for k in range(self._visual_env.n_slots)]
            values_list = [current_obs.get(sid, 0.5) for sid in slot_ids]
            current_t   = torch.tensor(values_list, dtype=torch.float32, device=self.device)
            # Прогоняем текущие значения через GNN (опционально Neural ODE sub-steps)
            with torch.inference_mode():
                full_state = torch.tensor(
                    [self.agent.graph.nodes.get(n, 0.5) for n in node_ids],
                    dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                z = torch.zeros_like(full_state)
                pred_full = integrate_world_model_step(
                    self.agent.graph._core, full_state, z
                ).squeeze(0)
            # Выбираем только slot_ переменные
            slot_pred = torch.tensor(values_list, dtype=torch.float32, device=self.device)
            for i, sid in enumerate(slot_ids):
                if sid in node_ids:
                    idx = node_ids.index(sid)
                    if idx < len(pred_full):
                        p = float(pred_full[idx].item())
                        slot_pred[i] = min(0.95, max(0.05, p))
            self._visual_env.set_gnn_prediction(slot_pred)
        except Exception:
            pass
