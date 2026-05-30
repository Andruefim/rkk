from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from engine.agent import RKKAgent


class NeurogenesisEngine:
    """
    Structural ASI: dynamic latent node growth with deferred apply.
    """

    def __init__(self, min_interventions: int = 1500, error_threshold: float = 0.25):
        self.min_interventions = min_interventions
        self.error_threshold = error_threshold
        self._last_growth_tick = 0
        self._cooldown = 2000
        self._pending_growth: dict | None = None
        self._stable_ticks = 0

    def _body_ok_for_request(self, obs: dict, *, posture_min: float = 0.75) -> bool:
        posture = float(obs.get("posture_stability", obs.get("phys_posture_stability", 0.5)))
        com_z = float(obs.get("com_z", obs.get("phys_com_z", 0.5)))
        if posture < posture_min or com_z < 0.45:
            self._stable_ticks = 0
            return False
        if posture >= posture_min:
            self._stable_ticks += 1
        else:
            self._stable_ticks = 0
        return self._stable_ticks >= 20

    def _find_stress_pair(self, agent: "RKKAgent") -> tuple[str, str, float] | None:
        graph = agent.graph
        if graph._core is None:
            return None
        with torch.no_grad():
            W_grad = graph._core.W.grad
            if W_grad is None:
                return None
            grad_norm = W_grad.abs().cpu().numpy()
            alpha_trust = graph._core.alpha_trust_matrix().cpu().numpy()
        uncertainty = 1.0 - alpha_trust
        stress_matrix = grad_norm * uncertainty
        i, j = np.unravel_index(np.argmax(stress_matrix), stress_matrix.shape)
        max_stress = float(stress_matrix[i, j])
        if max_stress < self.error_threshold:
            return None
        node_from = graph._node_ids[i]
        node_to = graph._node_ids[j]
        if "leg" in node_from or "leg" in node_to or "hip" in node_from:
            return None
        return node_from, node_to, max_stress

    def queue_growth_pair(
        self,
        agent: "RKKAgent",
        tick: int,
        src: str,
        dst: str,
        *,
        trigger: str = "manual",
    ) -> dict | None:
        if agent._total_interventions < self.min_interventions:
            return None
        if tick - self._last_growth_tick < self._cooldown:
            return None
        self._pending_growth = {
            "src_node": str(src),
            "dst_node": str(dst),
            "tick_requested": int(tick),
            "trigger": trigger,
        }
        return dict(self._pending_growth)

    def queue_growth(
        self,
        agent: "RKKAgent",
        tick: int,
        *,
        obs: dict | None = None,
        trigger: str = "stress",
    ) -> dict | None:
        if agent._total_interventions < self.min_interventions:
            return None
        if tick - self._last_growth_tick < self._cooldown:
            return None
        if obs is None:
            try:
                obs = dict(agent.env.observe())
            except Exception:
                obs = {}
        if not self._body_ok_for_request(obs):
            return None
        pair = self._find_stress_pair(agent)
        if pair is None:
            return None
        src, dst, stress = pair
        self._pending_growth = {
            "src_node": src,
            "dst_node": dst,
            "tick_requested": int(tick),
            "trigger": trigger,
            "stress": round(stress, 4),
        }
        return dict(self._pending_growth)

    def scan_and_queue(self, agent: "RKKAgent", tick: int, obs: dict | None = None) -> dict | None:
        """Legacy entry: queue only, never immediate execute."""
        return self.queue_growth(agent, tick, obs=obs, trigger="stress")

    def scan_and_grow(self, agent: "RKKAgent", tick: int, obs: dict | None = None) -> dict | None:
        """Compat: queue growth (no immediate rebind)."""
        return self.scan_and_queue(agent, tick, obs=obs)

    def apply_pending_growth(
        self,
        agent: "RKKAgent",
        tick: int,
        *,
        obs: dict | None = None,
        fixed_root: bool = False,
        stable_ticks_required: int = 20,
        posture_min: float = 0.80,
    ) -> dict | None:
        if self._pending_growth is None:
            return None
        if obs is None:
            try:
                obs = dict(agent.env.observe())
            except Exception:
                obs = {}
        posture = float(obs.get("posture_stability", obs.get("phys_posture_stability", 0.5)))
        com_z = float(obs.get("com_z", obs.get("phys_com_z", 0.5)))
        fallen = posture < 0.35 or com_z < 0.40

        can_apply = fixed_root
        if not can_apply and not fallen:
            if posture >= posture_min:
                self._stable_ticks += 1
            else:
                self._stable_ticks = 0
            can_apply = self._stable_ticks >= stable_ticks_required

        if not can_apply:
            return None

        pending = self._pending_growth
        self._pending_growth = None
        return self._execute_neurogenesis(
            agent,
            pending["src_node"],
            pending["dst_node"],
            tick,
            trigger=str(pending.get("trigger", "pending")),
        )

    def _execute_neurogenesis(
        self,
        agent: "RKKAgent",
        src_node: str,
        dst_node: str,
        tick: int,
        *,
        trigger: str = "stress",
    ) -> dict:
        graph = agent.graph
        latent_id = (
            f"latent_{src_node.split('_')[0]}_to_{dst_node.split('_')[0]}_{str(uuid.uuid4())[:4]}"
        )
        current_ids = list(graph._node_ids)
        new_ids = current_ids + [latent_id]
        values = {nid: float(graph.nodes.get(nid, 0.5)) for nid in current_ids}
        values[latent_id] = 0.5
        graph.rebind_variables(new_ids, values, preserve_state=True)
        graph.set_edge(src_node, latent_id, weight=0.4, alpha=0.1)
        graph.set_edge(latent_id, dst_node, weight=0.4, alpha=0.1)
        graph.remove_edge(src_node, dst_node)
        self._last_growth_tick = tick
        print(f"[Neurogenesis] Applied '{latent_id}' mediating {src_node} -> {dst_node} ({trigger})")
        return {
            "type": "structural_asi_growth",
            "new_node": latent_id,
            "mediated_path": f"{src_node} -> {dst_node}",
            "gnn_d_new": graph._d,
            "trigger": trigger,
        }

    @property
    def has_pending(self) -> bool:
        return self._pending_growth is not None

    def pending_snapshot(self) -> dict | None:
        return dict(self._pending_growth) if self._pending_growth else None
