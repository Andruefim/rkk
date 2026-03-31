

"""
RKKAgent — Python/PyTorch.

Улучшение 5 (Seed Diversity): каждый агент получает разную активацию.
  Agent 0: ReLU  — жёсткие логические связи
  Agent 1: GELU  — вероятностные мягкие связи
  Agent 2: Tanh  — насыщаемые связи

Улучшение 4: pinned memory для буферов наблюдений.

Исправления:
  - peak_discovery_rate: монотонно растёт, никогда не снижается
  - activation корректно передаётся в snapshot
"""
from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from collections import deque

from engine.causal_graph import CausalGraph
from engine.environment  import Environment

ACTIVATION_MAP = {
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "tanh": nn.Tanh(),
}

ACTIVATIONS = ["relu", "gelu", "tanh"]


class System1(nn.Module):
    """
    Амортизированный эпистемический скорер.
    Предсказывает E[IG] для пары (variable, target) без полного перебора.
    Input:  [edge_weight, alpha_trust, node_value_from, node_value_to, uncertainty]
    Output: ожидаемый information gain [0, 1]
    """
    def __init__(self, activation: str = "relu"):
        super().__init__()
        act = ACTIVATION_MAP.get(activation, nn.ReLU())
        self.net = nn.Sequential(
            nn.Linear(5, 32), act,
            nn.Linear(32, 16), act,
            nn.Linear(16, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RKKAgent:
    def __init__(self, agent_id: int, name: str, env: Environment, device: torch.device):
        self.id         = agent_id
        self.name       = name
        self.env        = env
        self.device     = device
        self.activation = ACTIVATIONS[agent_id % 3]  # Seed Diversity

        self.graph   = CausalGraph(device)
        self.system1 = System1(self.activation).to(device)
        self.optim   = torch.optim.Adam(self.system1.parameters(), lr=3e-4)

        self._cg_history:  deque[float] = deque(maxlen=20)
        self._phi_history: deque[float] = deque(maxlen=30)
        self._total_interventions = 0
        self._last_do             = "—"
        self._last_result: dict | None = None

        # ── Peak discovery rate: монотонно растёт, никогда не снижается ──────
        self._peak_discovery_rate: float = 0.0

        self._bootstrap()

    # ── Bootstrap ─────────────────────────────────────────────────────────────
    def _bootstrap(self):
        for var_id, val in self.env.variables.items():
            self.graph.set_node(var_id, val)

        gt = self.env.gt_edges()
        for e in gt[:2]:
            noisy_w = e["weight"] * 0.3 + (np.random.rand() - 0.5) * 0.4
            self.graph.set_edge(e["from_"], e["to"], noisy_w, alpha=0.06)

        var_ids = self.env.variable_ids
        if len(var_ids) >= 4:
            self.graph.set_edge(var_ids[1], var_ids[3],  0.35, alpha=0.05)
            self.graph.set_edge(var_ids[2], var_ids[0], -0.20, alpha=0.04)

    # ── Epistemic scoring (System 1) ───────────────────────────────────────────
    def score_interventions(self) -> list[dict]:
        var_ids = self.env.variable_ids
        features_list: list[torch.Tensor] = []
        candidates:    list[dict]         = []

        for v_from in var_ids:
            for v_to in var_ids:
                if v_from == v_to:
                    continue
                uncertainty = self.graph.edge_uncertainty(v_from, v_to)
                edge_w      = next(
                    (e.weight      for e in self.graph.edges if e.from_ == v_from and e.to == v_to),
                    0.0
                )
                alpha = next(
                    (e.alpha_trust for e in self.graph.edges if e.from_ == v_from and e.to == v_to),
                    0.05
                )
                val_from = self.graph.nodes.get(v_from, 0.5)
                val_to   = self.graph.nodes.get(v_to,   0.5)

                feat = torch.tensor(
                    [edge_w, alpha, val_from, val_to, uncertainty],
                    dtype=torch.float32, device=self.device
                )
                features_list.append(feat)
                candidates.append({
                    "variable":    v_from,
                    "target":      v_to,
                    "value":       0.9 if np.random.rand() > 0.5 else 0.1,
                    "uncertainty": uncertainty,
                })

        if not features_list:
            return []

        batch = torch.stack(features_list)
        with torch.no_grad():
            ig_scores = self.system1(batch).squeeze(-1)

        for i, cand in enumerate(candidates):
            cand["expected_ig"] = ig_scores[i].item()

        return sorted(candidates, key=lambda x: -x["expected_ig"])

    # ── Один шаг ──────────────────────────────────────────────────────────────
    def step(self) -> dict:
        scores = self.score_interventions()
        if not scores:
            return {}

        best  = scores[0]
        var   = best["variable"]
        value = best["value"]

        mdl_before = self.graph.mdl_size
        predicted  = self.graph.propagate(var, value)
        observed   = self.env.intervene(var, value)

        updated_edges: list[str] = []
        pruned_edges:  list[str] = []

        for node_id, obs_val in observed.items():
            if node_id == var:
                continue

            input_delta = value - self.graph.nodes.get(var, value)
            if abs(input_delta) < 1e-3:
                continue

            empirical_w = float(
                np.tanh(
                    (obs_val - self.graph.nodes.get(node_id, obs_val))
                    / (input_delta + 1e-4)
                )
            )

            if abs(empirical_w) > 0.08:
                existing  = next(
                    (e for e in self.graph.edges if e.from_ == var and e.to == node_id),
                    None
                )
                prev_alpha = existing.alpha_trust if existing else 0.0
                new_alpha  = min(0.98, prev_alpha + 0.12 * (1 - prev_alpha))
                self.graph.set_edge(var, node_id, empirical_w, new_alpha)
                updated_edges.append(f"{var}→{node_id}")
            else:
                existing = next(
                    (e for e in self.graph.edges if e.from_ == var and e.to == node_id),
                    None
                )
                if existing and existing.alpha_trust < 0.3:
                    existing.alpha_trust = max(0, existing.alpha_trust - 0.08)
                    if existing.alpha_trust < 0.02:
                        self.graph.remove_edge(var, node_id)
                        pruned_edges.append(f"PRUNED:{var}→{node_id}")

            self.graph.nodes[node_id] = obs_val

        self.graph.nodes[var] = value

        mdl_after         = self.graph.mdl_size
        compression_delta = mdl_before - mdl_after
        self._cg_history.append(compression_delta)

        # Обучаем System 1
        pred_val  = predicted.get(list(observed.keys())[-1], 0.5)
        obs_val_  = list(observed.values())[-1]
        actual_ig = float(abs(pred_val - obs_val_))

        s1_input = torch.tensor(
            [best.get("expected_ig", 0), best["uncertainty"], 0.0, 0.0, best["uncertainty"]],
            dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        s1_target = torch.tensor([[actual_ig]], dtype=torch.float32, device=self.device)

        self.optim.zero_grad()
        s1_pred = self.system1(s1_input)
        s1_loss = nn.functional.mse_loss(s1_pred, s1_target)
        s1_loss.backward()
        self.optim.step()

        self._total_interventions += 1
        self._last_do = f"do({var}={value:.2f})"

        # ── Обновляем peak discovery rate ──────────────────────────────────
        current_dr = self.discovery_rate
        if current_dr > self._peak_discovery_rate:
            self._peak_discovery_rate = current_dr

        self._last_result = {
            "variable":           var,
            "value":              value,
            "compression_delta":  compression_delta,
            "updated_edges":      updated_edges,
            "pruned_edges":       pruned_edges,
            "prediction_error":   float(np.mean([
                abs(predicted.get(k, 0) - v) for k, v in observed.items()
            ])),
        }
        return self._last_result

    # ── Φ_approx ──────────────────────────────────────────────────────────────
    def phi_approx(self) -> float:
        if len(self._phi_history) < 4:
            return 0.1
        hist    = torch.tensor(list(self._phi_history), device=self.device).unsqueeze(-1)
        var_self = hist.var().item()
        return float(np.clip(var_self * 10, 0, 1))

    def record_phi(self, val: float):
        self._phi_history.append(val)

    # ── Demon disruption ──────────────────────────────────────────────────────
    def demon_disrupt(self) -> str:
        vulnerable = [e for e in self.graph.edges if e.alpha_trust < 0.5]
        if not vulnerable:
            return "no vulnerable edges"
        edge = vulnerable[int(np.random.randint(len(vulnerable)))]
        edge.alpha_trust = max(0.02, edge.alpha_trust - 0.12)
        edge.weight     += (np.random.rand() - 0.5) * 0.2
        self.graph._mdl_cache = None
        return f"corrupted {edge.from_}→{edge.to}"

    # ── Properties ────────────────────────────────────────────────────────────
    @property
    def compression_gain(self) -> float:
        if not self._cg_history:
            return 0.0
        return float(np.mean(list(self._cg_history)))

    @property
    def discovery_rate(self) -> float:
        """Текущий discovery rate (может флуктуировать)."""
        return self.env.discovery_rate([
            {"from_": e.from_, "to": e.to, "weight": e.weight}
            for e in self.graph.edges
        ])

    @property
    def peak_discovery_rate(self) -> float:
        """Монотонно растущий максимум — никогда не снижается."""
        return self._peak_discovery_rate

    # ── Snapshot ──────────────────────────────────────────────────────────────
    def snapshot(self) -> dict:
        current_dr = self.discovery_rate

        # Обновляем peak здесь тоже на случай если step() не вызвался
        if current_dr > self._peak_discovery_rate:
            self._peak_discovery_rate = current_dr

        return {
            "id":                    self.id,
            "name":                  self.name,
            "env_type":              self.env.preset,
            "activation":            self.activation,
            "graph_mdl":             round(self.graph.mdl_size, 3),
            "compression_gain":      round(self.compression_gain, 4),
            "alpha_mean":            round(self.graph.alpha_mean, 3),
            "phi":                   round(self.phi_approx(), 3),
            "node_count":            len(self.graph.nodes),
            "edge_count":            len(self.graph.edges),
            "total_interventions":   self._total_interventions,
            "last_do":               self._last_do,
            # ── Оба нужны simulation.py ──────────────────────────────────────
            "discovery_rate":        round(current_dr, 3),
            "peak_discovery_rate":   round(self._peak_discovery_rate, 3),
            # ── Рёбра для визуализации ───────────────────────────────────────
            "edges": [
                {
                    "from_":              e.from_,
                    "to":                 e.to,
                    "weight":             round(e.weight, 3),
                    "alpha_trust":        round(e.alpha_trust, 3),
                    "intervention_count": e.intervention_count,
                }
                for e in self.graph.edges
            ],
        }