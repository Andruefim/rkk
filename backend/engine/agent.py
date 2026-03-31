"""
agent.py — RKKAgent с NOTEARS-ядром.

Изменения:
  - После каждого do() записываем obs_before/obs_after в граф (record_intervention)
  - Каждые TRAIN_EVERY шагов вызываем graph.train_step() → NOTEARS backprop
  - Alpha-trust теперь производная от градиентной уверенности матрицы W
  - Epistemic scoring использует edge_uncertainty из NOTEARS
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

# Каждые N интервенций запускаем NOTEARS train_step
TRAIN_EVERY = 8


class System1(nn.Module):
    """
    Амортизированный эпистемический скорер.
    Предсказывает E[IG] для пары (variable, target).
    Теперь принимает на вход alpha_trust из NOTEARS (не только uncertainty).
    Input:  [w_ij, alpha_trust, val_from, val_to, uncertainty, h_W_norm]
    Output: E[IG] ∈ [0, 1]
    """
    def __init__(self, activation: str = "relu"):
        super().__init__()
        act = ACTIVATION_MAP.get(activation, nn.ReLU())
        self.net = nn.Sequential(
            nn.Linear(6, 48), act,
            nn.Linear(48, 24), act,
            nn.Linear(24, 1), nn.Sigmoid(),
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
        self.s1_optim = torch.optim.Adam(self.system1.parameters(), lr=3e-4)

        self._cg_history:  deque[float] = deque(maxlen=20)
        self._phi_history: deque[float] = deque(maxlen=30)
        self._total_interventions = 0
        self._last_do             = "—"
        self._last_result: dict | None = None
        self._peak_discovery_rate: float = 0.0

        # NOTEARS training stats
        self._last_notears_loss: dict | None = None
        self._notears_steps = 0

        self._bootstrap()

    # ── Bootstrap ─────────────────────────────────────────────────────────────
    def _bootstrap(self):
        """Инициализируем узлы и сеем text priors в матрицу W."""
        for var_id, val in self.env.variables.items():
            self.graph.set_node(var_id, val)

        # Записываем начальное наблюдение в буфер
        self.graph.record_observation(dict(self.env.variables))

        # Text priors: первые 2 GT-ребра с шумом + spurious
        gt = self.env.gt_edges()
        for e in gt[:2]:
            noisy_w = e["weight"] * 0.3 + (np.random.rand() - 0.5) * 0.4
            self.graph.set_edge(e["from_"], e["to"], noisy_w, alpha=0.06)

        var_ids = self.env.variable_ids
        if len(var_ids) >= 4:
            # Spurious correlations — выгорят через L_intervention
            self.graph.set_edge(var_ids[1], var_ids[3],  0.35, alpha=0.05)
            self.graph.set_edge(var_ids[2], var_ids[0], -0.20, alpha=0.04)

    # ── h(W) getter ──────────────────────────────────────────────────────────
    def _get_h_W(self) -> float:
        if self.graph._core is None:
            return 0.0
        return float(self.graph._core.dag_constraint().item())

    # ── Epistemic scoring ─────────────────────────────────────────────────────
    def score_interventions(self) -> list[dict]:
        """System 1 предсказывает E[IG] с учётом текущего h(W)."""
        var_ids   = self.env.variable_ids
        h_W_norm  = min(abs(self._get_h_W()) / max(self.graph._d, 1), 1.0)

        features_list: list[torch.Tensor] = []
        candidates:    list[dict]         = []

        for v_from in var_ids:
            for v_to in var_ids:
                if v_from == v_to:
                    continue
                uncertainty = self.graph.edge_uncertainty(v_from, v_to)

                # Достаём вес из матрицы W
                w_ij = 0.0
                if (self.graph._core is not None
                        and v_from in self.graph._node_ids
                        and v_to   in self.graph._node_ids):
                    i = self.graph._node_ids.index(v_from)
                    j = self.graph._node_ids.index(v_to)
                    w_ij = self.graph._core.W_masked()[i, j].item()

                # Alpha-trust напрямую из NOTEARS
                alpha = 1.0 - uncertainty
                val_from = self.graph.nodes.get(v_from, 0.5)
                val_to   = self.graph.nodes.get(v_to,   0.5)

                feat = torch.tensor(
                    [w_ij, alpha, val_from, val_to, uncertainty, h_W_norm],
                    dtype=torch.float32, device=self.device
                )
                features_list.append(feat)
                candidates.append({
                    "variable":    v_from,
                    "target":      v_to,
                    "value":       0.9 if np.random.rand() > 0.5 else 0.1,
                    "uncertainty": uncertainty,
                    "expected_ig": 0.0,
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

        # Снимок ПЕРЕД интервенцией
        obs_before = dict(self.env.observe())

        # Предсказание из NOTEARS-графа
        predicted = self.graph.propagate(var, value)

        # Ground truth (интервенционная жёсткость)
        observed = self.env.intervene(var, value)

        # Записываем в буферы NOTEARS
        self.graph.record_observation(obs_before)
        self.graph.record_observation(observed)
        self.graph.record_intervention(var, value, obs_before, observed)

        # ── NOTEARS train_step каждые TRAIN_EVERY интервенций ──────────────
        notears_result = None
        if self._total_interventions % TRAIN_EVERY == 0:
            notears_result = self.graph.train_step()
            if notears_result is not None:
                self._notears_steps += 1
                self._last_notears_loss = notears_result

        # ── Обновляем узлы ────────────────────────────────────────────────
        for node_id, obs_val in observed.items():
            self.graph.nodes[node_id] = obs_val

        mdl_after         = self.graph.mdl_size
        compression_delta = mdl_before - mdl_after
        self._cg_history.append(compression_delta)

        # ── Обучаем System 1 ──────────────────────────────────────────────
        pred_val  = predicted.get(list(observed.keys())[-1], 0.5)
        obs_val_  = list(observed.values())[-1]
        actual_ig = float(abs(pred_val - obs_val_))
        h_W_norm  = min(abs(self._get_h_W()) / max(self.graph._d, 1), 1.0)

        s1_input = torch.tensor(
            [best.get("expected_ig", 0), 0.5, 0.0, 0.0, best["uncertainty"], h_W_norm],
            dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        s1_target = torch.tensor([[actual_ig]], dtype=torch.float32, device=self.device)

        self.s1_optim.zero_grad()
        s1_pred = self.system1(s1_input)
        s1_loss = nn.functional.mse_loss(s1_pred, s1_target)
        s1_loss.backward()
        self.s1_optim.step()

        self._total_interventions += 1
        self._last_do = f"do({var}={value:.2f})"

        current_dr = self.discovery_rate
        if current_dr > self._peak_discovery_rate:
            self._peak_discovery_rate = current_dr

        updated_edges = [f"{e.from_}→{e.to}" for e in self.graph.edges[:4]]

        self._last_result = {
            "variable":           var,
            "value":              value,
            "compression_delta":  compression_delta,
            "updated_edges":      updated_edges,
            "pruned_edges":       [],
            "prediction_error":   float(np.mean([
                abs(predicted.get(k, 0) - v) for k, v in observed.items()
            ])),
            "notears":            notears_result,
        }
        return self._last_result

    # ── Φ_approx ──────────────────────────────────────────────────────────────
    def phi_approx(self) -> float:
        if len(self._phi_history) < 4:
            return 0.1
        hist    = torch.tensor(list(self._phi_history), device=self.device)
        var_self = hist.var().item()
        return float(np.clip(var_self * 10, 0, 1))

    def record_phi(self, val: float):
        self._phi_history.append(val)

    # ── Demon disruption — теперь портит матрицу W ────────────────────────────
    def demon_disrupt(self) -> str:
        if self.graph._core is None:
            return "no core"
        with torch.no_grad():
            # Добавляем шум к случайному ненулевому ребру
            W   = self.graph._core.W
            sig = (W.abs() > 0.05).nonzero(as_tuple=False)
            if len(sig) == 0:
                return "no significant edges"
            idx  = sig[np.random.randint(len(sig))]
            i, j = idx[0].item(), idx[1].item()
            noise = (np.random.rand() - 0.5) * 0.3
            W[i, j] += noise
            from_name = self.graph._node_ids[i] if i < len(self.graph._node_ids) else f"v{i}"
            to_name   = self.graph._node_ids[j] if j < len(self.graph._node_ids) else f"v{j}"
        self.graph._invalidate_cache()
        return f"corrupted W[{from_name}→{to_name}] +{noise:.3f}"

    # ── Properties ────────────────────────────────────────────────────────────
    @property
    def compression_gain(self) -> float:
        if not self._cg_history:
            return 0.0
        return float(np.mean(list(self._cg_history)))

    @property
    def discovery_rate(self) -> float:
        return self.env.discovery_rate([
            {"from_": e.from_, "to": e.to, "weight": e.weight}
            for e in self.graph.edges
        ])

    @property
    def peak_discovery_rate(self) -> float:
        return self._peak_discovery_rate

    # ── Snapshot ──────────────────────────────────────────────────────────────
    def snapshot(self) -> dict:
        current_dr = self.discovery_rate
        if current_dr > self._peak_discovery_rate:
            self._peak_discovery_rate = current_dr

        # h(W): идеально = 0 (DAG). Нормализуем для UI.
        h_W = self._get_h_W()

        notears_info = None
        if self._last_notears_loss:
            notears_info = {
                "steps":   self._notears_steps,
                "loss":    self._last_notears_loss.get("loss", 0),
                "h_W":     round(h_W, 4),
                "l_int":   self._last_notears_loss.get("l_int", 0),
            }

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
            "discovery_rate":        round(current_dr, 3),
            "peak_discovery_rate":   round(self._peak_discovery_rate, 3),
            "h_W":                   round(h_W, 4),        # DAG constraint
            "notears":               notears_info,
            "edges": [e.as_dict() for e in self.graph.edges],
        }