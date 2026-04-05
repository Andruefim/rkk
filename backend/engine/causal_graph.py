"""
causal_graph.py — NOTEARS / GNN как nn.Module.

Фаза 10+: GNN заменяет NOTEARS как ядро каузального графа.

Фаза B (Predictive World Model):
  train_step: смесь интервенций (_int_buffer) и пассивных пар подряд из _obs_buffer
    (X_t, a=0)→X_{t+1} — физика/кубы без явного do. Доля пассива: RKK_WM_PASSIVE_MIX (по умолч. 0.35).
  Интервенции: X_pred = forward_dynamics(X_t, a_t), L_rec = MSE(X_pred, X_{t+1}).
  propagate / imagination: то же f(state, action); rollout_step_free — f(X, 0).

  USE_GNN = True   → CausalGNNCore  (message passing + action_enc)
  USE_GNN = False  → NOTEARSCore    (forward_dynamics ≈ forward(X+a))

L_total = L_rec + λ_dag*h(W) + λ_l1*|W|₁
h(W)    = tr(exp(W∘W)) - d
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass

# ── Переключение ядра ─────────────────────────────────────────────────────────
USE_GNN = True   # False → NOTEARS (откат к фазам 1-9)


# ─── Edge ─────────────────────────────────────────────────────────────────────
@dataclass
class Edge:
    from_:  str
    to:     str
    weight: float
    alpha_trust: float
    intervention_count: int = 1

    def as_dict(self):
        return {
            "from_": self.from_, "to": self.to,
            "weight": round(self.weight, 4),
            "alpha_trust": round(self.alpha_trust, 4),
            "intervention_count": self.intervention_count,
        }


# ─── NOTEARSCore (фазы 1-9, fallback) ────────────────────────────────────────
class NOTEARSCore(nn.Module):
    def __init__(self, d: int, device: torch.device):
        super().__init__()
        self.d      = d
        self.device = device
        self.W      = nn.Parameter(torch.zeros(d, d, device=device))
        mask = 1.0 - torch.eye(d, device=device)
        self.register_buffer("mask", mask)

    def W_masked(self) -> torch.Tensor:
        return self.W * self.mask

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X @ self.W_masked()

    def dag_constraint(self) -> torch.Tensor:
        W2  = self.W_masked() ** 2
        exp = torch.linalg.matrix_exp(W2)
        return exp.trace() - self.d

    def intervention_loss(self, X_obs, X_int, int_var_idx, int_val):
        a = torch.zeros_like(X_obs)
        a[:, int_var_idx] = int_val
        return F.mse_loss(self.forward_dynamics(X_obs, a), X_int)

    def forward_dynamics(self, X: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        X' ≈ forward(X_do), где в координатах с ненулевым a подставляется абсолютное
        целевое значение do (как в старом X_do[:, idx] = val); иначе — пассивный шаг.
        """
        m = (torch.abs(a) > 1e-8).float()
        x_in = X * (1.0 - m) + a * m
        return self.forward(x_in)

    def l1_reg(self) -> torch.Tensor:
        return self.W_masked().abs().sum()

    def alpha_trust_matrix(self) -> torch.Tensor:
        W_abs = self.W_masked().abs()
        return W_abs / W_abs.max() if W_abs.max() > 0 else W_abs


# ─── CausalGraph ──────────────────────────────────────────────────────────────
class CausalGraph:
    LAMBDA_DAG  = 0.5
    LAMBDA_INT  = 2.0
    LAMBDA_L1   = 0.02
    EDGE_THRESH = 0.05

    def __init__(self, device: torch.device):
        self.device     = device
        self.nodes:     dict[str, float] = {}
        self._node_ids: list[str]        = []
        self._d         = 0
        self._core      = None
        self._optim     = None
        self._obs_buffer: list[list[float]] = []
        self._int_buffer: list[dict]        = []
        self.BUFFER_SIZE = 64
        self._edge_cache: list[Edge] | None = None
        self._mdl_cache:  float | None      = None
        self.train_losses: list[float]      = []

    def set_node(self, id_: str, value: float = 0.0) -> None:
        self._invalidate_cache()
        if id_ not in self.nodes:
            self._node_ids.append(id_)
            self._d += 1
        self.nodes[id_] = value
        if self._core is None or self._core.d != self._d:
            self._rebuild_core()

    def rebind_variables(self, ordered_ids: list[str], values: dict[str, float]) -> None:
        """
        Полностью заменить набор узлов (например humanoid → только slot_* в visual mode).
        Сбрасывает NOTEARS/GNN буферы, чтобы размеры строк совпадали с новым d.
        """
        self._invalidate_cache()
        self.nodes = {k: float(values.get(k, 0.5)) for k in ordered_ids}
        self._node_ids = list(ordered_ids)
        self._d = len(ordered_ids)
        self._obs_buffer.clear()
        self._int_buffer.clear()
        self._core = None
        self._optim = None
        self._rebuild_core()

    def _rebuild_core(self):
        if USE_GNN:
            from engine.causal_gnn import CausalGNNCore
            old_W = None
            if self._core is not None:
                # GNN resize: мигрируем через resize_to если можем
                if hasattr(self._core, 'resize_to'):
                    new_core  = self._core.resize_to(self._d)
                    self._core = new_core
                    self._optim = torch.optim.Adam(self._core.parameters(), lr=5e-3)
                    self._invalidate_cache()
                    return
                old_W = self._core.W_masked().detach().clone()
            self._core = CausalGNNCore(self._d, self.device)
        else:
            old_W = None
            if self._core is not None and self._core.d < self._d:
                old_W = self._core.W_masked().detach().clone()
            self._core = NOTEARSCore(self._d, self.device)

        if old_W is not None:
            old_d = old_W.shape[0]
            with torch.no_grad():
                self._core.W[:old_d, :old_d] = old_W

        self._optim = torch.optim.Adam(self._core.parameters(), lr=5e-3)
        self._invalidate_cache()

    def record_observation(self, obs: dict[str, float]) -> None:
        if not self._node_ids:
            return
        vec = [obs.get(nid, 0.0) for nid in self._node_ids]
        self._obs_buffer.append(vec)
        if len(self._obs_buffer) > self.BUFFER_SIZE * 4:
            self._obs_buffer = self._obs_buffer[-self.BUFFER_SIZE * 2:]

    def record_intervention(self, var_name: str, val: float,
                            obs_before: dict, obs_after: dict) -> None:
        if var_name not in self._node_ids:
            return
        self._int_buffer.append({
            "idx":        self._node_ids.index(var_name),
            "val":        val,
            "obs_before": [obs_before.get(n, 0.0) for n in self._node_ids],
            "obs_after":  [obs_after.get(n, 0.0)  for n in self._node_ids],
        })
        if len(self._int_buffer) > self.BUFFER_SIZE:
            self._int_buffer = self._int_buffer[-self.BUFFER_SIZE:]

    def train_step(self) -> dict[str, float] | None:
        """
        World model: батч = интервенции + пассивные переходы из подряд идущих наблюдений.
        Пассив: a=0, L_rec штрафует f(X_t,0) vs X_{t+1} (кубы/физика между тиками).
        """
        if self._core is None:
            return None

        int_b = self._int_buffer
        obs_b = self._obs_buffer
        batch_cap = 32
        try:
            passive_ratio = float(os.environ.get("RKK_WM_PASSIVE_MIX", "0.35"))
        except ValueError:
            passive_ratio = 0.35
        passive_ratio = max(0.0, min(0.75, passive_ratio))

        L = len(obs_b)
        max_pairs = max(0, L - 1)
        n_p = min(int(round(batch_cap * passive_ratio)), max_pairs)
        n_i = min(batch_cap - n_p, len(int_b))

        if n_p + n_i < 4:
            n_i = min(batch_cap, len(int_b))
            n_p = min(max_pairs, batch_cap - n_i)
        if n_p + n_i < 4:
            return None

        rows_X: list[list[float]] = []
        rows_Y: list[list[float]] = []
        rows_a: list[list[float]] = []

        for item in int_b[-n_i:] if n_i > 0 else []:
            rows_X.append(item["obs_before"])
            rows_Y.append(item["obs_after"])
            arow = [0.0] * self._d
            arow[item["idx"]] = item["val"]
            rows_a.append(arow)

        if n_p > 0:
            start_t = L - n_p - 1
            for k in range(n_p):
                rows_X.append(obs_b[start_t + k])
                rows_Y.append(obs_b[start_t + k + 1])
                rows_a.append([0.0] * self._d)

        X_t = torch.tensor(rows_X, dtype=torch.float32, device=self.device)
        X_tp1 = torch.tensor(rows_Y, dtype=torch.float32, device=self.device)
        a_t = torch.tensor(rows_a, dtype=torch.float32, device=self.device)

        self._optim.zero_grad()

        fd = getattr(self._core, "forward_dynamics", None)
        if callable(fd):
            X_pred = fd(X_t, a_t)
        else:
            X_pred = self._core(X_t + a_t)
        l_rec = F.mse_loss(X_pred, X_tp1)

        h_W   = self._core.dag_constraint()
        l_dag = self.LAMBDA_DAG * h_W.abs()
        l_l1  = self.LAMBDA_L1  * self._core.l1_reg()

        l_int = torch.tensor(0.0, device=self.device)

        loss = l_rec + l_dag + l_l1 + l_int
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._core.parameters(), max_norm=1.0)
        self._optim.step()

        self._invalidate_cache()
        self.train_losses.append(loss.item())
        if len(self.train_losses) > 100:
            self.train_losses.pop(0)

        return {
            "loss":  round(loss.item(), 5),
            "l_rec": round(l_rec.item(), 5),
            "l_dag": round(l_dag.item(), 5),
            "l_int": round(l_int.item(), 5),
            "l_l1":  round(l_l1.item(), 5),
            "h_W":   round(h_W.item(), 5),
            "batch_int": n_i,
            "batch_passive": n_p,
        }

    def set_edge(self, from_: str, to: str, weight: float, alpha: float) -> None:
        if self._core is None or from_ not in self._node_ids or to not in self._node_ids:
            return
        i, j = self._node_ids.index(from_), self._node_ids.index(to)
        with torch.no_grad():
            old_w = self._core.W[i, j].item()
            self._core.W[i, j] = 0.7 * old_w + 0.3 * weight
        self._invalidate_cache()

    def remove_edge(self, from_: str, to: str) -> None:
        if self._core is None or from_ not in self._node_ids or to not in self._node_ids:
            return
        i, j = self._node_ids.index(from_), self._node_ids.index(to)
        with torch.no_grad():
            self._core.W[i, j] = 0.0
        self._invalidate_cache()

    @property
    def edges(self) -> list[Edge]:
        if self._edge_cache is not None:
            return self._edge_cache
        if self._core is None:
            return []
        W     = self._core.W_masked().detach()
        alpha = self._core.alpha_trust_matrix().detach()
        result = []
        for i, from_ in enumerate(self._node_ids):
            for j, to in enumerate(self._node_ids):
                w = W[i, j].item()
                if abs(w) >= self.EDGE_THRESH:
                    result.append(Edge(
                        from_=from_, to=to,
                        weight=round(w, 4),
                        alpha_trust=round(alpha[i, j].item(), 4),
                        intervention_count=1,
                    ))
        self._edge_cache = result
        return result

    def edge_uncertainty(self, from_: str, to: str) -> float:
        if self._core is None or from_ not in self._node_ids or to not in self._node_ids:
            return 1.0
        i, j  = self._node_ids.index(from_), self._node_ids.index(to)
        alpha = self._core.alpha_trust_matrix()[i, j].item()
        return 1.0 - alpha

    @property
    def alpha_mean(self) -> float:
        if self._core is None:
            return 0.05
        alpha = self._core.alpha_trust_matrix()
        mask  = alpha > 0.01
        return float(alpha[mask].mean().item()) if mask.sum() > 0 else 0.05

    @property
    def mdl_size(self) -> float:
        if self._mdl_cache is not None:
            return self._mdl_cache
        if self._core is None:
            return 0.0
        W        = self._core.W_masked().detach()
        alpha    = self._core.alpha_trust_matrix().detach()
        sig_mask = W.abs() >= self.EDGE_THRESH
        if sig_mask.sum() == 0:
            return 0.0
        mdl = (1 + (1 - alpha[sig_mask])).sum().item()
        self._mdl_cache = mdl
        return mdl

    def propagate(self, variable: str, value: float) -> dict[str, float]:
        return self.propagate_from(self.nodes, variable, value)

    def propagate_from(
        self, base: dict[str, float], variable: str, value: float
    ) -> dict[str, float]:
        """
        Предсказание после do(variable=value) от произвольного снимка узлов
        (Фаза 13: imagination rollout без мутации self.nodes).
        """
        if self._core is None:
            return dict(base)
        state_vec = torch.tensor(
            [[float(base.get(n, 0.0)) for n in self._node_ids]],
            dtype=torch.float32, device=self.device,
        )
        a_vec = torch.zeros(1, self._d, dtype=torch.float32, device=self.device)
        if variable in self._node_ids:
            a_vec[0, self._node_ids.index(variable)] = float(value)
        with torch.no_grad():
            fd = getattr(self._core, "forward_dynamics", None)
            pred = fd(state_vec, a_vec) if callable(fd) else self._core(state_vec + a_vec)
        result = {nid: float(pred[0, i].item()) for i, nid in enumerate(self._node_ids)}
        return result

    def rollout_step_free(self, base: dict[str, float]) -> dict[str, float]:
        """
        Один шаг «свободной» динамики GNN: X' = core(X), без интервенции.
        Используется для многошагового imagination после мысленного do().
        """
        if self._core is None:
            return dict(base)
        state_vec = torch.tensor(
            [[float(base.get(n, 0.0)) for n in self._node_ids]],
            dtype=torch.float32, device=self.device,
        )
        z = torch.zeros_like(state_vec)
        with torch.no_grad():
            fd = getattr(self._core, "forward_dynamics", None)
            pred = fd(state_vec, z) if callable(fd) else self._core(state_vec)
        return {nid: float(pred[0, i].item()) for i, nid in enumerate(self._node_ids)}

    def _invalidate_cache(self):
        self._edge_cache = None
        self._mdl_cache  = None

    def to_dict(self) -> dict:
        W_data = h_data = core_type = None
        if self._core is not None:
            W_data    = self._core.W_masked().detach().cpu().tolist()
            h_data    = round(self._core.dag_constraint().item(), 5)
            core_type = "gnn" if USE_GNN else "notears"
        return {
            "nodes":     self.nodes,
            "edges":     [e.as_dict() for e in self.edges],
            "mdl":       self.mdl_size,
            "W":         W_data,
            "h_W":       h_data,
            "d":         self._d,
            "core_type": core_type,
        }

    def clone(self) -> "CausalGraph":
        g = CausalGraph(self.device)
        g.nodes      = dict(self.nodes)
        g._node_ids  = list(self._node_ids)
        g._d         = self._d
        if self._core is not None:
            g._rebuild_core()
            with torch.no_grad():
                g._core.W.copy_(self._core.W)
        return g