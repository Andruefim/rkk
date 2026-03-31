"""
causal_graph.py — NOTEARS как nn.Module.

Ядро графа теперь — обучаемая матрица смежности W (d×d).

L_total = L_reconstruction + λ_dag * h(W) + λ_int * L_intervention + λ_l1 * |W|

h(W) = trace(exp(W ∘ W)) - d  ← гладкое ограничение ацикличности (NOTEARS)

Alpha-trust: производное от градиентной уверенности по конкретному ребру.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass


# ─── Edge (для совместимости с остальным кодом) ───────────────────────────────
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


# ─── NOTEARS Core ─────────────────────────────────────────────────────────────
class NOTEARSCore(nn.Module):
    """
    Дифференцируемый каузальный граф.
    W[i,j] = каузальный вес от переменной i к переменной j.
    """

    def __init__(self, d: int, device: torch.device):
        super().__init__()
        self.d      = d
        self.device = device

        # Обучаемая матрица смежности
        self.W = nn.Parameter(torch.zeros(d, d, device=device))

        # Маска: запрет self-loops
        mask = 1.0 - torch.eye(d, device=device)
        self.register_buffer("mask", mask)

    def W_masked(self) -> torch.Tensor:
        """W без диагонали."""
        return self.W * self.mask

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Линейный SCM: X_pred = X @ W
        X: (B, d) — батч наблюдений
        """
        return X @ self.W_masked()

    def dag_constraint(self) -> torch.Tensor:
        """
        h(W) = trace(exp(W ∘ W)) - d
        h(W) = 0  ↔  W — матрица смежности DAG
        """
        W2  = self.W_masked() ** 2
        exp = torch.linalg.matrix_exp(W2)
        return exp.trace() - self.d

    def intervention_loss(
        self,
        X_obs: torch.Tensor,
        X_int: torch.Tensor,
        int_var_idx: int,
        int_val: float,
    ) -> torch.Tensor:
        """
        L_intervention: после do(var=val) предсказание должно совпасть с наблюдением.
        Это ключевое отличие от чистого NOTEARS — мы добавляем интервенционную жёсткость.
        """
        X_do = X_obs.clone()
        X_do[:, int_var_idx] = int_val
        predicted = self.forward(X_do)
        return F.mse_loss(predicted, X_int)

    def l1_reg(self) -> torch.Tensor:
        """L1 на веса → разреженность → MDL."""
        return self.W_masked().abs().sum()

    def alpha_trust_matrix(self) -> torch.Tensor:
        """
        Alpha-trust как уверенность градиента по каждому ребру.
        |W[i,j]| нормализованное → [0, 1]
        """
        W_abs = self.W_masked().abs()
        if W_abs.max() > 0:
            return W_abs / W_abs.max()
        return W_abs


# ─── CausalGraph (публичный интерфейс) ───────────────────────────────────────
class CausalGraph:
    """
    Публичный интерфейс к NOTEARS-ядру.
    Сохраняет совместимость со старым API (nodes, edges, set_edge, etc.)
    и добавляет train_step() для градиентного обучения.
    """

    # Гиперпараметры NOTEARS
    LAMBDA_DAG = 0.5    # штраф за нарушение ацикличности
    LAMBDA_INT = 2.0    # штраф за ошибку интервенции
    LAMBDA_L1  = 0.02   # L1 разреженность (MDL)
    EDGE_THRESH = 0.05  # порог для считывания рёбер из W

    def __init__(self, device: torch.device):
        self.device    = device
        self.nodes:    dict[str, float] = {}
        self._node_ids: list[str]       = []   # упорядоченный список узлов
        self._d        = 0

        # NOTEARS-ядро (создаётся после первого set_node)
        self._core: NOTEARSCore | None  = None
        self._optim: torch.optim.Optimizer | None = None

        # Буфер наблюдений для батч-обучения
        self._obs_buffer:  list[list[float]] = []
        self._int_buffer:  list[dict]        = []  # {idx, val, obs_before, obs_after}
        self.BUFFER_SIZE   = 64

        # Кэш для совместимости со старым API
        self._edge_cache:  list[Edge] | None = None
        self._mdl_cache:   float | None      = None

        # История обучения
        self.train_losses: list[float] = []

    # ── Инициализация узлов ───────────────────────────────────────────────────
    def set_node(self, id_: str, value: float = 0.0) -> None:
        self._invalidate_cache()
        if id_ not in self.nodes:
            self._node_ids.append(id_)
            self._d += 1
        self.nodes[id_] = value

        # Пересоздаём NOTEARS-ядро при изменении размерности
        if self._core is None or self._core.d != self._d:
            self._rebuild_core()

    def _rebuild_core(self):
        """Пересоздаём матрицу W при добавлении новых узлов."""
        old_W = None
        if self._core is not None and self._core.d < self._d:
            old_W = self._core.W_masked().detach().clone()

        self._core  = NOTEARSCore(self._d, self.device)
        self._optim = torch.optim.Adam(self._core.parameters(), lr=5e-3)

        # Переносим старые веса если возможно
        if old_W is not None:
            old_d = old_W.shape[0]
            with torch.no_grad():
                self._core.W[:old_d, :old_d] = old_W
        self._invalidate_cache()

    # ── Буферизация наблюдений ────────────────────────────────────────────────
    def record_observation(self, obs: dict[str, float]) -> None:
        """Добавляем наблюдение в буфер для батч-обучения."""
        if len(self._node_ids) == 0:
            return
        vec = [obs.get(nid, 0.0) for nid in self._node_ids]
        self._obs_buffer.append(vec)
        if len(self._obs_buffer) > self.BUFFER_SIZE * 4:
            self._obs_buffer = self._obs_buffer[-self.BUFFER_SIZE * 2:]

    def record_intervention(
        self,
        var_name: str,
        val: float,
        obs_before: dict[str, float],
        obs_after: dict[str, float],
    ) -> None:
        """Запоминаем результат do()-операции для L_intervention."""
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

    # ── Обучение NOTEARS ──────────────────────────────────────────────────────
    def train_step(self) -> dict[str, float] | None:
        """
        Один шаг градиентного спуска.
        Вызывается из RKKAgent.step() после накопления буфера.
        """
        if self._core is None:
            return None
        if len(self._obs_buffer) < 8:
            return None

        # Батч наблюдений
        obs_t = torch.tensor(
            self._obs_buffer[-self.BUFFER_SIZE:],
            dtype=torch.float32, device=self.device
        )

        self._optim.zero_grad()

        # L_reconstruction: W предсказывает X из X
        X_pred = self._core(obs_t)
        l_rec  = F.mse_loss(X_pred, obs_t)

        # h(W): DAG constraint
        h_W   = self._core.dag_constraint()
        l_dag = self.LAMBDA_DAG * h_W.abs()

        # L1 разреженность
        l_l1  = self.LAMBDA_L1 * self._core.l1_reg()

        # L_intervention: если есть интервенционные данные
        l_int = torch.tensor(0.0, device=self.device)
        if len(self._int_buffer) >= 4:
            int_batch = self._int_buffer[-min(16, len(self._int_buffer)):]
            for item in int_batch:
                X_obs = torch.tensor(
                    [item["obs_before"]], dtype=torch.float32, device=self.device
                )
                X_int = torch.tensor(
                    [item["obs_after"]], dtype=torch.float32, device=self.device
                )
                l_int = l_int + self._core.intervention_loss(
                    X_obs, X_int, item["idx"], item["val"]
                )
            l_int = self.LAMBDA_INT * l_int / len(int_batch)

        loss = l_rec + l_dag + l_l1 + l_int
        loss.backward()

        # Gradient clipping для стабильности
        torch.nn.utils.clip_grad_norm_(self._core.parameters(), max_norm=1.0)
        self._optim.step()

        self._invalidate_cache()
        self.train_losses.append(loss.item())
        if len(self.train_losses) > 100:
            self.train_losses.pop(0)

        return {
            "loss":        round(loss.item(), 5),
            "l_rec":       round(l_rec.item(), 5),
            "l_dag":       round(l_dag.item(), 5),
            "l_int":       round(l_int.item(), 5),
            "l_l1":        round(l_l1.item(), 5),
            "h_W":         round(h_W.item(), 5),
        }

    # ── Совместимость со старым API ───────────────────────────────────────────

    def set_edge(self, from_: str, to: str, weight: float, alpha: float) -> None:
        """
        Ручное задание ребра — теперь инициализирует W напрямую.
        Используется в bootstrap и Epistemic Annealing.
        """
        if self._core is None or from_ not in self._node_ids or to not in self._node_ids:
            return
        i = self._node_ids.index(from_)
        j = self._node_ids.index(to)
        with torch.no_grad():
            # Мягкое обновление: не перезаписываем, а смешиваем
            old_w = self._core.W[i, j].item()
            new_w = 0.7 * old_w + 0.3 * weight  # EMA
            self._core.W[i, j] = new_w
        self._invalidate_cache()

    def remove_edge(self, from_: str, to: str) -> None:
        """Удаляем ребро — зануляем вес в W."""
        if self._core is None or from_ not in self._node_ids or to not in self._node_ids:
            return
        i = self._node_ids.index(from_)
        j = self._node_ids.index(to)
        with torch.no_grad():
            self._core.W[i, j] = 0.0
        self._invalidate_cache()

    @property
    def edges(self) -> list[Edge]:
        """Читаем рёбра из матрицы W по порогу EDGE_THRESH."""
        if self._edge_cache is not None:
            return self._edge_cache
        if self._core is None:
            return []

        W     = self._core.W_masked().detach()
        alpha = self._core.alpha_trust_matrix().detach()
        result: list[Edge] = []

        for i, from_ in enumerate(self._node_ids):
            for j, to in enumerate(self._node_ids):
                w = W[i, j].item()
                if abs(w) >= self.EDGE_THRESH:
                    result.append(Edge(
                        from_=from_,
                        to=to,
                        weight=round(w, 4),
                        alpha_trust=round(alpha[i, j].item(), 4),
                        intervention_count=1,
                    ))

        self._edge_cache = result
        return result

    def edge_uncertainty(self, from_: str, to: str) -> float:
        """Неопределённость ребра: 1 - alpha_trust."""
        if self._core is None or from_ not in self._node_ids or to not in self._node_ids:
            return 1.0
        i     = self._node_ids.index(from_)
        j     = self._node_ids.index(to)
        alpha = self._core.alpha_trust_matrix()[i, j].item()
        return 1.0 - alpha

    @property
    def alpha_mean(self) -> float:
        if self._core is None:
            return 0.05
        alpha = self._core.alpha_trust_matrix()
        mask  = alpha > 0.01
        if mask.sum() == 0:
            return 0.05
        return float(alpha[mask].mean().item())

    @property
    def mdl_size(self) -> float:
        """
        MDL через L1-норму W: разреженный граф = меньше бит = лучше понимание.
        """
        if self._mdl_cache is not None:
            return self._mdl_cache
        if self._core is None:
            return 0.0
        W     = self._core.W_masked().detach()
        alpha = self._core.alpha_trust_matrix().detach()
        # MDL = сумма (1 + неопределённость) по значимым рёбрам
        sig_mask   = W.abs() >= self.EDGE_THRESH
        if sig_mask.sum() == 0:
            return 0.0
        uncertainty = (1 - alpha[sig_mask])
        mdl = (1 + uncertainty).sum().item()
        self._mdl_cache = mdl
        return mdl

    def propagate(self, variable: str, value: float) -> dict[str, float]:
        """Forward pass через граф для предсказания до do()."""
        if self._core is None:
            return dict(self.nodes)

        state_vec = torch.tensor(
            [[self.nodes.get(n, 0.0) for n in self._node_ids]],
            dtype=torch.float32, device=self.device
        )
        if variable in self._node_ids:
            idx = self._node_ids.index(variable)
            state_vec[0, idx] = value

        with torch.no_grad():
            pred = self._core(state_vec)

        result = dict(self.nodes)
        for i, nid in enumerate(self._node_ids):
            result[nid] = pred[0, i].item()
        return result

    def _invalidate_cache(self):
        self._edge_cache = None
        self._mdl_cache  = None

    def to_dict(self) -> dict:
        W_data = None
        h_data = None
        if self._core is not None:
            W_data = self._core.W_masked().detach().cpu().tolist()
            h_data = round(self._core.dag_constraint().item(), 5)
        return {
            "nodes":   self.nodes,
            "edges":   [e.as_dict() for e in self.edges],
            "mdl":     self.mdl_size,
            "W":       W_data,
            "h_W":     h_data,
            "d":       self._d,
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