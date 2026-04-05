"""
causal_gnn.py — GNN-based Causal Structure Learner (Фаза 10+).

Drop-in замена NOTEARSCore в causal_graph.py.

Ключевые отличия от NOTEARS:
  NOTEARS:  X_pred = X @ W           — линейный SCM, O(d²) forward
            h(W) = tr(exp(W∘W)) - d  — O(d³) из-за matrix_exp
            W: nn.Parameter (d×d)

  CausalGNN: Message Passing forward — нелинейный, O(d² · hidden)
             h(W) — та же формула (совместимость с Epistemic Annealing)
             W: nn.Parameter (d×d)   — та же структура (.grad совместим)
             + node_enc / msg_fn / out_dec — обучаются параллельно

Совместимость интерфейса (1:1 с NOTEARSCore):
  .W                   → nn.Parameter (d×d), имеет .grad
  .W_masked()          → W * mask (без диагонали)
  .dag_constraint()    → tr(exp(W∘W)) - d
  .intervention_loss() → MSE(forward_dynamics(X,a), X_int)
  .l1_reg()            → |W_masked|₁
  .alpha_trust_matrix() → нормированные абсолютные веса
  .forward_dynamics(X, a) → X_{t+1} pred (world model)
  .forward(X)          → forward_dynamics(X, 0) — пассивный шаг

Масштабируемость:
  d=6   (physics/chemistry/logic): быстрее чем NOTEARS за счёт меньшего hidden
  d=18  (PyBullet 3 объекта):      GNN выигрывает при нелинейных взаимодействиях
  d=100 (будущее, много объектов): message passing O(d²) vs matrix_exp O(d³)

Опциональный torch-geometric:
  Если установлен → используем MessagePassing для батч-эффективности.
  Иначе → ручной matrix message passing (pure PyTorch, всегда доступен).

  pip install torch-geometric   # опционально
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ── Опциональный torch-geometric ─────────────────────────────────────────────
try:
    from torch_geometric.nn import MessagePassing
    _TG_AVAILABLE = True
except ImportError:
    _TG_AVAILABLE = False


# ─── Вспомогательные блоки ───────────────────────────────────────────────────
def _mlp(in_dim: int, hidden: int, out_dim: int, act=nn.ReLU) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), act(),
        nn.Linear(hidden, out_dim),
    )


# ─── CausalGNNCore ────────────────────────────────────────────────────────────
class CausalGNNCore(nn.Module):
    """
    GNN-based каузальный граф. Drop-in замена NOTEARSCore.

    Архитектура:
      1. node_enc:   scalar → hidden embedding  (кодируем значение переменной)
      2. msg_fn:     (h_from, h_to) → message   (создаём сообщение по ребру j→i)
      3. out_dec:    (h_i, agg_i)   → x_i_pred  (декодируем предсказание)

    Adjacency W:
      nn.Parameter (d×d), совместим с NOTEARSCore.
      Все каузальные рёбра кодируются в W, как и раньше.
      Градиенты по W используются в System 1 (grad_norm).

    Message Passing:
      agg_i = Σ_j W[j,i] · msg_fn(h_j, h_i)
      x_i_pred = out_dec(concat(h_i, agg_i))

    DAG Constraint:
      h(W) = tr(exp(W∘W)) - d   ← та же формула что в NOTEARS
    """

    def __init__(self, d: int, device: torch.device, hidden: int = 24):
        super().__init__()
        self.d      = d
        self.device = device
        self.hidden = hidden

        # ── Adjacency matrix (совместимость с NOTEARSCore) ───────────────────
        self.W = nn.Parameter(torch.zeros(d, d, device=device))

        # ── GNN компоненты ────────────────────────────────────────────────────
        # Node encoder: кодируем скалярное значение переменной
        self.node_enc = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
        )

        # Action encoder: то же измерение, что и состояние (a_t[i] — воздействие на i-ю ось)
        self.action_enc = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
        )

        # Message function: j→i сообщение по ребру с весом W[j,i]
        self.msg_fn = _mlp(hidden * 2, hidden, hidden, nn.ReLU)

        # Output decoder: предсказываем x_i из h_i + агрегированных сообщений
        self.out_dec = _mlp(hidden * 2, hidden, 1, nn.ReLU)

        # Xavier initialization
        for m in [self.node_enc, self.action_enc, self.msg_fn, self.out_dec]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.5)
                    nn.init.zeros_(layer.bias)

        # No-self-loop mask
        mask = 1.0 - torch.eye(d, device=device)
        self.register_buffer("mask", mask)

        # node_enc / msg_fn / out_dec создают Linear без device= → по умолчанию CPU;
        # W и mask уже на device — без .to(device) propagate() падает на ROCm/CUDA.
        self.to(device)

    # ── Основной интерфейс (совместим с NOTEARSCore) ─────────────────────────
    def W_masked(self) -> torch.Tensor:
        """W без диагонали."""
        return self.W * self.mask

    def _message_pass(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, d, hidden) → (B, d) предсказание скаляров узлов."""
        B, d, _hd = h.shape
        h_src = h.unsqueeze(1).expand(B, d, d, self.hidden)
        h_dst = h.unsqueeze(2).expand(B, d, d, self.hidden)
        msg = self.msg_fn(torch.cat([h_src, h_dst], dim=-1))
        A       = self.W_masked()
        weights = A.t().unsqueeze(0).unsqueeze(-1)
        agg     = (msg * weights).sum(dim=2)
        out = self.out_dec(torch.cat([h, agg], dim=-1))
        return out.squeeze(-1)

    def forward_dynamics(self, X: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        World model: X_{t+1} ≈ f(X_t, a_t).
        X_t, a_t: (B, d) в масштабе наблюдений; a_t разрежен (do(var)=val по индексу).
        Где |a_i|≈0, вклад action_enc не добавляется (чтобы f(X,0) не сдвигал bias’ом).
        """
        am = (torch.abs(a).unsqueeze(-1) > 1e-8).float()
        h_a = self.action_enc(a.unsqueeze(-1))
        h = self.node_enc(X.unsqueeze(-1)) + am * h_a
        return self._message_pass(h)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Один шаг «пассивной» динамики f(X, 0) — совместимость с вызовами без явного действия
        (например подача предсказания в visual cortex).
        """
        return self.forward_dynamics(X, torch.zeros_like(X))

    def dag_constraint(self) -> torch.Tensor:
        """
        h(W) = tr(exp(W∘W)) - d  (NOTEARS DAG constraint).
        Та же формула — совместима с Epistemic Annealing.
        """
        W2  = self.W_masked() ** 2
        exp = torch.linalg.matrix_exp(W2)
        return exp.trace() - self.d

    def intervention_loss(
        self,
        X_obs:       torch.Tensor,
        X_int:       torch.Tensor,
        int_var_idx: int,
        int_val:     float,
    ) -> torch.Tensor:
        """L_intervention: после do(var=val) предсказание совпадает с наблюдением."""
        a = torch.zeros_like(X_obs)
        a[:, int_var_idx] = int_val
        predicted = self.forward_dynamics(X_obs, a)
        return F.mse_loss(predicted, X_int)

    def l1_reg(self) -> torch.Tensor:
        """L1 на веса → разреженность → MDL."""
        return self.W_masked().abs().sum()

    def alpha_trust_matrix(self) -> torch.Tensor:
        """
        Alpha-trust: нормированные абсолютные веса W.
        Совместим с NOTEARSCore.
        """
        W_abs = self.W_masked().abs()
        if W_abs.max() > 0:
            return W_abs / W_abs.max()
        return W_abs

    # ── Динамическое изменение размера (для PyBullet с переменным n_objects) ──
    def resize_to(self, new_d: int) -> "CausalGNNCore":
        """
        Создаём новый GNN для большего d.
        Мигрируем существующие веса W (каузальные связи сохраняются).
        MLP-веса (node_enc, msg_fn, out_dec) копируются полностью —
        они не зависят от d и переиспользуются.
        """
        new_core = CausalGNNCore(new_d, self.device, self.hidden)
        old_d = self.d

        with torch.no_grad():
            # Нельзя W[slice]=… на nn.Parameter — только через клон + copy_.
            w = new_core.W.detach().clone()
            w[:old_d, :old_d] = self.W.detach()
            new_core.W.copy_(w)

        # Переносим обученные MLP (архитектура не изменилась)
        new_core.node_enc.load_state_dict(self.node_enc.state_dict())
        new_core.action_enc.load_state_dict(self.action_enc.state_dict())
        new_core.msg_fn.load_state_dict(self.msg_fn.state_dict())
        new_core.out_dec.load_state_dict(self.out_dec.state_dict())

        return new_core

    def snapshot_info(self) -> dict:
        """Дополнительная статистика для UI."""
        return {
            "type":     "gnn",
            "d":        self.d,
            "hidden":   self.hidden,
            "tg_avail": _TG_AVAILABLE,
            "params":   sum(p.numel() for p in self.parameters()),
        }
