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

LeWM integration (Фаза N):
  .encode_latent(X, a)        → (B, d, hidden) latent node embeddings
  .forward_dynamics_seq(X, A) → parallel (B, T, d) teacher-forcing prediction
  SIGReg                      → anti-collapse regularizer from LeWorldModel
  JEPA: h=node_enc(X)+action_enc(a) in MP; latent_predictor(h,agg) only (no second action_enc).
        cosine loss in causal_graph._jepa_latent_loss; resize_to preserves target_enc EMA weights.

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

import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

_DAG_TAYLOR_WARNED: bool = False


def _warn_dag_taylor_order4_if_large(d: int) -> None:
    """Taylor expansion of tr(exp(M)) omits cycle contributions beyond order 4."""
    global _DAG_TAYLOR_WARNED
    if d <= 50 or _DAG_TAYLOR_WARNED:
        return
    warnings.warn(
        f"CausalGNNCore(d={d}): DAG Taylor penalty uses expansion to order 4; "
        "for dense/large W, longer cycles may be underestimated.",
        UserWarning,
        stacklevel=3,
    )
    _DAG_TAYLOR_WARNED = True


# ── Опциональный torch-geometric ─────────────────────────────────────────────
try:
    from torch_geometric.nn import MessagePassing
    _TG_AVAILABLE = True
except ImportError:
    _TG_AVAILABLE = False


# ─── SIGReg: Sketched-Isotropic-Gaussian Regularizer (from LeWorldModel) ──────
class SIGReg(nn.Module):
    """
    Anti-collapse regularizer from LeWorldModel paper (Maes et al., 2026).

    Enforces isotropic Gaussian distribution on latent embeddings via:
    1. Project embeddings onto M random unit-norm directions (Cramér-Wold theorem)
    2. Compute univariate Epps-Pulley test statistic for Gaussianity
    3. Minimize divergence from N(0,1) along each projection

    By the Cramér-Wold theorem, matching all 1D marginals ≡ matching the full
    joint distribution. This makes the regularizer scale gracefully with
    embedding dimension.

    Usage:
        sigreg = SIGReg(knots=17, num_proj=1024)
        loss = sigreg(embeddings)  # embeddings: (B, D) or (T, B, D)

    Args:
        knots: number of quadrature nodes for Epps-Pulley integral (default 17)
        num_proj: number of random projection directions M (default 1024)
    """

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj
        # Quadrature grid for Epps-Pulley test statistic
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)
        # Trapezoidal weights
        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        weights[0] = dt
        weights[-1] = dt
        # Gaussian window (target characteristic function of N(0,1))
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute SIGReg loss.

        Args:
            z: latent embeddings, shape (B, D) or (T, B, D).
               For 2D input, we treat it as a single batch of B samples.
               For 3D input, T steps are computed independently and averaged.

        Returns:
            Scalar loss (lower = closer to isotropic Gaussian).
        """
        if z.dim() == 2:
            z = z.unsqueeze(0)  # (1, B, D) — single "timestep"

        # z: (T, B, D) — T timesteps, B batch samples, D embedding dim
        D = z.size(-1)
        # Random projections: unit-norm directions on S^{D-1}
        A = torch.randn(D, self.num_proj, device=z.device, dtype=z.dtype)
        A = A / A.norm(p=2, dim=0, keepdim=True)

        # Project: (T, B, D) @ (D, M) → (T, B, M)
        proj = z @ A

        # Epps-Pulley: compare empirical char func vs Gaussian char func
        # x_t: (T, B, M, knots)
        x_t = proj.unsqueeze(-1) * self.t

        # Empirical char func: E[cos(t·h)], E[sin(t·h)] over batch dim (dim=-3 = B)
        cos_mean = x_t.cos().mean(dim=-3)   # (T, M, knots)
        sin_mean = x_t.sin().mean(dim=-3)   # (T, M, knots)

        # |ϕ_N(t) - ϕ_0(t)|² where ϕ_0(t) = exp(-t²/2) for N(0,1)
        err = (cos_mean - self.phi).square() + sin_mean.square()

        # Integrate with trapezoidal weights, scale by batch size
        B = z.size(-2)
        statistic = (err @ self.weights) * B  # (T, M)

        return statistic.mean()  # average over projections and timesteps


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

        # JEPA latent predictor: concat(h, agg); action already in h via encode_latent (no duplicate path)
        self.latent_predictor = _mlp(hidden * 2, hidden, hidden, nn.ReLU)

        # Output decoder: предсказываем x_i из h_i + агрегированных сообщений
        self.out_dec = _mlp(hidden * 2, hidden, 1, nn.ReLU)
        
        # JEPA Target Encoder (EMA from node_enc after each train step)
        self.target_enc = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
        )

        # Xavier initialization
        for m in [self.node_enc, self.action_enc, self.msg_fn, self.latent_predictor, self.out_dec]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.5)
                    nn.init.zeros_(layer.bias)

        # Target encoder: same init as node_enc, then frozen (EMA-updated from node_enc)
        for layer in self.target_enc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
                nn.init.zeros_(layer.bias)
        self.target_enc.load_state_dict(self.node_enc.state_dict())
        for p in self.target_enc.parameters():
            p.requires_grad = False

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

    def _message_pass(
        self,
        h: torch.Tensor,
        return_latent: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """h: (B, d, hidden) → (B, d) scalars; latent head uses concat(h, agg) only."""
        B, d, _hd = h.shape
        A = self.W_masked()

        if torch.is_grad_enabled():
            # Dense mode: required for training to pass gradients through A=0
            h_src = h.unsqueeze(1).expand(B, d, d, self.hidden)
            h_dst = h.unsqueeze(2).expand(B, d, d, self.hidden)
            msg = self.msg_fn(torch.cat([h_src, h_dst], dim=-1))
            weights = A.t().unsqueeze(0).unsqueeze(-1)
            agg = (msg * weights).sum(dim=2)
        else:
            # Sparse mode: massive speedup & memory reduction during inference (EIG scoring)
            active_edges = (A.abs() > 1e-4).nonzero(as_tuple=False)
            if active_edges.numel() == 0:
                agg = torch.zeros_like(h)
            else:
                j = active_edges[:, 0]
                i = active_edges[:, 1]
                h_src_sparse = h[:, j, :]  # (B, E, hidden)
                h_dst_sparse = h[:, i, :]  # (B, E, hidden)
                
                msg = self.msg_fn(torch.cat([h_src_sparse, h_dst_sparse], dim=-1))
                weights = A[j, i].unsqueeze(0).unsqueeze(-1)  # (1, E, 1)
                weighted_msg = msg * weights

                agg = torch.zeros_like(h)
                # Scatter add across the node dimension
                idx = i.unsqueeze(0).unsqueeze(2).expand(B, len(i), _hd)
                agg.scatter_add_(1, idx, weighted_msg)

        h_next = torch.cat([h, agg], dim=-1)
        out = self.out_dec(h_next).squeeze(-1)

        if return_latent:
            h_pred = self.latent_predictor(h_next)
            return out, h_pred
        return out

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

    # ── LeWM: Latent encoding + parallel sequence forward ─────────────────────
    def encode_latent(self, X: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Encode state + action into latent node embeddings (before message passing).

        X, a: (B, d) or (B*T, d)
        Returns: (B, d, hidden) — latent embedding per node.
        """
        am = (torch.abs(a).unsqueeze(-1) > 1e-8).float()
        h_a = self.action_enc(a.unsqueeze(-1))
        h = self.node_enc(X.unsqueeze(-1)) + am * h_a
        return h

    def forward_dynamics_seq(
        self, X_seq: torch.Tensor, A_seq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parallel teacher-forcing prediction over sequences (LeWM-style).

        Instead of processing transitions one-by-one:
            for t in range(T): predict(X_t, a_t)
        We reshape → one forward pass → reshape back.

        Args:
            X_seq: (B, T, d) — sequence of observations
            A_seq: (B, T, d) — sequence of actions (sparse, mostly zeros)

        Returns:
            X_pred: (B, T-1, d)  — predicted next states (teacher forcing)
            H_pred: (B, T-1, d, hidden) — predicted next latent states (JEPA)
            Z_flat: (B*T, d*hidden) — flattened latent embeddings for SIGReg
        """
        B, T, d = X_seq.shape

        # 1. Flatten B and T → one big batch for parallel encoding
        X_flat = X_seq.reshape(B * T, d)        # (B*T, d)
        A_flat = A_seq.reshape(B * T, d)        # (B*T, d)

        # 2. Encode all timesteps in one pass → latent node embeddings
        H_flat = self.encode_latent(X_flat, A_flat)  # (B*T, d, hidden)

        # 3. Message pass → predicted next-state scalars and latents for all timesteps
        X_pred_flat, H_pred_flat = self._message_pass(H_flat, return_latent=True)

        # 4. Reshape back
        X_pred_all = X_pred_flat.reshape(B, T, d)    # (B, T, d)
        H_pred_all = H_pred_flat.reshape(B, T, d, self.hidden)

        # 5. Teacher forcing: X_pred[t] predicts X[t+1]
        X_pred = X_pred_all[:, :-1, :]  # (B, T-1, d)
        H_pred = H_pred_all[:, :-1, :, :]  # (B, T-1, d, hidden)

        # 6. Flatten latents for SIGReg (anti-collapse on full batch)
        Z_flat = H_flat.reshape(B * T, d * self.hidden)

        return X_pred, H_pred, Z_flat

    def dag_constraint(self) -> torch.Tensor:
        """
        Оптимизированный DAG constraint через Taylor expansion (до 4-го порядка).
        Снижает вычислительную нагрузку по сравнению с полным matrix_exp.
        """
        _warn_dag_taylor_order4_if_large(int(self.d))
        M = self.W_masked() ** 2
        
        # Вычисляем степени матрицы для следа
        M2 = torch.matmul(M, M)
        M3 = torch.matmul(M2, M)
        M4 = torch.matmul(M3, M)
        
        # h(W) ≈ tr(M^2)/2! + tr(M^3)/3! + tr(M^4)/4!
        # tr(I) и tr(M) опущены, так как они сокращаются с -d и равны 0 соответственно.
        trace_sum = (M2.trace() / 2.0) + (M3.trace() / 6.0) + (M4.trace() / 24.0)
        return trace_sum

    def dag_constraint_masked(self, free_mask: torch.Tensor) -> torch.Tensor:
        """
        Фаза 1: Оптимизированный DAG-штраф с учетом маски свободных параметров.
        Использует разложение Тейлора для ускорения расчета.
        """
        _warn_dag_taylor_order4_if_large(int(self.d))
        # Применяем маску к весам перед возведением в квадрат
        Wm = self.W_masked() * free_mask
        M = Wm ** 2
        
        M2 = torch.matmul(M, M)
        M3 = torch.matmul(M2, M)
        M4 = torch.matmul(M3, M)
        
        trace_sum = (M2.trace() / 2.0) + (M3.trace() / 6.0) + (M4.trace() / 24.0)
        return trace_sum

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

    @torch.no_grad()
    def update_target_encoder(self, tau: float = 0.006) -> None:
        """
        EMA update: target_enc ← (1−τ)·target_enc + τ·node_enc (JEPA / BYOL-style).
        τ is small (e.g. 0.004–0.01); override via RKK_JEPA_EMA_TAU.
        """
        tau = float(max(0.0, min(1.0, tau)))
        for t_p, s_p in zip(self.target_enc.parameters(), self.node_enc.parameters()):
            t_p.data.mul_(1.0 - tau).add_(s_p.data, alpha=tau)

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
        try:
            new_core.latent_predictor.load_state_dict(self.latent_predictor.state_dict(), strict=True)
        except Exception:
            for layer in new_core.latent_predictor:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.5)
                    nn.init.zeros_(layer.bias)
        new_core.out_dec.load_state_dict(self.out_dec.state_dict())
        # Preserve slow target branch (EMA history); do not reset to fresh node_enc
        try:
            new_core.target_enc.load_state_dict(self.target_enc.state_dict(), strict=True)
        except Exception:
            new_core.target_enc.load_state_dict(new_core.node_enc.state_dict())
        for p in new_core.target_enc.parameters():
            p.requires_grad = False

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
