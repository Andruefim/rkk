"""
temporal_world_model.py — Level 2-F: Temporal World Model (RSSM-lite).

Проблема: GNN rollout использует только текущий X_t для предсказания X_{t+1}.
Это memoryless — не учитывает инерцию тела, историю падений, фазу походки.

RSSM-lite (Recurrent State Space Model, упрощённый):
  h_t = GRU(h_{t-1}, concat(X_t, a_t))    ← детерминированный hidden
  X_{t+1} ≈ decode(h_t)                   ← предсказание следующего состояния

Преимущества vs текущего GNN rollout:
  - Imagination 4 → 20+ шагов без накопления ошибки
  - Motor cortex может планировать шаги наперёд
  - Агент «помнит» что нога была в воздухе 3 тика назад

Интеграция:
  - RSSMLiteCore заменяет integrate_world_model_step() когда включён
  - causal_graph.train_step() дополнительно обновляет GRU через TBPTT
  - agent.score_interventions(): imagination horizon 4→20 шагов через RSSM

Совместимость:
  - Все интерфейсы NOTEARSCore/CausalGNNCore сохранены
  - .W nn.Parameter сохранён для EIG/gradient-based exploration
  - RKK_WM_RSSM=1 включает; RKK_WM_RSSM=0 → прежний путь

RKK_WM_RSSM=1                   — включить (default 0, пока Motor Cortex не обучен)
RKK_WM_RSSM_HIDDEN=64           — размер GRU hidden
RKK_WM_RSSM_IMAGINATION=12      — горизонт imagination (шаги)
RKK_WM_RSSM_TBPTT=8             — TBPTT unroll length
RKK_WM_RSSM_LR=3e-4             — learning rate GRU (отдельный от Adam GNN)
"""
from __future__ import annotations

import os
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def rssm_enabled() -> bool:
    return os.environ.get("RKK_WM_RSSM", "0").strip().lower() in ("1", "true", "yes", "on")


def _env_int(key: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(key, str(default))))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


# ── RSSM-lite Core ────────────────────────────────────────────────────────────
class RSSMLiteCore(nn.Module):
    """
    Recurrent State Space Model (lite variant) для world model.

    Architecture:
      input_encoder: [X_t; a_t] (2d) → z_t (rssm_hidden)
      gru:           (h_{t-1}, z_t) → h_t
      state_decoder: h_t → X_{t+1} (d)

    Interface compatible with CausalGNNCore:
      .W       — causal adjacency (d,d), kept for EIG
      .forward_dynamics(X, a) → X_next
      .W_masked()
      .dag_constraint()
      .alpha_trust_matrix()
      .l1_reg()
    """

    def __init__(self, d: int, device: torch.device, hidden: int = 64, gnn_core=None):
        super().__init__()
        self.d = d
        self.device = device
        self.hidden = hidden

        # Keep causal adjacency W from GNN (for EIG, alpha_trust, etc.)
        self.W = nn.Parameter(torch.zeros(d, d, device=device))
        mask = 1.0 - torch.eye(d, device=device)
        self.register_buffer("mask", mask)

        # RSSM components
        rssm_h = _env_int("RKK_WM_RSSM_HIDDEN", 64)
        self.rssm_hidden = rssm_h

        # Input encoder: [X; a] → z
        self.input_enc = nn.Sequential(
            nn.Linear(d * 2, rssm_h),
            nn.Tanh(),
        )

        # Recurrent core
        self.gru = nn.GRUCell(rssm_h, rssm_h)

        # State decoder: h → X_next
        self.state_dec = nn.Sequential(
            nn.Linear(rssm_h, rssm_h),
            nn.Tanh(),
            nn.Linear(rssm_h, d),
        )

        # Action encoder for quick GNN-style forward (used when h not available)
        self.action_enc = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
        )
        self.node_enc = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
        )
        self.out_dec_quick = nn.Sequential(
            nn.Linear(16 * 2 + d, d),
        )

        # Initialize
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.4)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.to(device)

        # Hidden state buffer (per-agent, maintained externally)
        # Not stored as nn.Parameter to avoid optimizer issues
        self._h: torch.Tensor = torch.zeros(1, rssm_h, device=device)

        # Import GNN weights if provided
        if gnn_core is not None:
            self._import_gnn_weights(gnn_core)

    def _import_gnn_weights(self, gnn_core) -> None:
        """Copy W matrix from existing GNN core."""
        try:
            with torch.no_grad():
                self.W.copy_(gnn_core.W.detach().to(self.device))
            print(f"[RSSM] Imported W from GNN (d={self.d})")
        except Exception as e:
            print(f"[RSSM] W import failed: {e}")

    def reset_hidden(self) -> None:
        """Reset GRU hidden state (call after environment reset)."""
        self._h = torch.zeros(1, self.rssm_hidden, device=self.device)

    def detach_hidden(self) -> None:
        """Detach hidden state from computation graph (TBPTT)."""
        self._h = self._h.detach()

    # ── GNN-compatible interface ────────────────────────────────────────────────
    def W_masked(self) -> torch.Tensor:
        return self.W * self.mask

    def dag_constraint(self) -> torch.Tensor:
        M = self.W_masked() ** 2
        M2 = torch.matmul(M, M)
        M3 = torch.matmul(M2, M)
        M4 = torch.matmul(M3, M)
        return (M2.trace() / 2.0) + (M3.trace() / 6.0) + (M4.trace() / 24.0)

    def dag_constraint_masked(self, free_mask: torch.Tensor) -> torch.Tensor:
        Wm = self.W_masked() * free_mask
        M = Wm ** 2
        M2 = torch.matmul(M, M)
        M3 = torch.matmul(M2, M)
        M4 = torch.matmul(M3, M)
        return (M2.trace() / 2.0) + (M3.trace() / 6.0) + (M4.trace() / 24.0)

    def l1_reg(self) -> torch.Tensor:
        return self.W_masked().abs().sum()

    def alpha_trust_matrix(self) -> torch.Tensor:
        W_abs = self.W_masked().abs()
        return W_abs / W_abs.max() if W_abs.max() > 0 else W_abs

    def intervention_loss(self, X_obs, X_int, int_var_idx, int_val):
        a = torch.zeros_like(X_obs)
        a[:, int_var_idx] = int_val
        predicted = self.forward_dynamics(X_obs, a)
        return F.mse_loss(predicted, X_int)

    # ── Core forward dynamics ──────────────────────────────────────────────────
    def forward_dynamics(self, X: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        World model step with recurrent state.

        X: (B, d) current state
        a: (B, d) action (sparse, do(var)=val)
        Returns: X_next (B, d)

        Note: maintains internal hidden state self._h.
        For batched imagination rollouts, use forward_dynamics_stateless().
        """
        B = X.shape[0]

        # Expand hidden state to batch size if needed
        if self._h.shape[0] != B:
            h = self._h.expand(B, -1).contiguous()
        else:
            h = self._h

        # Encode input
        xa = torch.cat([X, a], dim=1)  # (B, 2d)
        z = self.input_enc(xa)          # (B, rssm_h)

        # GRU step
        h_next = self.gru(z, h)         # (B, rssm_h)

        # Update internal state (use first batch element for consistency)
        self._h = h_next[:1].detach()

        # Decode next state
        X_next = self.state_dec(h_next)  # (B, d)
        return X_next

    def forward_dynamics_stateless(
        self,
        X: torch.Tensor,
        a: torch.Tensor,
        h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Stateless version for imagination rollouts.
        Returns (X_next, h_next) without updating self._h.
        """
        B = X.shape[0]
        if h is None:
            h = torch.zeros(B, self.rssm_hidden, device=self.device)
        elif h.shape[0] != B:
            h = h.expand(B, -1).contiguous()

        xa = torch.cat([X, a], dim=1)
        z = self.input_enc(xa)
        h_next = self.gru(z, h)
        X_next = self.state_dec(h_next)
        return X_next, h_next

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Passive step (no action) for compatibility."""
        return self.forward_dynamics(X, torch.zeros_like(X))

    def snapshot_info(self) -> dict:
        return {
            "type": "rssm_lite",
            "d": self.d,
            "hidden": self.hidden,
            "rssm_hidden": self.rssm_hidden,
            "params": sum(p.numel() for p in self.parameters()),
        }


# ── Long-horizon imagination ──────────────────────────────────────────────────
class RSSMImagination:
    """
    Long-horizon imagination rollouts using RSSM.

    Используется в agent.score_interventions() и motor cortex planning.
    Заменяет CausalGraph.rollout_step_free() для RSSM.
    """

    def __init__(self, rssm: RSSMLiteCore, device: torch.device):
        self.rssm = rssm
        self.device = device

    @torch.no_grad()
    def rollout(
        self,
        initial_state: dict[str, float],
        node_ids: list[str],
        actions: list[tuple[str, float]],  # sequence of (var, val)
        horizon: int | None = None,
    ) -> list[dict[str, float]]:
        """
        Multi-step imagination with optional action sequence.

        initial_state: current graph nodes
        node_ids: ordered list of variable names
        actions: sequence of (var, val) or empty for free rollout
        horizon: total steps (defaults to RKK_WM_RSSM_IMAGINATION)

        Returns: list of state dicts for each step
        """
        if horizon is None:
            horizon = _env_int("RKK_WM_RSSM_IMAGINATION", 12)

        d = len(node_ids)
        x0 = torch.tensor(
            [[float(initial_state.get(n, 0.0)) for n in node_ids]],
            dtype=torch.float32, device=self.device,
        )

        # Start with current RSSM hidden state
        h = self.rssm._h.expand(1, -1).contiguous()

        states: list[dict[str, float]] = []
        x = x0

        for step in range(horizon):
            # Build action vector
            a = torch.zeros(1, d, device=self.device)
            if step < len(actions):
                var, val = actions[step]
                if var in node_ids:
                    a[0, node_ids.index(var)] = float(val)

            x_next, h = self.rssm.forward_dynamics_stateless(x, a, h)
            state_dict = {
                node_ids[i]: float(x_next[0, i].item())
                for i in range(d)
            }
            states.append(state_dict)
            x = x_next

        return states

    @torch.no_grad()
    def evaluate_action_sequence(
        self,
        initial_state: dict[str, float],
        node_ids: list[str],
        candidate_actions: list[tuple[str, float]],
        target_var: str,
        target_direction: str = "minimize",  # "minimize" or "maximize"
        horizon: int = 8,
    ) -> float:
        """
        Score a single action (or short sequence) over multi-step horizon.
        Returns normalized score [0, 1].
        """
        states = self.rollout(initial_state, node_ids, candidate_actions, horizon)
        if not states or target_var not in states[0]:
            return 0.5

        values = [s.get(target_var, 0.5) for s in states]
        mean_val = float(np.mean(values))

        if target_direction == "minimize":
            return float(np.clip(1.0 - mean_val, 0.0, 1.0))
        else:
            return float(np.clip(mean_val, 0.0, 1.0))


# ── RSSM training integration ─────────────────────────────────────────────────
class RSSMTrainer:
    """
    TBPTT (Truncated Backpropagation Through Time) trainer for RSSM.

    Хранит rolling sequence buffer и раз в TBPTT шагов делает grad step.
    Интегрируется в CausalGraph.train_step() как дополнительный шаг.
    """

    def __init__(self, rssm: RSSMLiteCore, device: torch.device):
        self.rssm = rssm
        self.device = device
        self.tbptt = _env_int("RKK_WM_RSSM_TBPTT", 8)
        self.lr = _env_float("RKK_WM_RSSM_LR", 3e-4)

        # Separate optimizer for RSSM (don't touch GNN's Adam)
        self.optim = torch.optim.Adam(
            list(rssm.input_enc.parameters())
            + list(rssm.gru.parameters())
            + list(rssm.state_dec.parameters()),
            lr=self.lr,
            weight_decay=1e-5,
        )

        # Sequence buffer: list of (X_t, a_t, X_{t+1})
        self._seq_buf: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.train_steps: int = 0
        self.mean_loss: float = 0.0
        self._loss_history: deque[float] = deque(maxlen=50)

    def push(
        self,
        X_t: list[float],
        a_t: list[float],
        X_tp1: list[float],
    ) -> None:
        """Add one transition to sequence buffer."""
        def t(x): return torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
        self._seq_buf.append((t(X_t), t(a_t), t(X_tp1)))

    def maybe_train(self) -> float | None:
        """Train RSSM if sequence buffer has TBPTT steps."""
        if len(self._seq_buf) < self.tbptt:
            return None

        # Take TBPTT steps from buffer
        batch = self._seq_buf[-self.tbptt:]
        self._seq_buf = self._seq_buf[-(self.tbptt * 2):]  # keep some history

        self.optim.zero_grad()
        h = torch.zeros(1, self.rssm.rssm_hidden, device=self.device)
        total_loss = torch.tensor(0.0, device=self.device)

        d_rm = int(self.rssm.d)
        for X_t, a_t, X_tp1 in batch:
            X_s = X_t[..., :d_rm]
            a_s = a_t[..., :d_rm]
            tgt = X_tp1[..., :d_rm]
            X_pred, h = self.rssm.forward_dynamics_stateless(X_s, a_s, h)
            loss_t = F.mse_loss(X_pred, tgt.detach())
            total_loss = total_loss + loss_t

        total_loss = total_loss / self.tbptt
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.rssm.input_enc.parameters())
            + list(self.rssm.gru.parameters())
            + list(self.rssm.state_dec.parameters()),
            max_norm=0.5,
        )
        self.optim.step()

        # Update internal hidden to match trained sequence
        self.rssm.detach_hidden()

        v = float(total_loss.item())
        self._loss_history.append(v)
        self.mean_loss = float(np.mean(self._loss_history))
        self.train_steps += 1
        return v

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": rssm_enabled(),
            "train_steps": self.train_steps,
            "mean_loss": round(self.mean_loss, 5),
            "seq_buf_len": len(self._seq_buf),
            "tbptt": self.tbptt,
            "rssm_hidden": self.rssm.rssm_hidden,
            "imagination_horizon": _env_int("RKK_WM_RSSM_IMAGINATION", 12),
        }


# ── Integration with CausalGraph ──────────────────────────────────────────────
def maybe_upgrade_graph_to_rssm(graph, device: torch.device) -> tuple[bool, Any]:
    """
    Upgrade existing CausalGraph._core to RSSMLiteCore.
    Preserves W matrix from GNN.
    Returns (upgraded: bool, trainer: RSSMTrainer | None)
    """
    if not rssm_enabled():
        return False, None

    core = graph._core
    if core is None:
        return False, None

    # Already RSSM
    base = getattr(core, "_orig_mod", core)
    if isinstance(base, RSSMLiteCore):
        return True, None

    d = graph._d
    print(f"[RSSM] Upgrading GNN (d={d}) to RSSM-lite hidden={_env_int('RKK_WM_RSSM_HIDDEN', 64)}")

    rssm = RSSMLiteCore(d=d, device=device, gnn_core=core)
    trainer = RSSMTrainer(rssm, device)

    # Replace core and optim in graph
    graph._core = rssm
    graph._optim = torch.optim.Adam(rssm.parameters(), lr=5e-3)
    graph._invalidate_cache()

    return True, trainer


def integrate_world_model_step_rssm(
    core,
    y0: torch.Tensor,
    a: torch.Tensor,
) -> torch.Tensor:
    """
    Drop-in replacement for integrate_world_model_step when RSSM is active.
    Automatically falls back to GNN if not RSSM.
    """
    if not rssm_enabled():
        from engine.wm_neural_ode import integrate_world_model_step
        return integrate_world_model_step(core, y0, a)

    base = getattr(core, "_orig_mod", core)
    if isinstance(base, RSSMLiteCore):
        return base.forward_dynamics(y0, a)

    # Fallback to standard
    from engine.wm_neural_ode import integrate_world_model_step
    return integrate_world_model_step(core, y0, a)
