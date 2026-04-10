"""
temporal.py — Phase-Coupled Temporal Blankets (РКК v5, Фаза 8).

Архитектура: два параллельных SSM с разными временными константами.

  Fast SSM (dt=0.01):  реакции на каждый do()-эксперимент
                       → моторный уровень: "что сейчас происходит?"

  Slow SSM (dt=1.0):   интегрирует быстрые события
                       → планирующий уровень: "что происходит в целом?"

Связь:
  Fast → Slow: передаём усреднённый интеграл каждые WINDOW шагов
  Slow → Fast: slow-состояние как граничное условие (константа контекста)

Математика (дискретный SSM):
  h_t = A_disc · h_{t-1} + B · u_t
  y_t = C · h_t
  A_disc = exp(A * dt)  — матричная экспонента (дискретизация ZOH)

Φ_approx (автономия):
  Φ = var(h_fast) / (var(h_fast) + var(external_input) + ε)
  
  Интерпретация: насколько динамика агента определяется его внутренним
  состоянием, а не внешними воздействиями (интервенциями Демона).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from collections import deque


# ─── MinimalSSM ───────────────────────────────────────────────────────────────
class MinimalSSM(nn.Module):
    """
    Минимальный State Space Model — работает на любом PyTorch backend (HIP/CUDA/CPU).

    State equation:  h_t = A_disc · h_{t-1} + B · u_t
    Output equation: y_t = C · h_t + D · u_t
    """

    def __init__(self, d_input: int, d_state: int, dt: float, device: torch.device):
        super().__init__()
        self.d_state = d_state
        self.d_input = d_input
        self.dt      = dt
        self.device  = device

        # Обучаемые матрицы
        # A: диагональная (для стабильности инициализируем отрицательной)
        self.A_log = nn.Parameter(torch.randn(d_state, device=device) * 0.1 - 1.0)
        self.B     = nn.Parameter(torch.randn(d_state, d_input, device=device) * 0.1)
        self.C     = nn.Parameter(torch.randn(d_input, d_state, device=device) * 0.1)
        self.D     = nn.Parameter(torch.zeros(d_input, d_input, device=device))

        # Начальное состояние
        self.h0 = nn.Parameter(torch.zeros(d_state, device=device))

    def A_disc(self) -> torch.Tensor:
        """Дискретизация: A_disc = diag(exp(A * dt))."""
        return torch.exp(self.A_log * self.dt)   # (d_state,)

    def step(self, u: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Один шаг SSM.
        u: (d_input,)  — входной сигнал
        h: (d_state,)  — предыдущее состояние
        Возвращает: (y, h_new)
        """
        A_d   = self.A_disc()                     # (d_state,)
        h_new = A_d * h + self.B @ u              # (d_state,)
        y     = self.C @ h_new + self.D @ u       # (d_input,)
        return y, h_new

    def forward(self, U: torch.Tensor, h0: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Батчевый прогон.
        U: (T, d_input)  — последовательность входов
        Возвращает: Y (T, d_input), h_final (d_state,)
        """
        T      = U.shape[0]
        h      = h0 if h0 is not None else self.h0.clone()
        outputs = []

        for t in range(T):
            y, h = self.step(U[t], h)
            outputs.append(y)

        return torch.stack(outputs, dim=0), h   # (T, d_input), (d_state,)


# ─── TemporalBlankets ─────────────────────────────────────────────────────────
class TemporalBlankets:
    """
    Два SSM (fast/slow) с phase-coupled интеграцией.

    Fast SSM: обновляется каждый do()-шаг
    Slow SSM: обновляется каждые WINDOW fast-шагов

    API:
      tb = TemporalBlankets(d_input=6, device=...)
      context = tb.step(observation_vec)  → (d_input,) — контекст для агента
      phi     = tb.phi_approx()           → float ∈ [0,1]
    """

    WINDOW     = 10    # fast-шагов на один slow-шаг
    D_STATE_F  = 16    # размер состояния fast SSM
    D_STATE_S  = 32    # размер состояния slow SSM

    def __init__(self, d_input: int, device: torch.device):
        self.d_input = d_input
        self.device  = device

        # Fast SSM: маленькое состояние, малый dt
        self.fast = MinimalSSM(d_input, self.D_STATE_F, dt=0.01, device=device)

        # Slow SSM: большое состояние, большой dt
        self.slow = MinimalSSM(d_input, self.D_STATE_S, dt=1.0,  device=device)

        # Fast → Slow интегратор (усредняем fast-выходы)
        self.integrator = nn.Linear(d_input, d_input, device=device)

        # Slow → Fast контекст
        self.context_proj = nn.Linear(d_input, d_input, device=device)

        # Оптимизатор для всех параметров
        all_params = (
            list(self.fast.parameters())
            + list(self.slow.parameters())
            + list(self.integrator.parameters())
            + list(self.context_proj.parameters())
        )
        self.optim = torch.optim.Adam(all_params, lr=1e-3)

        # Состояния
        self.h_fast = torch.zeros(self.D_STATE_F, device=device)
        self.h_slow = torch.zeros(self.D_STATE_S, device=device)
        self.slow_context = torch.zeros(d_input, device=device)

        # Буферы
        self._fast_output_buf: list[torch.Tensor] = []  # для интеграции в slow
        self._fast_input_buf:  list[torch.Tensor] = []  # для Φ_approx
        self._slow_output_buf: list[torch.Tensor] = []

        self._step_count = 0

        # История для Φ_approx
        self._h_fast_history: deque[torch.Tensor] = deque(maxlen=50)
        self._u_history:      deque[torch.Tensor] = deque(maxlen=50)

    def step(self, observation: dict[str, float] | torch.Tensor) -> torch.Tensor:
        """
        Один шаг TemporalBlankets.

        observation: dict узлов или тензор (d_input,)
        Возвращает: контекстный вектор (d_input,) для использования агентом
        """
        if isinstance(observation, dict):
            vals = list(observation.values())[:self.d_input]
            # Паддинг если переменных меньше d_input
            while len(vals) < self.d_input:
                vals.append(0.0)
            u = torch.tensor(vals, dtype=torch.float32, device=self.device)
        else:
            u = observation.to(self.device).reshape(-1)
            if u.numel() > self.d_input:
                u = u[: self.d_input]
            elif u.numel() < self.d_input:
                u = torch.nn.functional.pad(u, (0, self.d_input - u.numel()))

        # ── Fast step ──
        # Контекст от slow SSM добавляется как граничное условие
        u_with_context = u + 0.1 * self.slow_context.detach()

        y_fast, self.h_fast = self.fast.step(u_with_context, self.h_fast.detach())
        self._fast_output_buf.append(y_fast.detach())
        self._fast_input_buf.append(u.detach())
        self._h_fast_history.append(self.h_fast.detach())
        self._u_history.append(u.detach())

        self._step_count += 1

        # ── Slow step (каждые WINDOW fast-шагов) ──
        if len(self._fast_output_buf) >= self.WINDOW:
            # Интеграл fast-выходов → один slow-вход
            fast_stack    = torch.stack(self._fast_output_buf, dim=0)  # (W, d)
            integrated    = self.integrator(fast_stack.mean(dim=0))     # (d,)

            y_slow, self.h_slow = self.slow.step(integrated, self.h_slow.detach())
            self._slow_output_buf.append(y_slow.detach())

            # Контекст slow → fast
            self.slow_context = self.context_proj(y_slow).detach()

            self._fast_output_buf.clear()

        return y_fast.detach()

    def phi_approx(self) -> float:
        """
        Φ_approx: автономия через соотношение дисперсий.

        Φ = var(h_fast) / (var(h_fast) + var(u) + ε)

        Интерпретация:
          Φ → 1: динамика определяется внутренним состоянием (высокая автономия)
          Φ → 0: динамика определяется внешними входами (низкая автономия)
        """
        if len(self._h_fast_history) < 5:
            return 0.1

        # Дисперсия внутреннего состояния
        h_stack  = torch.stack(list(self._h_fast_history), dim=0)  # (T, d_state)
        var_internal = h_stack.var(dim=0).mean().item()

        # Дисперсия внешних входов
        if len(self._u_history) < 2:
            return 0.1
        u_stack   = torch.stack(list(self._u_history), dim=0)       # (T, d_input)
        var_external = u_stack.var(dim=0).mean().item()

        phi = var_internal / (var_internal + var_external + 1e-8)
        return float(np.clip(phi, 0.0, 1.0))

    def slow_state_summary(self) -> dict:
        """Сводка для UI."""
        return {
            "fast_steps":     self._step_count,
            "slow_steps":     len(self._slow_output_buf),
            "phi":            round(self.phi_approx(), 4),
            "h_fast_norm":    round(self.h_fast.norm().item(), 4),
            "h_slow_norm":    round(self.h_slow.norm().item(), 4),
            "context_norm":   round(self.slow_context.norm().item(), 4),
        }

    def train_step(self, target_obs: torch.Tensor) -> float | None:
        """
        Обучаем SSM предсказывать следующее наблюдение.
        Вызывается опционально из агента.
        """
        if len(self._fast_input_buf) < 2:
            return None

        # Берём последние 2 входа: предсказываем u_t из u_{t-1}
        u_prev = self._fast_input_buf[-2] if len(self._fast_input_buf) >= 2 else None
        if u_prev is None:
            return None

        self.optim.zero_grad()
        y_pred, _ = self.fast.step(u_prev, self.h_fast.detach())
        target = target_obs.to(self.device).reshape(-1)
        d = self.d_input
        if target.numel() != d:
            if target.numel() > d:
                target = target[:d]
            else:
                target = torch.nn.functional.pad(target, (0, d - target.numel()))
        loss = nn.functional.mse_loss(y_pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.fast.parameters(), 1.0)
        self.optim.step()
        return loss.item()
